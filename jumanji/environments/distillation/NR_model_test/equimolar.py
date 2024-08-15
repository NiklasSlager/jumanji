import jax.numpy as jnp
import jax
from jax import vmap, lax, jit
from jumanji.environments.distillation.NR_model_test.distillation_types import State, Tray, Thermo
from jumanji.environments.distillation.NR_model_test import functions, initial_composition, matrix_transforms, jacobian, \
    thermodynamics, costing, purity_constraint
import os

# from NR_model_test.analytic_jacobian import jacobian as pure_jac
# from NR_model_test import analytic_jacobian
import os


def g_sol(state: State, tray_low, tray, tray_high, j):
    f_s = jacobian.g_vector_function(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return jnp.concatenate((jnp.asarray(f_s.H)[None], jnp.asarray(f_s.M), jnp.asarray(f_s.E)), axis=0)


def bubble_point(state):
    '''
    def for_body(state, i):
        state = model_solver(state)
        state = functions.stage_temperature(state)
        return state, i

    state, add = jax.lax.scan(for_body, state, jnp.arange(30))
    '''
    #state_init = model_solver(state)
    state, iterators = converge_temperature(state)
    #state = state.replace(X=(state_init.X*5+state.X)/6)
    state = functions.y_func(state)

    return state


def x_initial(state: State):
    tray_low, tray_high, tray = matrix_transforms.trays_func(state)
    a, b, c = jacobian.g_jacobian_func(state, tray_low, tray_high, tray)
    g = vmap(g_sol, in_axes=(None, None, None, None, 0))(state, tray_low, tray,
                                                             tray_high,
                                                             jnp.arange(len(tray.T)))
    dx = jnp.nan_to_num(functions.thomas(a, b, c, -1 * g, state.Nstages))  # .reshape(-1,1)
    def min_res(t, state, tray, dx):
        dx_v = dx[:, :len(state.components)].transpose()
        dx_l = dx[:, -len(state.components):].transpose()
        dx_t = dx[:, len(state.components)].transpose()

        v_new = (tray.v + t *  dx_v)
        l_new = (tray.l + t * dx_l)
        t_new = (tray.T + t * dx_t)

        v_new_final = jnp.where(v_new >= 0., v_new, tray.v
                                * jnp.exp(t * dx_v / jnp.where(tray.v > 0, tray.v, 1e-10)))
        l_new_final = jnp.where(l_new >= 0., l_new, tray.l
                                * jnp.exp(t * dx_l / jnp.where(tray.l > 0, tray.l, 1e-10)))
        t_new_final = jnp.where(t_new >= state.temperature_bounds[-1], state.temperature_bounds[-1],
                                jnp.where(t_new <= state.temperature_bounds[0], state.temperature_bounds[0], t_new)) * jnp.where(tray.T > 0, 1, 0)

        state = state.replace(
            Y=jnp.nan_to_num(v_new_final / jnp.sum(v_new_final, axis=0)),
            X=jnp.nan_to_num(l_new_final / jnp.sum(l_new_final, axis=0)),
            temperature=t_new_final,
        )

        #state = matrix_transforms.trays_func(state)
        tray_low, tray_high, tray = matrix_transforms.trays_func(state)
        g = vmap(g_sol, in_axes=(None, None, None, None, 0))(state, tray_low, tray,
                                                                 tray_high,
                                                                 jnp.arange(len(tray.T)))

        return jnp.nan_to_num(jnp.sum(g ** 2), nan=1e10), state

    carry = vmap(min_res, in_axes=(0, None, None, None))(jnp.arange(0.1, 1.1, 0.1), state, tray, dx)
    result, states = carry
    new_t = jnp.max(jnp.where(result == jnp.min(result), jnp.arange(0.1, 1.1, 0.1), 0))

    res, state_new = min_res(new_t, state, tray, dx)
    return res, state_new


def cond_fn(args):
    state, iterations, res = args
    comps = jnp.sum(jnp.where(state.z > 0, 1, 0))
    cond = state.Nstages * (2 * comps + 1) * jnp.sum(state.F) * 1e-8
    return (iterations < 100) & (res > cond)


def body_fn(args):
    state, iterations, res = args
    res_new, nr_state_new = x_initial(state)
    nr_state_new = nr_state_new.replace(residuals=nr_state_new.residuals.at[iterations].set(res_new))
    iterations += 1
    return nr_state_new, iterations, res_new


def converge_equimolar(state: State):
    # nr_state = initialize_NR(state)x

    iterations = 0
    res = 0

    state, iterations, res = (
        lax.while_loop(cond_fun=cond_fn, body_fun=body_fn,
                       init_val=(state,
                                 iterations,
                                 jnp.array(10, dtype=float),
                                 )
                       )
    )
    return state, iterations, res
