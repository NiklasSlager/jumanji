import jax.numpy as jnp
import jax
from jax import vmap, lax, jit
from jumanji.environments.distillation.NR_model_test.distillation_types import State, NR_State, Trays, Tray, Thermo
from jumanji.environments.distillation.NR_model_test import functions, initial_composition, matrix_transforms, jacobian, \
    thermodynamics, costing, purity_constraint
import os

# from NR_model_test.analytic_jacobian import jacobian as pure_jac
# from NR_model_test import analytic_jacobian
import os



def x_initial(state: State):
    a, b, c = jacobian.g_jacobian_func(state)
    g = vmap(g_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                         state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    dx = jnp.nan_to_num(functions.thomas(a, b, c, -1 * g, state.Nstages))
    def min_res(t, state, dx):
        dx_v = dx[:, :len(state.components)].transpose()
        dx_l = dx[:, -len(state.components):].transpose()
        dx_t = dx[:, len(state.components)].transpose()

        v_new = (state.trays.tray.v + t *  dx_v)
        l_new = (state.trays.tray.l + t * dx_l)
        t_new = (state.trays.tray.T + t * dx_t)
        max_t = functions.t_sat_solver(state.components, state.pressure)
        max_t = jnp.where(state.z > 0, max_t, 0)
        min_t = jnp.min(jnp.where((max_t >0) & (max_t != jnp.max(max_t)), max_t, jnp.max(max_t)-1))
        v_new_final = jnp.where(v_new >= 0., v_new, state.trays.tray.v
                                * jnp.exp(t * dx_v / jnp.where(state.trays.tray.v > 0, state.trays.tray.v, 1e30)))
        l_new_final = jnp.where(l_new >= 0., l_new, state.trays.tray.l
                                * jnp.exp(t * dx_l / jnp.where(state.trays.tray.l > 0, state.trays.tray.l, 1e30)))
        t_new_final = jnp.where(t_new >= jnp.max(max_t),
                                jnp.max(max_t),
                                jnp.where(t_new <= jnp.min(min_t), jnp.min(min_t), t_new)
                                )*jnp.where(state.temperature>0, 1, 0)
        state = state.replace(
            Y=jnp.nan_to_num(v_new_final / jnp.sum(v_new_final, axis=0)),
            X=jnp.nan_to_num(l_new_final / jnp.sum(l_new_final, axis=0)),
            temperature=t_new_final,
        )

        state = matrix_transforms.trays_func(state)

        g = vmap(g_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                                 state.trays.high_tray,
                                                                 jnp.arange(len(state.trays.tray.T)))

        return jnp.nan_to_num(jnp.sum(g ** 2), nan=1e10), state

    carry = vmap(min_res, in_axes=(0, None, None))(jnp.arange(0.7, 1.3, 0.1), state, dx)
    result, states = carry
    new_t = jnp.max(jnp.where(result == jnp.min(result), jnp.arange(0.7, 1.3, 0.1), 0))

    res, state_new = min_res(new_t, state, dx)
    return res, state_new


def cond_fn(args):
    state, iterations, res = args
    comps = jnp.sum(jnp.where(state.z > 0, 1, 0))
    cond = state.Nstages * (2 * comps + 1) * jnp.sum(state.F) * 1e-9
    return (iterations < 100) & (res > cond)


def body_fn(args):
    state, iterations, res = args
    res_new, nr_state_new = x_initial(state)
    nr_state_new = nr_state_new.replace(residuals=nr_state_new.residuals.at[iterations].set(res_new))
    iterations += 1
    return nr_state_new, iterations, res_new


def converge_equimolar(state: State):
    # nr_state = initialize_NR(state)x

    state = matrix_transforms.trays_func(state)
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
