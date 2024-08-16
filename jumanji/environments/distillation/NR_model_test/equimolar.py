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
    state = initial_composition.converge_temperature(state)
    state = functions.y_func(state)
    '''
    state = functions.y_func(state)
    
    state = state.replace(
        Hliq=jnp.sum(vmap(thermo.liquid_enthalpy, in_axes=(0, None))(state.temperature,
                                                                               state.components).transpose() * state.X,
                     axis=0),
        Hvap=jnp.sum(vmap(thermo.vapor_enthalpy, in_axes=(0, None))(state.temperature,
                                                                              state.components).transpose() * state.Y,
                     axis=0)
    )
    state = flowrates(state)
    '''
    return state


def x_initial(state: State):
    tray_low, tray_high, tray = matrix_transforms.trays_func(state)
    a, b, c = jacobian.g_jacobian_func(state, tray_low, tray_high, tray)
    g = vmap(g_sol, in_axes=(None, None, None, None, 0))(state, tray_low, tray,
                                                             tray_high,
                                                             jnp.arange(len(tray.T)))
    dx = jnp.nan_to_num(functions.thomas(a, b, c, -1 * g, state.Nstages))  # .reshape(-1,1)

    t = jnp.array(1.0, dtype=float)
    dx_v = dx[:, :len(state.components)].transpose()
    dx_l = dx[:, -len(state.components):].transpose()
    dx_t = dx[:, len(state.components)].transpose()

    v_new = (tray.v + t * dx_v)
    l_new = (tray.l + t * dx_l)
    t_new = (tray.T + t * dx_t)

    v_new_final = jnp.where(v_new >= 0., v_new, tray.v
                            * jnp.exp(t * dx_v / jnp.where(tray.v > 0, tray.v, 1e-10)))
    l_new_final = jnp.where(l_new >= 0., l_new, tray.l
                            * jnp.exp(t * dx_l / jnp.where(tray.l > 0, tray.l, 1e-10)))
    t_new_final = jnp.where(t_new >= state.temperature_bounds[-1]*1.01, state.temperature_bounds[-1]*1.01,
                            jnp.where(t_new <= state.temperature_bounds[0]*0.99, state.temperature_bounds[0]*0.99, t_new)) * jnp.where(tray.T > 0, 1, 0)

    #v_new_final = jnp.where(v_new_final > jnp.sum(tray.v, axis=0), jnp.sum(tray.v), v_new_final)
    #l_new_final = jnp.where(l_new_final > jnp.sum(tray.l, axis=0), jnp.sum(tray.l), l_new_final)
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
    state = state.replace(EQU_residuals=jnp.nan_to_num(jnp.sum(g ** 2), nan=1e3))

    return state


def cond_fn(state):
    comps = jnp.sum(jnp.where(state.z > 0, 1, 0))
    cond = state.Nstages * (2 * comps + 1) * jnp.sum(state.F) * 1e-9
    return (state.EQU_iterations < 100) & (state.EQU_residuals > cond)


def body_fn(state):
    state = x_initial(state)
    state = state.replace(EQU_iterations=state.EQU_iterations+1)
    return state


def converge_equimolar(state: State):
    # nr_state = initialize_NR(state)x

    state = (
        lax.while_loop(cond_fun=cond_fn, body_fun=body_fn,
                       init_val=state
                       )
    )

    return state


def scan_fn(state, _):
    state = x_initial(state)
    state = nr_state_new.replace(EQU_iterations=state.EQU_iterations+1)
    return state, None


def scan_equimolar(state: State):
    state, _ = lax.scan(f=scan_fn,
             init=state,
             xs=jnp.arange(20)
             )
    return state


def flowrates(state: State):
    vnew = jnp.zeros(len(state.V))
    vnew = vnew.at[0].set(state.distillate)
    vnew = vnew.at[1].set(state.V[0] + state.L[0] + state.U[0] - state.F[0])

    def update_vnew(carry, j):
        vnew, state = carry
        mask = jnp.where(jnp.arange(len(state.temperature)) < j, 1, 0)
        numerator = ((jnp.sum(state.F * mask - state.U * mask - state.W * mask) - state.V[0]) * (
                    state.Hliq[j - 1] - state.Hliq[j]) - state.F[j] * (
                             state.Hliq[j] - state.Hfeed[j])
                     + state.W[j] * (state.Hvap[j] - state.Hliq[j])) - (state.Hvap[j] - state.Hliq[j - 1]) * vnew[j]
        denominator = (state.Hliq[j] - state.Hvap[j + 1])
        vnew = vnew.at[j + 1].set(numerator / denominator)
        carry = vnew, state
        return carry, None

    carry_vnew, _ = jax.lax.scan(update_vnew, (vnew, state), jnp.arange(1, len(state.V)))
    vnew, state = carry_vnew

    lnew = jnp.zeros(len(state.L))

    # lnew = lnew.at[0].set(state.RR * state.U[0])
    # lnew = lnew.at[-1].set(state.bottom)

    def update_lnew(carry, j):
        lnew, state = carry
        mask = jnp.where(jnp.arange(len(state.temperature)) < j+1, 1, 0)
        lnew = lnew.at[j].set(vnew[j + 1] + jnp.sum(state.F * mask - state.U * mask - state.W * mask) - vnew[0])
        carry = lnew, state
        return carry, j

    # lnew = vmap(update_lnew, in_axes=[0, None])(jnp.arange(1, len(state.L)-1), vnew)
    carry_lnew, add = jax.lax.scan(update_lnew, (lnew, state), jnp.arange(1, len(state.L) - 1))
    lnew, state = carry_lnew

    mask = jnp.arange(len(state.V))
    return state.replace(L=jnp.where((mask > 0) & (mask < state.Nstages-1), lnew, state.L), V=jnp.where((mask > 1) & (mask < state.Nstages), vnew, state.V))
