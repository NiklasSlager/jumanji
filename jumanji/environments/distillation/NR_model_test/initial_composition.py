import chex
import jax.numpy as jnp
import jax
from jax import vmap, scipy
from jumanji.environments.distillation.NR_model_test.distillation_types import State
from jumanji.environments.distillation.NR_model_test import functions
from jumanji.environments.distillation.NR_model_test import thermodynamics as thermo


def get_aj(state, j):
    return state.L[j - 1]


def get_bj(state, component, j):
    return -(state.L[j] + state.U[j] + (state.V[j] + state.W[j]) *
             thermo.k_eq(state.temperature[j], component, state.pressure))
             #k_eq(state.temperature[j], state.components[component], state.pressure))


def get_cj(state, component, j):
    return state.V[j + 1] * thermo.k_eq(state.temperature[j + 1], component, state.pressure)
    #return state.V[j + 1] * k_eq(state.temperature[j+1], state.components[component], state.pressure)


def massmatrix(state, component):

    aj_values = vmap(get_aj, in_axes=[None, 0])(state, jnp.arange(1, len(state.L)))
    aj_values = jnp.where(jnp.arange(1, len(state.L)) < state.Nstages, aj_values, 0)

    bj_values = vmap(get_bj, in_axes=[None, None, 0])(state, component, jnp.arange(0, len(state.L)))
    bj_values = jnp.where(jnp.arange(0, len(state.L)) < state.Nstages, bj_values, 1)

    cj_values = vmap(get_cj, in_axes=[None, None, 0])(state, component, jnp.arange(0, len(state.L) - 1))
    cj_values = jnp.where(jnp.arange(0, len(state.L) - 1) < state.Nstages-1, cj_values, 0)
    return jnp.diag(aj_values, k=-1) + jnp.diag(bj_values) + jnp.diag(cj_values, k=1)



def bij(state, j):
    #indexes = jnp.arange(len(state.components))
    #component = jnp.where(j == 2, 0, jnp.where(j == 3, 1, 2))
    component=j
    return -state.z[component] * state.F


def solve_x(state, component):
    def matvec(x):
        return jnp.dot(massmatrix(state, component), x)
    b = bij(state, component)
    #matrix = massmatrix(state, component)
    #solution = lineax.linear_solve(lineax.MatrixLinearOperator(massmatrix(state, component)), b, solver=lineax.QR()).value
    return jax.scipy.linalg.solve(massmatrix(state, component), b)
    #return jnp.where(component > 0, solution, 0)



def model_solver(state: State):
    #components = jnp.where(state.z > 0, state.components, 0)
    x = jnp.abs(vmap(solve_x, in_axes=(None, 0))(state, state.components))
    #x = jnp.ones(len(state.temperature))[:, None]*state.z
    #return state._replace(X=x.transpose())
    return state.replace(X=(x / jnp.sum(x, axis=0)))


def cond_fn_temperature(args):
    state, t_old, tol, iterations = args
    error = jnp.abs(state.temperature - t_old)
    return (iterations < 25) & (jnp.any(error > tol))


def body_fn_temperature(args):
    state, t_old, tol, iterations = args
    t_old = state.temperature
    state = model_solver(state)
    state = functions.stage_temperature(state)
    iterations += 1
    return state, t_old, tol, iterations


def converge_temperature(state: State):
    tol = jnp.array(0.01, dtype=float)
    state, t_old, tol, iterations = jax.lax.while_loop(
        cond_fn_temperature,
        body_fn_temperature,
        (state, jnp.zeros_like(state.temperature, dtype=float), tol, 1)
    )
    return state, iterations



def bubble_point(state):
    state, iterators = converge_temperature(state)
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


def flowrates1(state: State):
    vnew = jnp.zeros(len(state.V))
    vnew = vnew.at[0].set(state.distillate)
    vnew = vnew.at[1].set(state.V[0] + state.L[0] + state.U[0] - state.F[0])

    def update_vnew(vnew, j):
        numerator = ((jnp.sum(state.F[:j] - state.U[:j] - state.W[:j])-state.V[0]) * (state.Hliq[j-1] - state.Hliq[j]) - state.F[j] * (
                    state.Hliq[j] - state.Hfeed[j])
                     + state.W[j] * (state.Hvap[j] - state.Hliq[j])) - (state.Hvap[j] - state.Hliq[j - 1]) * vnew[j]
        denominator = (state.Hliq[j] - state.Hvap[j + 1])
        vnew = vnew.at[j+1].set(numerator / denominator)
        return vnew

    #vnew, add = jax.lax.scan(update_vnew, vnew, jnp.arange(1, len(state.V)))
    for j in range(1, len(state.V)):
        vnew = update_vnew(vnew, j)

    lnew = jnp.zeros(len(state.L))
    #lnew = lnew.at[0].set(state.RR * state.U[0])
    #lnew = lnew.at[-1].set(state.bottom)

    def update_lnew(j, lnew):
        lnew = lnew.at[j].set(vnew[j + 1] + jnp.sum(state.F[:j + 1] - state.U[:j + 1] - state.W[:j + 1]) - vnew[0])
        return lnew
    #lnew = vmap(update_lnew, in_axes=[0, None])(jnp.arange(1, len(state.L)-1), vnew)
    #lnew, add = jax.lax.scan(update_lnew, lnew, jnp.arange(1, len(state.L)-1))
    for j in range(1, len(state.L)-1):
        lnew = update_lnew(j, lnew)

    mask = jnp.arange(len(state.V))
    return state.replace(L=jnp.where((mask > 0) & (mask < state.Nstages-1), lnew, state.L), V=jnp.where((mask > 1) & (mask < state.Nstages), vnew, state.V))
