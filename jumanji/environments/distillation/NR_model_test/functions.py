import jax.numpy as jnp
import jax.lax
from jax import vmap
from jaxopt import Bisection
from jumanji.environments.distillation.NR_model_test.distillation_types import State, Tray, Trays
from jumanji.environments.distillation.NR_model_test import thermodynamics



def t_sat_solver(components, pressure):
    return vmap(thermodynamics.t_sat, in_axes=(0, None, None))(components, jnp.array(500., dtype=float), pressure)


def tray_t_solver(state, tray, temperature, xc):
    def function(tempcalc):
        return 1 - jnp.sum(vmap(thermodynamics.k_eq, in_axes=(None, 0, None))(tempcalc, state.components, state.pressure) * xc[:, tray])
    result = Bisection(optimality_fun=function, lower=500., upper=100., check_bracket=False).run(temperature[tray]).params
    return result


def stage_temperature(state: State):
    temperature = vmap(tray_t_solver, in_axes=(None, 0, None, None))(state, jnp.arange(0, len(state.L)),
                                                               state.temperature, state.X)
    return state.replace(temperature=temperature)


def y_func(state: State):
    def calculate_y(i):
        return vmap(thermodynamics.k_eq, in_axes=(None, 0, None))(state.temperature[i], state.components, state.pressure) * state.X[:, i]
    y = vmap(calculate_y)(jnp.arange(0, len(state.Y[0])))
    return state.replace(Y=jnp.nan_to_num(y.transpose() / jnp.sum(y, axis=1)))


def tray_tuple_to_matrix(tuple_input: Tray, tray_index):
    return jnp.concatenate((tuple_input.V[:, tray_index], jnp.array([tuple_input.T[tray_index]]), tuple_input.L[:, tray_index]))


def part_matrix(a, b, c):
    mat = jnp.zeros([len(b), len(b), len(b[0,:]), len(b[0,:])])
    for j in range(len(b)):
        mat = mat.at[j, j, :, :].set(b[j])

    for j in range(len(a)):
        mat = mat.at[j+1, j, :, :].set(a[j])

    for j in range(len(c)):
        mat = mat.at[j, j+1, :, :].set(c[j])
    mat = jnp.concatenate(jnp.concatenate(mat, axis=(2)))
    return mat


def thomas(a, b, c, f, n_stages):
    # Forward sweep
    n = len(f)
    u = jnp.zeros_like(f)

    u = u.at[0].set(jnp.linalg.solve(b[0], f[0]))
    c = c.at[0].set(jnp.linalg.solve(b[0], c[0]))

    def for_body_forward(carry, i):
        a, b, c, u, f = carry
        c = c.at[i].set(jnp.linalg.solve(b[i] - jnp.matmul(a[i - 1], c[i - 1]), c[i]))
        u = u.at[i].set(
            jnp.linalg.solve(b[i] - jnp.matmul(a[i - 1], c[i - 1]), f[i] - jnp.matmul(a[i - 1], u[i - 1])))
        return (a, b, c, u, f), i
    carry, add = jax.lax.scan(for_body_forward, (a, b, c, u, f), jnp.arange(1, len(f)))
    a, b, c, u, f = carry
    # Backward sweep
    x = jnp.zeros_like(f)
    x = x.at[n_stages-1].set(u[n_stages-1])

    def for_body(carry, i):
        x, u, c, n_stages = carry
        x = jnp.where(i == n_stages - 1, x, x.at[i].set(u[i] - jnp.matmul(c[i], x[i + 1])))
        return (x, u, c, n_stages), None

    x_carry, _ = jax.lax.scan(for_body, (x, u, c, n_stages), jnp.arange(n-2, -1, -1))
    x, u, c, n_stages = x_carry
    #for i in range(n-2, -1, -1):
    #    x = jnp.where(i == n_stages - 1, x, x.at[i].set(u[i] - jnp.matmul(c[i], x[i+1])))


    return x
