import jax.numpy as jnp
from jax import vmap
from jumanji.environments.distillation.NR_model_test.distillation_types import Tray, Mesh, State


def trays_func(state: State):
    tray_l = vmap(tray_func, in_axes=(None, 0))(state, jnp.arange(1, len(state.temperature) + 1))
    tray_h = vmap(tray_func, in_axes=(None, 0))(state, jnp.arange(-1, len(state.temperature) - 1))
    tray_m = vmap(tray_func, in_axes=(None, 0))(state, jnp.arange(len(state.temperature)))
    return (Tray(l=tray_l.l.transpose(),
                v=tray_l.v.transpose(),
                T=tray_l.T.transpose()),
            Tray(
                l=tray_h.l.transpose(),
                v=tray_h.v.transpose(),
                T=tray_h.T.transpose()),
            Tray(
                l=tray_m.l.transpose(),
                v=tray_m.v.transpose(),
                T=tray_m.T.transpose())
            )


def tray_func(state: State, j):
    return Tray(
        l=jnp.where((j < 0) | (j > state.Nstages-1), jnp.zeros_like(state.X[:, j]), state.L[j]*state.X[:, j]),
        v=jnp.where((j < 0) | (j > state.Nstages-1), jnp.zeros_like(state.Y[:, j]), state.V[j]*state.Y[:, j]),
        T=jnp.where((j < 0) | (j > state.Nstages-1), jnp.zeros_like(state.temperature[j]), state.temperature[j])
    )


def single_tuple_to_matrix(tuple_input: Mesh):
    matrix_h = jnp.concatenate((tuple_input.H.v, jnp.array([tuple_input.H.T]), tuple_input.H.l))
    matrix_m = jnp.concatenate((tuple_input.M.v.transpose(), jnp.array([tuple_input.M.T]), tuple_input.M.l.transpose()))
    matrix_e = jnp.concatenate((tuple_input.E.v.transpose(), jnp.array([tuple_input.E.T]), tuple_input.E.l.transpose()))
    return jnp.concatenate((matrix_h[None, :], matrix_m.transpose(), matrix_e.transpose()))

