import jax.numpy as jnp
from jax import jacfwd, vmap
from jumanji.environments.distillation.NR_model_test.distillation_types import Tray, Mesh, State
from jumanji.environments.distillation.NR_model_test import thermodynamics as thermo
from jumanji.environments.distillation.NR_model_test.matrix_transforms import single_tuple_to_matrix


def m_function(state, tray_low, tray, tray_high, i, j):
    return tray.l[i] + tray.v[i] - tray_high.l[i] - tray_low.v[i] - state.F[j]*state.z[i]


def m_simple(state, tray, i, j):
    eqmol_top = jnp.sum(tray.v)-state.distillate
    eqmol_bottom = jnp.sum(tray.l)-(jnp.sum(state.F)-state.distillate)
    eqmol = jnp.where(j < jnp.max(jnp.where(state.F > 0, jnp.arange(len(state.F)), 0)),
                      jnp.sum(tray.v) - state.V[j],
                      jnp.sum(tray.l) - state.L[j])
    return eqmol


def e_function(state: State, tray_low, tray, tray_high, i, j):
    return ((tray.l[i]/jnp.sum(tray.l) * thermo.k_eq(tray.T, state.components[i], state.pressure) - tray.v[i]/jnp.sum(tray.v))) #*jnp.where(state.z > 0, 1, 0)[i]



def h_function(state: State, tray_low, tray, tray_high, j):
    h_evap = thermo.h_evap(tray.T, 4)
    h_vap = jnp.average(thermo.liquid_enthalpy(tray.T, state.components))
    #-(jnp.sum(tray.v) - jnp.sum(tray.l) / state.RR)
    cond_spec = jnp.where(state.light_key>0,
                          (tray.v[state.light_key]-state.light_spec*jnp.sum(state.F)*state.z[state.light_key]),
                          -(jnp.sum(tray.v) - jnp.sum(tray.l) / state.RR)
                          )
    reb_spec = jnp.where(state.heavy_key>0,
                         (tray.l[state.heavy_key]-state.heavy_spec * jnp.sum(state.F)*state.z[state.heavy_key]),
                         -(jnp.sum(tray.l) - (jnp.sum(state.F) - state.distillate))
                         )

    result = jnp.where(j == 0, cond_spec, (
        jnp.where(j == state.Nstages - 1, reb_spec, (
                jnp.sum(tray.l * thermo.liquid_enthalpy(tray.T, state.components)) + jnp.sum(
            tray.v * thermo.vapor_enthalpy(tray.T, state.components))
                - jnp.sum(tray_high.l * thermo.liquid_enthalpy(tray_high.T, state.components)) - jnp.sum(
            tray_low.v * thermo.vapor_enthalpy(tray_low.T, state.components)) - jnp.sum(state.F*state.Hfeed[j]))/(-h_vap))))
    return result


def f_vector_function(state: State, tray_low, tray, tray_high, j):
    h = jnp.asarray(jnp.where(j < state.Nstages, h_function(state, tray_low, tray, tray_high, j), 0))
    m = jnp.asarray(jnp.where(j < state.Nstages, vmap(m_function, in_axes=(None, None, None, None, 0, None))(state, tray_low, tray, tray_high, jnp.arange(len(state.components)), j), 0))
    e = jnp.asarray(jnp.where(j < state.Nstages, vmap(e_function, in_axes=(None, None, None, None, 0, None))(state, tray_low, tray, tray_high, jnp.arange(len(state.components)), j), 0))
    return Mesh(H=h,
                M=m,
                E=e,
                )

def g_vector_function(state: State, tray_low, tray, tray_high, j):
    eqmol_top = jnp.sum(tray.v)-state.distillate
    eqmol_bottom = jnp.sum(tray.l)-(jnp.sum(state.F)-state.distillate)
    '''
    eqmol = jnp.where(j < jnp.max(jnp.where(state.F > 0, jnp.arange(len(state.F)), 0)),
                      jnp.sum(tray.v)/(state.RR+1) - jnp.sum(tray.l)/state.RR,
                      jnp.sum(tray.v)/(state.RR+1) - (jnp.sum(tray.l) - jnp.sum(state.F))/state.RR)
    '''
    eqmol = jnp.sum(tray.l)-state.L[j]
    m = jnp.asarray(jnp.where(j < state.Nstages, vmap(m_function, in_axes=(None, None, None, None, 0, None))(state, tray_low, tray, tray_high, jnp.arange(len(state.components)), j), 0))
    h = eqmol
    e = jnp.asarray(jnp.where(j < state.Nstages, vmap(e_function, in_axes=(None, None, None, None, 0, None))(state, tray_low, tray, tray_high, jnp.arange(len(state.components)), j), 0))

    return Mesh(H=h,
                M=m,
                E=e,
                )

def f_jac_a(state: State, tray_low, tray, tray_high, j):
    a = jacfwd(f_vector_function, argnums=3)(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return single_tuple_to_matrix(a)


def f_jac_b(state: State, tray_low, tray, tray_high, j):
    b = jacfwd(f_vector_function, argnums=2)(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return single_tuple_to_matrix(b)


def f_jac_c(state: State, tray_low, tray, tray_high, j):
    c = jacfwd(f_vector_function, argnums=1)(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return single_tuple_to_matrix(c)


def g_jac_a(state: State, tray_low, tray, tray_high, j):
    a = jacfwd(g_vector_function, argnums=3)(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return single_tuple_to_matrix(a)


def g_jac_b(state: State, tray_low, tray, tray_high, j):
    b = jacfwd(g_vector_function, argnums=2)(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return single_tuple_to_matrix(b)


def g_jac_c(state: State, tray_low, tray, tray_high, j):
    c = jacfwd(g_vector_function, argnums=1)(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return single_tuple_to_matrix(c)


def jacobian_func(state: State):
    b = vmap(f_jac_b, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                           state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    a = vmap(f_jac_a, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                           state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    c = vmap(f_jac_c, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                           state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    return a[1:, :, :], b, c[:-1, :, :]


def g_jacobian_func(state: State):
    b = vmap(g_jac_b, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                           state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    a = vmap(g_jac_a, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                           state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    c = vmap(g_jac_c, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                           state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    return a[1:, :, :], b, c[:-1, :, :]
