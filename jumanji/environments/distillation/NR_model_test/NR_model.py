import jax.numpy as jnp
import jax
from jax import vmap, lax, jit
from jumanji.environments.distillation.NR_model_test.distillation_types import State, NR_State, Trays, Tray, Thermo
from jumanji.environments.distillation.NR_model_test import functions, initial_composition, matrix_transforms, jacobian, thermodynamics, costing, purity_constraint
import os

# from NR_model_test.analytic_jacobian import jacobian as pure_jac
# from NR_model_test import analytic_jacobian
import os


def initialize():
    n_max = 100
    c_max = 3
    dir = os.path.join(os.getcwd(), 'Pure component parameters')
    # psat_params, cpvap_params, hvap_params, hform_params = thermodynamics.retrieve_params(dir)
    return State(
        L=jnp.zeros(n_max, dtype=float),
        V=jnp.zeros(n_max, dtype=float),
        U=jnp.zeros(n_max, dtype=float),
        W=jnp.zeros(n_max, dtype=float),
        X=jnp.zeros((c_max, n_max), dtype=float),
        Y=jnp.zeros((c_max, n_max), dtype=float),
        temperature=jnp.zeros(n_max, dtype=float),
        F=jnp.zeros(n_max, dtype=float),
        components=jnp.zeros(c_max, dtype=int),
        pressure=jnp.zeros((), dtype=float),
        z=jnp.zeros(c_max, dtype=float),
        RR=jnp.zeros((), dtype=float),
        distillate=jnp.zeros((), dtype=float),
        Nstages=jnp.zeros((), dtype=int),
        Hliq=jnp.zeros(n_max, dtype=float),
        Hvap=jnp.zeros(n_max, dtype=float),
        Hfeed=jnp.zeros(n_max, dtype=float),
        CD=jnp.zeros((), dtype=float),
        RD=jnp.zeros((), dtype=float),
        TAC=jnp.zeros((), dtype=float),
        trays=Trays(
            low_tray=Tray(
                l=jnp.zeros((c_max, n_max)),
                v=jnp.zeros((c_max, n_max)),
                T=jnp.zeros(n_max)
            ),
            high_tray=Tray(
                l=jnp.zeros((c_max, n_max)),
                v=jnp.zeros((c_max, n_max)),
                T=jnp.zeros(n_max)
            ),
            tray=Tray(
                l=jnp.zeros((c_max, n_max)),
                v=jnp.zeros((c_max, n_max)),
                T=jnp.zeros(n_max)
            )
        ),
        heavy_key=jnp.zeros((), dtype=int),
        light_key=jnp.zeros((), dtype=int),
        heavy_spec=jnp.zeros((), dtype=float),
        light_spec=jnp.zeros((), dtype=float),
        step_count=jnp.zeros((), dtype=int),  # ()
        action_mask=jnp.ones(7500, dtype=bool),  # (4,)
        key=jax.random.PRNGKey(0),  # (2,)
        residuals=jnp.zeros(100, dtype=float),
        analytics=jnp.zeros((), dtype=bool),
        converged=jnp.ones((), dtype=bool)
    )


def initial_guess(state: State, nstages, feedstage, pressure, feed, z, distillate, rr, analytics: bool, specs: bool,
                  heavy_recovery, light_recovery):
    #fug_state = purity_constraint.FUG_specs(z, heavy_recovery, light_recovery, pressure, feed)
    #fug_state = fug_state._replace(stages=nstages)
    #fug_state = purity_constraint.feedstage(fug_state)
    feedstage = jnp.ceil((nstages+1)/2) #jnp.where(specs == True, fug_state.feed_stage, feedstage)
    #rr = jnp.where(specs == True, fug_state.reflux, rr)
    #distillate = jnp.where(specs == True, fug_state.distillate, distillate)
    l_init = jnp.where(jnp.arange(len(state.L)) < feedstage - 1, rr * distillate, rr * distillate + jnp.sum(feed))
    l_init = l_init.at[nstages - 1].set((jnp.sum(feed) - distillate))
    v_init = jnp.where(jnp.arange(len(l_init)) > 0, (rr + jnp.ones_like(l_init)) * distillate, distillate)
    f = jnp.where(jnp.arange(len(l_init)) == feedstage - 1, feed, 0)
    mask = jnp.where(jnp.arange(len(state.temperature)) < nstages, 1, 0)
    return state.replace(
        L=l_init * mask,
        V=v_init * mask,
        z=z,
        RR=rr,
        distillate=distillate,
        pressure=pressure,
        F=f,
        Nstages=nstages,
        components=jnp.array([2, 3, 4], dtype=int),  #jnp.arange(len(z)),  #
        heavy_key=jnp.zeros((), dtype=int),  #jnp.where(specs == True, fug_state.heavy_key, jnp.zeros((), dtype=int)),
        light_key=jnp.zeros((), dtype=int),  #jnp.where(specs == True, fug_state.light_key, jnp.zeros((), dtype=int)),
        heavy_spec=heavy_recovery,
        light_spec=light_recovery
    )


def initial_temperature(state: State):
    t_range = functions.t_sat_solver(state.components, state.pressure)
    t_range = jnp.where(state.components > 0, t_range, jnp.max(t_range) - 1)

    delta_t = (jnp.max(t_range) - jnp.min(t_range)) / state.Nstages
    return state.replace(
        temperature=jnp.where(jnp.arange(len(state.temperature)) < state.Nstages,
                              jnp.min(t_range) + jnp.arange(len(state.temperature)) * delta_t, 0),
    )


def f_sol(state: State, tray_low, tray, tray_high, j):
    f_s = jacobian.f_vector_function(
        state,
        Tray(l=tray_low.l[:, j], v=tray_low.v[:, j], T=tray_low.T[j]),
        Tray(l=tray.l[:, j], v=tray.v[:, j], T=tray.T[j]),
        Tray(l=tray_high.l[:, j], v=tray_high.v[:, j], T=tray_high.T[j]),
        j,
    )
    return jnp.concatenate((jnp.asarray(f_s.H)[None], jnp.asarray(f_s.M), jnp.asarray(f_s.E)), axis=0)


def update_NR(state: State):
    a, b, c = jacobian.jacobian_func(state)
    # a, b, c = pure_jac(state)
    f = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                         state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))
    dx = jnp.nan_to_num(functions.thomas(a, b, c, -1 * f, state.Nstages))  # .reshape(-1,1)
    
    def min_res(t, state: State, dx):
        dx_v = dx[:, :len(state.components)].transpose()
        dx_l = dx[:, -len(state.components):].transpose()
        dx_t = dx[:, len(state.components)].transpose()

        v_new = (state.trays.tray.v + t * dx_v)
        l_new = (state.trays.tray.l + t * dx_l)
        temp = (state.trays.tray.T + t * dx_t)
        v_max = jnp.max(state.trays.tray.v)
        l_max = jnp.max(state.trays.tray.l)
        # temp = jnp.where(dx[:, len(state.components)].transpose() < 10., jnp.where(dx[:, len(state.components)].transpose() > -10., state.trays.tray.T + t * dx[:, len(state.components)].transpose(), state.trays.tray.T - 10.), state.trays.tray.T + 10.)
        v_new_final = jnp.where(v_new > 0., v_new, state.trays.tray.v
                                * jnp.exp(t * dx_v / jnp.where(state.trays.tray.v > 0, state.trays.tray.v, 1e30)))
        l_new_final = jnp.where(l_new > 0., l_new, state.trays.tray.l
                                * jnp.exp(t * dx_l / jnp.where(state.trays.tray.l > 0, state.trays.tray.l, 1e30)))
        temp_final = jnp.where(jnp.abs(dx_t * t) > 7., (state.trays.tray.T + 7. * (dx_t) / (jnp.abs(dx_t))),
                               (state.trays.tray.T + t * dx_t))

        v_new_final = jnp.where((jnp.abs(dx_v * t) > 0.45 * state.trays.tray.v),
                                state.trays.tray.v + 0.45 * state.trays.tray.v * dx_v / jnp.abs(dx_v), v_new_final)
        l_new_final = jnp.where((jnp.abs(dx_l * t) > 0.45 * state.trays.tray.l),
                                state.trays.tray.l + 0.45 * state.trays.tray.l * dx_l / jnp.abs(dx_l), l_new_final)

        state = state.replace(
            V=jnp.sum(v_new_final, axis=0),
            L=jnp.sum(l_new_final, axis=0),
            Y=jnp.nan_to_num(v_new_final / jnp.sum(v_new_final, axis=0)),
            X=jnp.nan_to_num(l_new_final / jnp.sum(l_new_final, axis=0)),
            temperature=temp_final,
        )

        state = matrix_transforms.trays_func(state)
        f = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                             state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))

        return jnp.sum(f ** 2), state

    carry = vmap(min_res, in_axes=(0, None, None))(jnp.arange(0.2, 1.3, 0.001), state, dx)
    result, states = carry
    new_t = jnp.max(jnp.where(result == jnp.min(result), jnp.arange(0.2, 1.3, 0.001), 0))

    res, state_new = min_res(new_t, state, dx)
    zeros = None
    dx_final = None

    return res, state_new, zeros, dx_final


def cond_fn(args):
    state, iterations, res = args
    comps = jnp.sum(jnp.where(state.z > 0, 1, 0))
    cond = state.Nstages * (2 * comps + 1) * jnp.sum(state.F) * 1e-9
    return (iterations < 100) & (res > cond)


def body_fn(args):
    state, iterations, res = args
    res_new, nr_state_new, new_t, dx = update_NR(state)
    nr_state_new = nr_state_new.replace(residuals=nr_state_new.residuals.at[iterations].set(res_new))
    iterations += 1
    return nr_state_new, iterations, res_new


def store_variable(small_array, larger_array, start_indices):
    # Assuming small_array has shape (2, 20) and larger_array has shape (9000, 20)
    updated_larger_array = lax.dynamic_update_slice(larger_array, small_array, start_indices)
    return updated_larger_array


def converge_column(state: State):
    # nr_state = initialize_NR(state)

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
    '''
    res_array = jnp.zeros(1000, dtype=float)
    f_damp = jnp.array(1., dtype=float)
    damping = jnp.ones(1000, dtype=float) * 0.01
    profiler = jnp.zeros((1000, 3, len(state.temperature)), dtype=float)

    for rang in range(5):
        res, state, new_t, f = update_NR(state)
    '''
    return state, iterations, res


def condensor_duty(state: State):
    result = jnp.sum(
        state.V[1] * state.Y[:, 1] * thermodynamics.vapor_enthalpy(state.temperature[1], state.components)) + jnp.sum(
        state.F[0] * state.z * state.Hfeed[0]) - jnp.sum(
        (state.L[0]) * state.X[:, 0] * thermodynamics.liquid_enthalpy(state.temperature[0],
                                                                      state.components)) - jnp.sum(
        state.V[0] * state.Y[:, 0] * thermodynamics.vapor_enthalpy(state.temperature[0], state.components))

    return state.replace(CD=result)


def reboiler_duty(state: State):
    result = jnp.sum(state.F * state.Hfeed) - state.CD - state.L[state.Nstages - 1] * jnp.sum(
        state.X[:, state.Nstages - 1] * thermodynamics.liquid_enthalpy(state.temperature[state.Nstages - 1],
                                                                       state.components)) - state.V[0] * jnp.sum(
        state.Y[:, 0] * thermodynamics.vapor_enthalpy(state.temperature[0], state.components))
    CD = jnp.sum(
        vmap(thermodynamics.h_evap, in_axes=(None, 0))(state.temperature[0], state.components) * state.Y[:, 0]) * \
         state.V[0]
    return state.replace(RD=result,
                         CD=state.CD + CD)


def test_simulation(state, nstages, feedstage, pressure, feed, z, distillate, rr, analytics=False, specs=False,
                      heavy_recovery=jnp.array(0.99, dtype=float), light_recovery=jnp.array(0.99, dtype=float)):
    iterations = 0
    res = 0
    # feedstage = jnp.floor((nstages + 1) / 2 )
    # state = initialize()

    state = initial_guess(state=state, nstages=nstages, feedstage=feedstage, pressure=pressure, feed=feed, z=z,
                          distillate=distillate, rr=rr, analytics=analytics, specs=specs, heavy_recovery=heavy_recovery,
                          light_recovery=light_recovery)

    state = initial_temperature(state)
    state = state.replace(Hfeed=jnp.where(state.F > 0, jnp.sum(thermodynamics.feed_enthalpy(state) * state.z), 0))


    return state, iterations, res

def inside_simulation(state, nstages, feedstage, pressure, feed, z, distillate, rr, analytics=False, specs=False,
                      heavy_recovery=jnp.array(0.99, dtype=float), light_recovery=jnp.array(0.99, dtype=float)):
    iterations = 0
    res = 0
    # feedstage = jnp.floor((nstages + 1) / 2 )
    # state = initialize()

    state = initial_guess(state=state, nstages=nstages, feedstage=feedstage, pressure=pressure, feed=feed, z=z,
                          distillate=distillate, rr=rr, analytics=analytics, specs=specs, heavy_recovery=heavy_recovery,
                          light_recovery=light_recovery)

    state = initial_temperature(state)

    state = state.replace(Hfeed=jnp.where(state.F > 0, jnp.sum(thermodynamics.feed_enthalpy(state) * state.z), 0))
    #state = state.replace(X=(jnp.ones(len(state.temperature))[:, None]*state.z).transpose())
    #state = functions.y_func(state)

    '''
    def for_body(state, i):
        return initial_composition.bubble_point(state), None

    state, _ = jax.lax.scan(for_body, state, jnp.arange(3))
    '''
    state = initial_composition.bubble_point(state)
    
    state, iterations, res = converge_column(state)
    state = state.replace(
        Hliq=jnp.sum(vmap(thermodynamics.liquid_enthalpy, in_axes=(0, None))(state.temperature,
                                                                             state.components).transpose() * state.X,
                     axis=0),
        Hvap=jnp.sum(vmap(thermodynamics.vapor_enthalpy, in_axes=(0, None))(state.temperature,
                                                                            state.components).transpose() * state.Y,
                     axis=0)
    )

    state = condensor_duty(state)
    state = reboiler_duty(state)
    state = costing.tac(state)
    real_tac = jnp.where(iterations<100, state.TAC, 75)
    state = state.replace(TAC=real_tac)
    state = convergence_check(state)
    #state = flowcheck(state)

    return state, iterations, res


def convergence_check(state: State):
    return state.replace(
        converged=jnp.abs(state.V[0] - state.distillate) < 1e-2
    )


def flowcheck(state: State):
    top_flow = state.Y[:, 0]*state.V[0]
    bot_flow = state.X[:, state.Nstages-1] * (jnp.sum(state.F) - state.V[0])
    top_flow = jnp.where(top_flow <= jnp.array((200, 300, 500)), top_flow, 1)
    bot_flow = jnp.where(bot_flow <= jnp.array((200, 300, 500)), bot_flow, 1)
    return state.replace(
        X=state.Y.at[:, 0].set(top_flow/jnp.sum(top_flow)),
        Y=state.X.at[:, state.Nstages-1].set(bot_flow/jnp.sum(bot_flow))
    )
