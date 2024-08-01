import jax.numpy as jnp
import jax
from jax import vmap, lax
from Distillation.NR_model.distillation_types import State, NR_State, Trays, Tray
from Distillation.NR_model.jacobian_total_condenser import jacobian_func
from Distillation.NR_model import functions, initial_composition, matrix_transforms, thermodynamics
from Distillation.NR_model import jacobian_total_condenser as jacobian



def initialize():
    n_max = 6
    c_max = 3
    return State(
        L=jnp.zeros(n_max, dtype=float),
        V=jnp.zeros(n_max, dtype=float),
        U=jnp.zeros(n_max, dtype=float),
        W=jnp.zeros(n_max, dtype=float),
        X=jnp.zeros((c_max, n_max), dtype=float),
        Y=jnp.zeros((c_max, n_max), dtype=float),
        s=jnp.zeros(n_max, dtype=float),
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
        )
    )


def initial_guess(state: State, nstages, feedstage, pressure, feed, z, distillate, rr):
    l_init = jnp.where(jnp.arange(len(state.L)) < feedstage-1, rr*distillate, rr*distillate+jnp.sum(feed))
    l_init = l_init.at[nstages-1].set((jnp.sum(feed)-distillate))
    v_init = jnp.where(jnp.arange(len(l_init)) > 0, (rr + jnp.ones_like(l_init))*distillate, 0)
    f = jnp.where(jnp.arange(len(l_init)) == feedstage-1, feed, 0)
    mask = jnp.where(jnp.arange(len(state.temperature)) < nstages, 1, 0)
    return state._replace(
        L=l_init*mask,
        V=v_init*mask,
        U=jnp.zeros_like(state.temperature).at[0].set(distillate),
        s=jnp.zeros_like(state.temperature).at[0].set(1/rr),
        z=z,
        RR=rr,
        distillate=distillate,
        pressure=pressure,
        F=f,
        Nstages=nstages,
        components=jnp.array([3,4,7], dtype=int),  #jnp.arange(len(z))
    )


def initial_temperature(state: State):
    t_range = functions.t_sat_solver(state.components, state.pressure)
    t_range = jnp.where(state.components > 0, t_range, jnp.max(t_range)-1)

    delta_t = (jnp.max(t_range)-jnp.min(t_range))/state.Nstages
    return state._replace(
        temperature=jnp.where(jnp.arange(len(state.temperature)) < state.Nstages, jnp.min(t_range)+jnp.arange(len(state.temperature))*delta_t, 0),
        )

def initialize_NR(state: State):
    return NR_State(
        l=(state.L*state.X),
        v=(state.V*state.Y),
        temperature=state.temperature,
        f=(state.F*state.z[:, None]),
        s=jnp.where(state.U > 0, 1/state.RR, 0 ),
        components=state.components,
        z=state.z,
        pressure=state.pressure,
        RR=state.RR,
        distillate=state.distillate,
        Nstages=state.Nstages,
        trays=Trays(
            low_tray=Tray(
                l=jnp.zeros_like(state.X),
                v=jnp.zeros_like(state.Y),
                T=jnp.zeros_like(state.temperature)
            ),
            high_tray=Tray(
                l=jnp.zeros_like(state.X),
                v=jnp.zeros_like(state.Y),
                T=jnp.zeros_like(state.temperature)
            ),
            tray=Tray(
                l=jnp.zeros_like(state.X),
                v=jnp.zeros_like(state.Y),
                T=jnp.zeros_like(state.temperature)
            ),
        ),
        h_feed=thermodynamics.feed_enthalpy(state),
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
    a, b, c = jacobian_func(state)
    f = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray,
                                                           state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))

    dx = jnp.nan_to_num(functions.thomas(a, b, c, -1 * f, state.Nstages)).reshape(-1,1)

    def min_res(t, state: State, dx):

        hopper = len(state.components) * 2 + 1
        dx_v = vmap(lambda j: dx[jnp.arange(len(state.components)) + j * hopper, 1])(
            jnp.arange(len(state.temperature))).transpose()
        dx_l = vmap(lambda j: dx[jnp.arange(len(state.components)) + (len(state.components) + 1) + j * hopper, 1])(
            jnp.arange(len(state.temperature))).transpose()
        dx_t = vmap(lambda j: dx[len(state.components) + j * hopper, 1])(
            jnp.arange(len(state.temperature))).transpose()
        '''
        dx_v = dx[:, :len(state.components)].transpose()
        dx_l = dx[:, -len(state.components):].transpose()
        dx_t = dx[:, len(state.components)].transpose()
        '''
        v_new = (state.trays.tray.v + t * dx_v)
        l_new = (state.trays.tray.l + t * dx_l)
        temp = (state.trays.tray.T + t * dx_t)
        # temp = jnp.where(dx[:, len(state.components)].transpose() < 10., jnp.where(dx[:, len(state.components)].transpose() > -10., state.trays.tray.T + t * dx[:, len(state.components)].transpose(), state.trays.tray.T - 10.), state.trays.tray.T + 10.)
        v_new_final = jnp.where(v_new > 0, v_new, state.trays.tray.v
                                * jnp.exp(t * dx_v / jnp.where(state.trays.tray.v > 0, state.trays.tray.v, 1e30)))
        l_new_final = jnp.where(l_new > 0, l_new, state.trays.tray.l
                                * jnp.exp(t * dx_l / jnp.where(state.trays.tray.l > 0, state.trays.tray.l, 1e30)))
        temp_final = jnp.where(jnp.abs(dx_t) > 7, (state.trays.tray.T + 7 * (dx_t) / (jnp.abs(dx_t))),
                               (state.trays.tray.T + t * dx_t))
        state = state._replace(
            V=jnp.sum(v_new_final, axis=0),
            L=jnp.sum(l_new_final, axis=0),
            Y=v_new_final/jnp.sum(v_new_final, axis=0),
            X=l_new_final/jnp.sum(l_new_final, axis=0),
            temperature=temp,
        )

        state = matrix_transforms.trays_func(state)
        f = vmap(f_sol, in_axes=(None, None, None, None, 0))(state, state.trays.low_tray, state.trays.tray, state.trays.high_tray, jnp.arange(len(state.trays.tray.T)))

        return jnp.sum(f**2), state

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
    cond = state.Nstages*(2*comps+1)*jnp.sum(state.F)*1e-9
    return (iterations < 50) & (res > cond)


def body_fn(args):
    state, iterations, res = args
    res_new, nr_state_new, new_t, dx = update_NR(state)
    iterations += 1
    return nr_state_new, iterations, res_new

def store_variable(small_array, larger_array, start_indices):
    # Assuming small_array has shape (2, 20) and larger_array has shape (9000, 20)
    updated_larger_array = lax.dynamic_update_slice(larger_array, small_array, start_indices)
    return updated_larger_array


def converge_column(state: State):
    #nr_state = initialize_NR(state)
    state = matrix_transforms.trays_func(state)
    iterations = 0
    res = 0
    '''
    state, iterations, res = (
        lax.while_loop(cond_fun=cond_fn, body_fun=body_fn,
                       init_val=(state,
                                 iterations,
                                 jnp.array(10, dtype=float),
                                 )
                       )
    )
    
    res_array = jnp.zeros(1000, dtype=float)
    f_damp = jnp.array(1., dtype=float)
    damping = jnp.ones(1000, dtype=float) * 0.01
    profiler = jnp.zeros((1000, 3, len(state.temperature)), dtype=float)
    '''
    for rang in range(4):
        res, state, new_t, f = update_NR(state)


    return state, iterations, res


def inside_simulation(nstages, feedstage, pressure, feed, z, distillate, rr):
    iterations = 0
    res = 0
    #feedstage = jnp.floor((nstages + 1) / 2 )
    state = initialize()
    state = initial_guess(state=state, nstages=nstages, feedstage=feedstage, pressure=pressure, feed=feed, z=z,
                               distillate=distillate, rr=rr)

    state = initial_temperature(state)
    state = state._replace(Hfeed=jnp.where(state.F > 0, jnp.sum(thermodynamics.feed_enthalpy(state)*state.z), 0))

    def for_body(state, i):
        return initial_composition.bubble_point(state), None

    state, _ = jax.lax.scan(for_body, state, jnp.arange(4))
    state, iterations, res = converge_column(state)
    state = state._replace(
        Hliq=jnp.sum(vmap(thermodynamics.liquid_enthalpy, in_axes=(0, None))(state.temperature,
                                                                               state.components).transpose() * state.X,
                     axis=0),
        Hvap=jnp.sum(vmap(thermodynamics.vapor_enthalpy, in_axes=(0, None))(state.temperature,
                                                                              state.components).transpose() * state.Y,
                     axis=0)
    )

    #state = condensor_duty(state)
    #state = reboiler_duty(state)
    #state = costing.tac(state)
    #feed_stage = jnp.max(jnp.where(state.F > 0, jnp.arange(len(state.temperature)), 0))
    #state, iteration = converge_bubble_point(state)


    return state, iterations, res
'''

state = initialize()
zf = jnp.array([0., 0., 0., 0.2, 0.5, 0., 0., 0.3], dtype=float)

state, iterations, res = inside_simulation(jnp.array(20, dtype=int), jnp.array(10, dtype=int), jnp.array(2.0, dtype=float),
                                      jnp.array(1000.0, dtype=float), jnp.array(zf, dtype=float),
                                      jnp.array(400, dtype=float), jnp.array(1, dtype=float))

for i in range(2):
    st = time()
    state_new, iterations, res = (vmap(jit(inside_simulation), in_axes=(0, 0, None, None, None, None, None)))(
        jnp.arange(10, 20),
        jnp.concatenate(jnp.concatenate((jnp.arange(5, 10)[:, None],jnp.arange(5, 10)[:, None]), axis=1)),
        jnp.array(2.0, dtype=float),
        jnp.array(1000.0, dtype=float),
        jnp.array(zf, dtype=float),
        jnp.array(400 + 100*i, dtype=float),
        jnp.array(1+i, dtype=float)
    )
    print(time()-st)


plot_function(jnp.arange(1, state.Nstages+1), state.L[0:state.Nstages], state.V[0:state.Nstages], state.temperature[0:state.Nstages], state.Hliq[0:state.Nstages], state.Hvap[0:state.Nstages], state.X[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], state.Y[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], jnp.where(zf != 0, jnp.arange(0, len(zf)), 0))

print(iterations)


cond = state.Nstages*(2*len(state.components)+1)*jnp.sum(state.F)*1e-9
fig4 = plt.figure(5)
ax4 = fig4.add_subplot(111)
# calculate mole fractions
ax4.plot(jnp.arange(len(state.res[0:iterations])), state.res[0:iterations])
ax4.set_ylabel('residuals')
ax4.set_xlabel('iteration')
ax4.set_ylim([0, 5*jnp.min(res)])
#ax4.set_xlim([0, 50])
#ax4.legend()
ax4.grid()
print(state.res)

fig5 = plt.figure(6)
ax5 = fig5.add_subplot(111)
# calculate mole fractions
ax5.plot(jnp.arange(len(state.damping[0:iterations])), state.damping[0:iterations])
ax5.set_ylabel('damping_factor')
ax5.set_xlabel('iteration')
ax5.grid()



print(iterations)
non_zero_mask = jnp.any(state.storage != 0, axis=1)

data = -(state.profiler[0:iterations+1, :, :]-jnp.min(state.profiler))/jnp.min(state.profiler)*100-100
fig, ax = plt.subplots()

# Set limits for the axes
ax.set_xlim(1, len(data[0].transpose()))
#ax.set_ylim(min(min(row) for row in data[1:, :]), max(max(row) for row in data[1:, :]))
ax.set_ylim(0, 1)
# Create an empty line object for the animation
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,


# Function to update the plot for each frame of the animation
def update(frame):
    ax.cla()
    x = jnp.arange(1, len(data[frame].transpose())+1)
    y = data[frame].transpose()
    ax.cla()
    ax.plot(x, y)
    ax.set_title([f'Iteration {frame}'])
    ax.legend(['H', 'M1', 'M2', 'E1', 'E2'], loc=9)
    ax.set_xlabel('Stage')
    ax.set_ylabel('C7 mole flow kmole/hr')
    ax.set_ylim([-0.5, 0.5])
    return ax,

ani = FuncAnimation(fig, update, frames=len(data) , repeat=True, interval=0.001)
#ani.save('oscillation', writer='ffmpeg')


plt.show()
'''