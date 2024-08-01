import jax.numpy as jnp
from time import time
from jax import jit, vmap
import matplotlib.pyplot as plt
from Distillation.plot_generation import plot_function
from Distillation.NR_model.NR_model import inside_simulation as simulation
from Distillation.NR_model.NR_model import initialize
#from Distillation.Newton_Raphson_directory.NR_algorithm_partial_final import inside_simulation as simulation
from Distillation.NR_model.NR_model import initialize
import os

#from Distillation.distillation_model import simulation

state_init= initialize()
zf = jnp.array([0.2, 0.5, 0.3], dtype=float)
tac_array = jnp.zeros(10, dtype=float)
'''
for i in range(2):
    st = time()
    carry = vmap(jit(simulation), in_axes=(0, 0, None, None, None, None, None))(
        jnp.arange(10, 20),
        jnp.concatenate(jnp.concatenate((jnp.arange(5, 10)[:, None],jnp.arange(5, 10)[:, None]), axis=1)),
        jnp.array(2.0, dtype=float),
        jnp.array(1000.0, dtype=float),
        jnp.array(zf, dtype=float),
        jnp.array(400 + 100*i, dtype=float),
        jnp.array(1+i, dtype=float)
    )

    print(time()-st)

state, iterations, res = carry
'''
zf = jnp.array([0.45, 0.55, 0.])
tac_array = jnp.zeros(30, dtype=float)
for i in range(2):
    state, iterations, res = vmap(jit(simulation), in_axes=(0, 0, None, None, None, None, None))(
#        jnp.array((10, 11), dtype=int),
#        jnp.array((5, 5), dtype=int),
        jnp.arange(10, 13),
        #jnp.concatenate(jnp.concatenate((jnp.arange(5, 10)[:, None], jnp.arange(5, 10)[:, None]), axis=1)),
        jnp.array((5, 5, 6,), dtype=int),
        jnp.array(2.0, dtype=float),
        jnp.array(1000.0, dtype=float),
        jnp.array(zf, dtype=float),
        jnp.array(450 + i*100, dtype=float),
        jnp.array(1 + i, dtype=float)
    )

print(iterations)
print(state.TAC)
#plot_function(jnp.arange(1, state.Nstages+1), state.L[0:state.Nstages], state.V[0:state.Nstages], state.temperature[0:state.Nstages], state.Hliq[0:state.Nstages], state.Hvap[0:state.Nstages], state.X[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], state.Y[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], jnp.where(zf != 0, jnp.arange(0, len(zf)), 0))
plot_function(jnp.arange(1, state.Nstages[0]+1), state.L[0, 0:state.Nstages[0]], state.V[0, 0:state.Nstages[0]], state.temperature[0, 0:state.Nstages[0]], state.Hliq[0, 0:state.Nstages[0]], state.Hvap[0, 0:state.Nstages[0]], state.X[0, jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages[0]], state.Y[0, jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages[0]], jnp.where(zf != 0, jnp.arange(0, len(zf)), 0))

plt.show()
'''
'''

'''

state = initialize()
stime = time()
state, iterations, res = jit(vmap(simulation, in_axes=(None, 0, None, None, None, None, None, None)))(
    state,
    jnp.arange(3, 13),
    jnp.array(jnp.floor((co+3)/2+1), dtype=int),
    jnp.array(1., dtype=float),
    jnp.array(1000., dtype=float),
    zf,
    jnp.array(500, dtype=float),
    jnp.array(1., dtype=float)
)
print(time()-stime)
state = initialize()
state, iterations, res = jit(vmap(simulation, in_axes=(None, 0, None, None, None, None, None, None)))(
    state,
    jnp.arange(3, 13),
    jnp.array(jnp.floor((co+3)/2+1), dtype=int),
    jnp.array(1., dtype=float),
    jnp.array(1000., dtype=float),
    zf,
    jnp.array(500, dtype=float),
    jnp.array(1., dtype=float)
)
print(time()-stime)

for tr in range(3):
    stime = time()
    state, iterations, res = jit(vmap(simulation, in_axes=(0, None, None, None, None, None, None)))(
        jnp.arange(3, 20),
        jnp.array(jnp.floor((co+3)/2+1), dtype=int),
        jnp.array(tr+1, dtype=float),
        jnp.array(1000., dtype=float),
        zf,
        jnp.array(500, dtype=float),
        jnp.array(1., dtype=float)
    )
    print(time()-stime)

stime = time()
state, iterations, res = jit(vmap(simulation, in_axes=(0, None, None, None, None, None, None)))(
    jnp.arange(3, 20),
    jnp.array(jnp.floor((co+3)/2+1), dtype=int),
    jnp.array(1., dtype=float),
    jnp.array(1000., dtype=float),
    zf,
    jnp.array(500, dtype=float),
    jnp.array(1., dtype=float)
)
print(time()-stime)

def small_fn(x):
    return (x-5)**2


x_in = jnp.arange(len(tac_array), dtype=float)

fig = plt.figure(4)
ax = fig.add_subplot(111)
ax.plot(x_in, small_fn(x_in))
ax.set_ylabel('Reward')
ax.set_xlabel('Number of stages')
ax.set_title('Model simplification')
ax.grid()


fig4 = plt.figure(5)
ax4 = fig4.add_subplot(111)
# calculate mole fractions

ax4.plot(jnp.arange(len(tac_array))+3, tac_array)
ax4.set_ylabel('TAC reward in M$')
ax4.set_xlabel('stagenumber')
ax4.set_title('Optimization profile test case')
#ax4.legend()
ax4.grid()

plot_function(jnp.arange(1, state.Nstages+1), state.L[0:state.Nstages], state.V[0:state.Nstages], state.temperature[0:state.Nstages], state.Hliq[0:state.Nstages], state.Hvap[0:state.Nstages], state.X[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], state.Y[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], jnp.where(zf != 0, jnp.arange(0, len(zf)), 0))
plt.show()
'''