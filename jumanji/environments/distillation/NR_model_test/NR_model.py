import jax.numpy as jnp
from time import time
from jax import jit, vmap

import matplotlib.pyplot as plt
from jumanji.environments.distillation.NR_model_test.plot_generation import plot_function
from jumanji.environments.distillation.NR_model_test.NR_model import inside_simulation as simulation
from jumanji.environments.distillation.NR_model_test.NR_model import initialize

state_init= initialize()
zf = jnp.array([0., 0.1, 0.2, 0.25, 0.25, 0., 0.15, 0.1])
zf = zf/jnp.sum(zf)
iterations = 0
st = time()

state, iterations, res = (simulation)(
    state=state_init,
    nstages=jnp.array(14, dtype=int),
    feedstage=jnp.array(5, dtype=int),
    pressure=jnp.array(1.0, dtype=float),
    feed=jnp.array(1000.0, dtype=float),
    z=jnp.array(zf, dtype=float),
    distillate=jnp.array(55, dtype=float),
    rr=jnp.array(3.1, dtype=float),
    specs=False
)
print(iterations)
print(state.converged)
plot_function(jnp.arange(1, state.Nstages+1), state.L[0:state.Nstages], state.V[0:state.Nstages], state.temperature[0:state.Nstages], state.Hliq[0:state.Nstages], state.Hvap[0:state.Nstages], state.X[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], state.Y[jnp.where(zf != 0, jnp.arange(0, len(zf)), 0), 0:state.Nstages], jnp.where(zf != 0, jnp.arange(0, len(zf)), 0))
plt.show()
