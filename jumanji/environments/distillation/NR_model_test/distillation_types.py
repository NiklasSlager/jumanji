from typing import TYPE_CHECKING, NamedTuple
if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
import chex





class Tray(NamedTuple):
    l: chex.Array
    v: chex.Array
    T: chex.Array


class Trays(NamedTuple):
    low_tray: Tray
    tray: Tray
    high_tray: Tray


class Mesh(NamedTuple):
    H: jnp.ndarray
    M: jnp.ndarray
    E: jnp.ndarray


class Derivatives(NamedTuple):
    H: Tray
    M: Tray
    E: Tray


class Thermo(NamedTuple):
    psat_params: chex.Array
    hvap_params: chex.Array
    hform_params: chex.Array
    cpvap_params: chex.Array


class FUG_state(NamedTuple):
    components: chex.Array
    z: chex.Array
    pressure: chex.Array
    heavy_key: chex.Array
    light_key: chex.Array
    heavy_x: chex.Array
    light_x: chex.Array
    heavy_spec: chex.Array
    light_spec: chex.Array
    alpha_avg: chex.Array
    feed: chex.Array
    stages: chex.Array
    feed_stage: chex.Array
    reflux: chex.Array
    t_cond: chex.Array
    t_reb: chex.Array
    distillate: chex.Array
    
    
@dataclass
class State:
    L: chex.Array
    V: chex.Array
    U: chex.Array
    W: chex.Array
    X: chex.Array
    Y: chex.Array
    temperature: chex.Array
    F: chex.Array
    components: chex.Array
    z: chex.Array
    pressure: chex.Array
    RR: chex.Array
    distillate: chex.Array
    Nstages: chex.Array
    Hliq: chex.Array
    Hvap: chex.Array
    Hfeed: chex.Array
    RD: chex.Array
    CD: chex.Array
    TAC: chex.Array
    trays: Trays
    heavy_key: chex.Array
    light_key: chex.Array
    heavy_spec: chex.Array
    light_spec: chex.Array
    #thermo: Thermo
    step_count: chex.Numeric  # ()
    action_mask: chex.Array  # (4,)
    key: chex.PRNGKey  # (2,)
    residuals: chex.Array
    analytics: bool
    converged: bool

class Jacobian(NamedTuple):
    A: chex.Array
    B: chex.Array
    C: chex.Array


class NR_State(NamedTuple):
    l: jnp.ndarray
    v: jnp.ndarray
    temperature: jnp.ndarray
    f: jnp.ndarray
    s: jnp.ndarray
    components: jnp.ndarray
    z: jnp.ndarray
    pressure: jnp.ndarray
    RR: jnp.ndarray
    distillate: jnp.ndarray
    Nstages: jnp.ndarray
    trays: Trays
    h_feed: jnp.ndarray


class Observation(NamedTuple):
    grid: chex.Array  # (num_rows, num_cols, 5)
    step_count: chex.Numeric  # Shape ()
    action_mask: chex.Array  # (4,)










