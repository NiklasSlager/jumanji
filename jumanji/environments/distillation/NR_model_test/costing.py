import jax.numpy as jnp
from jumanji.environments.distillation.NR_model_test.distillation_types import State
from jumanji.environments.distillation.NR_model_test.data_property import MW, DENSITY

def diameter(state: State, l_s):
    r_gas = 8.314
    mw_series = MW
    rho_series = DENSITY
    #mw_series = jnp.array([44.097, 58.12, 72.15])
    #rho_series = jnp.array([493.0, 625.0, 626.0])
    rho_v = jnp.where(jnp.sum(state.Y, axis=0) > 0, (state.pressure * 1e5 / (r_gas * state.temperature)*jnp.sum(state.Y*mw_series[:, None], axis=0))/1000, 0)
    rho_l = jnp.sum(state.X*rho_series[:, None], axis=0)

    f_flow = jnp.nan_to_num(state.L*jnp.sum(state.X*mw_series[:, None]/1000, axis=0) / (state.V*jnp.sum(state.Y*mw_series[:, None]/1000, axis=0)) * jnp.sqrt(rho_v/rho_l))
    l_s = l_s * 3.28084
    csb = ((0.26 * l_s - 0.029 * l_s ** 2)/(1 + 6 * f_flow ** 2 * l_s ** 0.7498) ** 0.5)/3.38084
    u_nf = jnp.nan_to_num(csb * (0.015/0.02)**0.2 * jnp.sqrt((rho_l - rho_v)/rho_v))
    a_n = jnp.nan_to_num((state.V/3.6 * jnp.sum(state.Y*mw_series[:, None], axis=0)/1000)/rho_v / (0.85 * u_nf))
    diameter = 2 * jnp.sqrt(a_n/(jnp.pi * (1-jnp.arcsin(0.77)/90) + 2 * 0.77 * jnp.cos(jnp.arcsin(0.77))))

    #u_sup = jnp.where(jnp.sum(state.Y, axis=0) > 0, (-0.171 * l_s**2 + 0.27 * l_s - 0.047)*((rho_l-rho_v)/rho_v)**0.5, 0)
    #d_eff = jnp.sqrt(4*jnp.max(state.V/3.6*state.Y*mw_series[:, None]/1000)/(jnp.pi*rho_v*u_sup))
    return jnp.max(diameter)
    #return jnp.max(state.V/3.6*state.Y*mw_series[:, None]/1000)#jnp.min(jnp.where(rho_v > 0, rho_v, 1e5))#jnp.max(jnp.where(u_sup > 0, (jnp.sqrt(4 * jnp.max(state.V / 3.6 * state.Y * mw_series[:, None] / 1000) / (jnp.pi * rho_v * u_sup))), 0))


def column_instalcost(state: State, ms):
    spacing = jnp.array(0.606, dtype=float)
    d_column = diameter(state, spacing)
    sump = jnp.array(4, dtype=int)
    height = spacing*(state.Nstages+sump)
    f_m = 1
    f_p = 1
    column_cost = ms/280*937.64*d_column**1.066*height**0.802*f_m*f_p

    f_s = 1.4
    f_t = 0
    f_m = 0
    internals_cost = ms/280*97.24*d_column**1.55*(spacing*state.Nstages)*(f_s + f_t + f_m)
    return column_cost #+ internals_cost


def hex_instalcost(state: State, ms):
    f_p = 0
    f_d = 0.8
    f_m = 1

    t_in = 30+273
    t_out = 40+273
    #log_temp = ((state.temperature[0] - t_in) * (state.temperature[0] - t_out) * ((state.temperature[0] - t_in) + (state.temperature[0] - t_out)) / 2) ** (3 / 2)
    log_temp = 15.
    k_cd = 500.
    area_cd = state.CD/0.0036/(k_cd*log_temp)
    cost_cd = ms/280*474.67*area_cd**0.65*(f_p+f_d)*f_m

    k_rb = 800
    t_steam = 250+273
    area_rb = jnp.abs(state.RD/0.0036)/(k_rb*(t_steam-state.temperature[state.Nstages-1]))
    cost_rb = ms/280*474.67*area_rb**0.65*(f_p+f_d)*f_m
    return cost_cd + cost_rb


def installed_cost(state: State):
    ms = jnp.array(2971.6, dtype=float)
    cost_eqp = (column_instalcost(state, ms) + hex_instalcost(state, ms))/1e6
    f_lang = jnp.array(5., dtype=float)
    return cost_eqp*f_lang


def operational_cost(state: State):
    uptime = 8000
    e_cd = uptime*state.CD/1000
    e_cost = jnp.array(1., dtype=float)
    cd_cost = e_cd*e_cost/1e6

    e_rb = uptime*jnp.abs(state.RD)/1000
    e_cost = jnp.array(20., dtype=float)
    rb_cost = e_rb*e_cost/1e6
    return cd_cost + rb_cost

def tac(state: State):
    depreciation = jnp.array(5., dtype=float)  #years
    return state.replace(TAC=(installed_cost(state)/depreciation + operational_cost(state)))
