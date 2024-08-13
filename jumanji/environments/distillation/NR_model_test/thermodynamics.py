import jax.numpy as jnp
from jaxopt import Bisection
from jax import vmap
from jumanji.environments.distillation.NR_model_test.distillation_types import State, Thermo
import os
from jumanji.environments.distillation.NR_model_test.physical_data_sets import psat_params, cpvap_params, hvap_params, hform_params
#from jumanji.environments.distillation.NR_model_test.data_property import DHV, HFORM, CPIG, PSAT

CPIG = jnp.array([[14.2051,	30.2403,	844.31,	20.5802,	2482.7,	298.15,	1500],
        [18.2464,	40.1309,	826.54,	24.5653,	2483.1,	298.15,	1500],
        [19.1445,	38.7934,	841.49,	25.258,	    2476.1,	298.15,	1500],
        [21.0304,	71.9165,	1650.2,	45.1896,	747.6,	200,	1500],
        [9.93599,	71.9882,	1461.7,	43.2192,	668.8,	100,	1500],
        [24.9355,	84.1454,	1694.6,	56.5826,	761.6,	200,	1500],
        [22.762,	114.742,	1573,	69.5997,	664.3,	200,	1500],
        [28.6973,	95.5622,	1676.6,	65.4438,	756.4,	200,	1500],
        [25.6043,	119.733,	1541.5,	78.6997,	660,	200,	1500],
        [32.3732,	105.833,	1635.6,	72.9435,	746.4,	200,	1500]], dtype=float)

DHV = jnp.array([[6.97645,	0.78237,	-0.77319,	0.39246,	0,	-187.68,	96.68],
        [9.4712,	1.274,	    -1.4255,	0.60708,	0,	-159.61,	134.65],
        [8.6553,	0.8337,	    -0.82274,	0.39613,	0,	-138.29,	151.97],
        [10.7688,	0.95886,	-0.92384,	0.39393,	0,	-129.73,	196.55],
        [8.17235,	-0.21723,	1.0245, 	-0.49752,	0,	-93.87,	238.55],
        [10.4729,	0.34057,	0.063282,	-0.017037,	0,	-95.32,	234.45],
        [11.5062,	0.75694,	-0.61674,	0.26462,	0,	-24.58,	257.95],
        [12.5432,	0.51283,	-0.10982,	-0.01018,	0,	-90.58,	267.05],
        [16.0989,	1.3555,	    -1.4474,	0.55232,	0,	-91.15,	276.85],
        [16.0356,	1.0769,	    -1.0124,	0.37075,	0,	-56.77,	295.55]], dtype=float)

PSAT = jnp.array([[47.5651,	-3492.6,	0,	0,	-6.0669,	1.09E-05,	2,	85.47,	369.83],
        [96.9171,	-5039.9,	0,	0,	-15.012,	0.022725,	1,	113.54,	407.8],
        [54.8301,	-4363.2,	0,	0,	-7.046,	    9.45E-06,	2,	134.86,	425.12],
        [67.2281,	-5420.3,	0,	0,	-8.8253,	9.62E-06,	2,	143.42,	469.7],
        [54.8281,	-5198.5,	0,	0,	-6.8103,	6.19E-06,	2,	179.28,	511.7],
        [93.1371,	-6995.5,	0,	0,	-12.702,	1.24E-05,	2,	177.83,	507.6],
        [67.8701,	-6127.1,	0,	0,	-8.7696,	7.36E-06,	2,	248.57,	531.1],
        [76.3161,	-6996.4,	0,	0,	-9.8802,	7.21E-06,	2,	182.57,	540.2],
        [85.5431,	-7517.2,	0,	0,	-11.282,	8.34E-06,	2,	182,	550],
        [84.5711,	-7900.2,	0,	0,	-11.003,	7.18E-06,	2,	216.38,	568.7]], dtype=float)

HFORM = jnp.array([-25.0024,
                   -32.2418,
                   -30.0444,
                   -35.053,
                   -18.3983,
                   -39.8729,
                   -48.8273,
                   -44.8194,
                   -53.1456,
                   -49.8591
                   ], dtype=float) * jnp.array(4.184, dtype=float)  #KCAL/mol

def p_sat(temp, compound):
    #dir = os.path.join(os.getcwd(), 'Pure component parameters')
    params = PSAT[compound]
    a, b, c, d, e, f, g, h, i = params
    return jnp.exp(a + b / (temp + c) + d * temp + e * jnp.log(temp) + f * temp ** g)


def t_sat(compound, tguess, pressure):
    def equality(tempcalc, compound, pressure):
        return 1 - p_sat(tempcalc, compound) / pressure
    return Bisection(optimality_fun=equality, lower=500., upper=100., check_bracket=False).run(tguess, compound, pressure).params


def k_eq(temp, component, pressure):

    result = p_sat(temp, component)/pressure
    return result

def cp_vap(temperature, compound):
    params = CPIG[compound]
    a, b, c, d, e, f, g = params

    def cp(temp):
        return a*temp + b*c/(jnp.tanh(c/temp))-d*e*jnp.tanh(e/temp)

    return (cp(temperature)-cp(298.15)) / 1000 * 4.184


def h_evap(temperature, compound):
    params = DHV[compound]
    a, b, c, d, e, f, g = params

    tc = g + 273.15
    tr = temperature / tc
    return jnp.where(tr < 1, (a * (1 - tr) ** (b + c * tr + d * tr ** 2 + e * tr ** 3)) * 4.184, 0)


def liquid_enthalpy(temperature, components):
    def calculate_h_liq(temperature, i):
        return HFORM[i] + cp_vap(temperature, i) - h_evap(temperature, i)
    return vmap(calculate_h_liq, in_axes=(None, 0))(temperature, components)


def vapor_enthalpy(temperature, components):
    def calculate_h_vap(temperature, i):
        return HFORM[i] + cp_vap(temperature, i)
    return vmap(calculate_h_vap, in_axes=(None, 0))(temperature, components)


def t_solver(state, tray, temperature, xc):
    def function(tempcalc):
        return 1 - jnp.sum(vmap(k_eq, in_axes=(None, 0, None))(tempcalc, state.components, state.pressure) * xc)
    result = Bisection(optimality_fun=function, lower=500., upper=100., check_bracket=False).run(temperature[tray]).params
    return result


def feed_enthalpy(state: State):
    feed_temp = t_solver(state, None, state.pressure, jnp.array(state.z))
    return liquid_enthalpy(feed_temp[0], state.components)
