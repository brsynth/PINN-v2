# # Lorena's Bioprocess Model

# This model aims to describe a growth system of *Pseudomonas putida* in different
# set-ups. The model not only studies the microorganism’s growth dynamics
# but also the consumption of a carbon source (substrate) and other nutrients
# (chemical elements). Additionally, it examines the dynamics of gases that may
# be used in the process, including both those in the gas phase of the system and
# those dissolved in the liquid phase.
# 
# For further information, please check the [*BIOS_Pseudomonas_model.pdf*](../misc/BIOS_Pseudomonas_model.pdf) document associated with this notebook in the *misc* folder.

# Lorena Bioprocess Model — Modular ODEs with flexible fed modes
# Requirements: numpy, scipy

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Iterable, Optional, Union, Sequence
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from pathlib import Path
import json
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import seaborn as sns
import torch 

#----------------------------------------------------------------------------------------------------------------------------
# BIOPROCESS MODELING 
# ----------------------------
# [A] Algebraic relationships
# ----------------------------

def mu(Sub: float, params: dict) -> float:
    """Specific growth rate μ = μmax * Sub / (Ksubs + Sub)."""
    mu_max = params["mu_max"]["values"][0]
    K_subs = params["K_subs"]["values"][0]
    return mu_max * Sub / (K_subs + max(Sub, 0.0))

def partial_pressure_O2(params: dict) -> float:
    """Partial pressure of O2 = FractionO2 * Pr."""
    return params["FractionO2"]["values"][0] * params["Pr"]["values"][0]

def partial_pressure_CO2(params: dict) -> float:
    """Partial pressure of CO2 = FractionCO2 * Pr."""
    return params["FractionCO2"]["values"][0] * params["Pr"]["values"][0]

def Csat_O2(params: dict) -> float:
    """CSatO2 = PartialPressureO2 * HO2."""
    return partial_pressure_O2(params) * params["H_O2"]["values"][0]

def Csat_CO2(params: dict) -> float:
    """CSatCO2 = PartialPressureCO2 * HCO2."""
    return partial_pressure_CO2(params) * params["H_CO2"]["values"][0]

def n_gas_moles(params: dict) -> float:
    """n = (Pr * Vgas) / (R * TKelvin)."""
    Pr = params["Pr"]["values"][0]
    Vgas = params["V_gas"]["values"][0]
    R = params["R"]["values"][0]
    TK = params["T_Kelvin"]["values"][0]
    return (Pr * Vgas) / (R * TK)

def inventory_O2_liq(params: dict) -> float:
    """IC_l_O2 = CSatO2 * V_l."""
    return Csat_O2(params) * params["V_l"]["values"][0]

def inventory_CO2_liq(params: dict) -> float:
    """IC_l_CO2 = CSatCO2 * V_l."""
    return Csat_CO2(params) * params["V_l"]["values"][0]

def inventory_O2_gas(params: dict) -> float:
    """IC_g_O2 = FractionO2 * n."""
    return params["FractionO2"]["values"][0] * n_gas_moles(params)

def inventory_CO2_gas(params: dict) -> float:
    """IC_g_CO2 = FractionCO2 * n."""
    return params["FractionCO2"]["values"][0] * n_gas_moles(params)

# Bundle algebraics (easy to import/use individually)
ALGEBRAICS: Dict[str, Callable[..., float]] = {
    "mu": mu,
    "partial_pressure_O2": partial_pressure_O2,
    "partial_pressure_CO2": partial_pressure_CO2,
    "Csat_O2": Csat_O2,
    "Csat_CO2": Csat_CO2,
    "n_gas_moles": n_gas_moles,
    "inventory_O2_liq": inventory_O2_liq,
    "inventory_CO2_liq": inventory_CO2_liq,
    "inventory_O2_gas": inventory_O2_gas,
    "inventory_CO2_gas": inventory_CO2_gas,
}

# %%
# ----------------------------
# [B] Fed-mode schedules
# ----------------------------
@dataclass
class Mode:
    """Callable schedules for time-varying flows (min units)."""
    Vf: Callable[[float], float]        # media feed (L/min)
    Vs: Callable[[float], float]        # liquid outflow/sampling (L/min)
    VGasIn: Callable[[float], float]    # gas in (L/min)
    VOffGas: Callable[[float], float]   # gas out (L/min)
    Va: Callable[[float], float]        # acid (L/min)
    Vb: Callable[[float], float]        # base (L/min)

def make_mode(mode: str, consts: dict, params: dict) -> Mode:
    """Create flow schedules for 'fed-batch' (default), 'batch', or 'continuous'."""
    Vf0 = consts.get("Vf_const", 0.0)
    Vs0 = consts.get("Vs_const", 0.0)
    D   = consts.get("D_const", None)  # for continuous
    VGasIn0 = consts.get("VGasIn_const", 0.0)
    VOffGas0 = consts.get("VOffGas_const", 0.0)
    Va0 = consts.get("Va_const", 0.0)
    Vb0 = consts.get("Vb_const", 0.0)

    if mode.lower() == "batch":
        return Mode(
            Vf=lambda t: 0.0, Vs=lambda t: 0.0,
            VGasIn=lambda t: VGasIn0, VOffGas=lambda t: VOffGas0,
            Va=lambda t: 0.0, Vb=lambda t: 0.0
        )
    elif mode.lower() in ("continuous", "chemostat"):
        # If D (1/min) is provided, Vs = D * V_l, Vf ~ Vs for constant volume.
        if D is None:
            raise ValueError("For 'continuous' mode, provide D_const in consts.")
        return Mode(
            Vf=lambda t: Vs0 if Vs0 > 0 else 0.0,  # user can override Vs_const
            Vs=lambda t: Vs0,                      # or leave both zero and manage via D in RHS
            VGasIn=lambda t: VGasIn0, VOffGas=lambda t: VOffGas0,
            Va=lambda t: Va0, Vb=lambda t: Vb0
        )
    else:  # fed-batch (default)
        return Mode(
            Vf=lambda t: Vf0, Vs=lambda t: Vs0,
            VGasIn=lambda t: VGasIn0, VOffGas=lambda t: Vs0 + Vf0 + params["V_g"]["values"][0],
            Va=lambda t: Va0, Vb=lambda t: Vb0
        )

# %%
# ----------------------------
# [C] State indexing helper
# ----------------------------
STATE_ORDER = [
    "Sub", "Biomass",
    "Ca","Cl","Co","Cu","Fe","Mg","Mo","Na","Zn","K","Ni","NH4","P","S",
    "CO2_l","CO2_g","O2_l","O2_g","H","OH"
]
IDX = {name:i for i,name in enumerate(STATE_ORDER)}

# %%
# ----------------------------
# [D] Differential equations
#    (each one is independently callable)
# ----------------------------

def dBiomass_dt(t, y, p, a, mode: Mode):
    Biomass = y[IDX["Biomass"]]
    V_l = p["V_l"]["values"][0]
    Vs = mode.Vs(t)
    mu_val = a["mu"](y[IDX["Sub"]], p)
    return mu_val * Biomass - (Biomass / V_l) * Vs

def dSub_dt(t, y, p, a, mode: Mode):
    Sub, Biomass = y[IDX["Sub"]], y[IDX["Biomass"]]
    V_l = p["V_l"]["values"][0]
    Vf, Vs = mode.Vf(t), mode.Vs(t)
    CSub_f = p["CSub_f"]["values"][0]
    Y_Sub = p["Y_Sub"]["values"][0]
    mu_val = a["mu"](Sub, p)
    return CSub_f * Vf - (Sub / V_l) * Vs - (1.0 / Y_Sub) * mu_val * Biomass

def _template_dTrace_dt(key: str, Cfeed_key: str, Y_key: str):
    def f(t, y, p, a, mode: Mode):
        X   = y[IDX[key]]
        Biomass = y[IDX["Biomass"]]
        V_l = p["V_l"]["values"][0]
        Vf, Vs = mode.Vf(t), mode.Vs(t)
        C_f = p[Cfeed_key]["values"][0]
        Y = p[Y_key]["values"][0]
        mu_val = a["mu"](y[IDX["Sub"]], p)
        return C_f * Vf - (X / V_l) * Vs - (1.0 / Y) * mu_val * Biomass
    return f

# Trace elements following Eq. (3)–(15)
dCa_dt  = _template_dTrace_dt("Ca",  "CCa_f",  "Y_Ca")
dCl_dt  = _template_dTrace_dt("Cl",  "CCl_f",  "Y_Cl")
dCo_dt  = _template_dTrace_dt("Co",  "CCo_f",  "Y_Co")
dCu_dt  = _template_dTrace_dt("Cu",  "CCu_f",  "Y_Cu")
dFe_dt  = _template_dTrace_dt("Fe",  "CFe_f",  "Y_Fe")
dMg_dt  = _template_dTrace_dt("Mg",  "CMg_f",  "Y_Mg")
dMo_dt  = _template_dTrace_dt("Mo",  "CMo_f",  "Y_Mo")
# Na has an extra base flow term (Eq. 10)
def dNa_dt(t, y, p, a, mode: Mode):
    Na, Biomass = y[IDX["Na"]], y[IDX["Biomass"]]
    V_l = p["V_l"]["values"][0]
    Vf, Vs, Vb = mode.Vf(t), mode.Vs(t), mode.Vb(t)
    C_f = p["CNa_f"]["values"][0]
    C_b = p["CNa_b"]["values"][0]
    Y = p["Y_Na"]["values"][0]
    mu_val = a["mu"](y[IDX["Sub"]], p)
    return C_f * Vf + C_b * Vb - (Na / V_l) * Vs - (1.0 / Y) * mu_val * Biomass

dZn_dt  = _template_dTrace_dt("Zn",  "CZn_f",  "Y_Zn")
dK_dt   = _template_dTrace_dt("K",   "CK_f",   "Y_K")
dNi_dt  = _template_dTrace_dt("Ni",  "CNi_f",  "Y_Ni")
dNH4_dt = _template_dTrace_dt("NH4", "CNH4_f", "Y_NH4")
dP_dt   = _template_dTrace_dt("P",   "CP_f",   "Y_P")
# Sulphate (note: coefficient Y_S appears as direct proportional, per Eq. 16)
def dS_dt(t, y, p, a, mode: Mode):
    S, Biomass = y[IDX["S"]], y[IDX["Biomass"]]
    V_l = p["V_l"]["values"][0]
    Vf, Vs = mode.Vf(t), mode.Vs(t)
    C_f = p["CS_f"]["values"][0]
    YS = p["Y_S"]["values"][0]
    mu_val = a["mu"](y[IDX["Sub"]], p)
    return C_f * Vf - (S / V_l) * Vs - YS * mu_val * Biomass

# CO2 and O2 (Eqs. 17–20)
def dCO2_l_dt(t, y, p, a, mode: Mode):
    CO2_l, Biomass = y[IDX["CO2_l"]], y[IDX["Biomass"]]
    V_l = p["V_l"]["values"][0]
    KLa = p["K_L_a"]["values"][0]
    CO2_sat = a["Csat_CO2"](p)
    Y_CO2 = p["Y_CO2"]["values"][0]
    mu_val = a["mu"](y[IDX["Sub"]], p)
    return KLa * (CO2_sat - CO2_l) * V_l + (1.0 / Y_CO2) * mu_val * Biomass

def dCO2_g_dt(t, y, p, a, mode: Mode):
    CO2_g = y[IDX["CO2_g"]]
    V_gas = p["V_gas"]["values"][0]
    KLa = p["K_L_a"]["values"][0]
    CO2_sat = a["Csat_CO2"](p)
    VGasIn, VOffGas = mode.VGasIn(t), mode.VOffGas(t)
    CO2_Gasin = p["CO2_Gasin"]["values"][0]
    CO2_l = y[IDX["CO2_l"]]
    return (CO2_Gasin * VGasIn
            - (KLa * (CO2_sat - CO2_l) * p["V_l"]["values"][0])
            - (CO2_g / V_gas) * VOffGas)

def dO2_l_dt(t, y, p, a, mode: Mode):
    O2_l, Biomass = y[IDX["O2_l"]], y[IDX["Biomass"]]
    V_l = p["V_l"]["values"][0]
    KLa = p["K_L_a"]["values"][0]
    O2_sat = a["Csat_O2"](p)
    Y_O2 = p["Y_O2"]["values"][0]
    mu_val = a["mu"](y[IDX["Sub"]], p)
    return KLa * (O2_sat - O2_l) * V_l - (1.0 / Y_O2) * mu_val * Biomass

def dO2_g_dt(t, y, p, a, mode: Mode):
    O2_g = y[IDX["O2_g"]]
    V_gas = p["V_gas"]["values"][0]
    KLa = p["K_L_a"]["values"][0]
    O2_sat = a["Csat_O2"](p)
    VGasIn, VOffGas = mode.VGasIn(t), mode.VOffGas(t)
    O2_Gasin = p["O2_Gasin"]["values"][0]
    O2_l = y[IDX["O2_l"]]
    return (O2_Gasin * VGasIn
            - (KLa * (O2_sat - O2_l) * p["V_l"]["values"][0])
            - (O2_g / V_gas) * VOffGas)

# Acid/base species (Eqs. 21–22)
def dH_dt(t, y, p, a, mode: Mode):
    H, Biomass = y[IDX["H"]], y[IDX["Biomass"]]
    V_l = p["V_l"]["values"][0]
    Vf, Vs, Va = mode.Vf(t), mode.Vs(t), mode.Va(t)
    C_H_f = p["CH_f"]["values"][0]
    C_H_a = p["CH_a"]["values"][0]
    Y_H = p["Y_H"]["values"][0]
    mu_val = a["mu"](y[IDX["Sub"]], p)
    return C_H_f * Vf + C_H_a * Va - (H / V_l) * Vs + (1.0 / Y_H) * mu_val * Biomass

def dOH_dt(t, y, p, a, mode: Mode):
    OH = y[IDX["OH"]]
    V_l = p["V_l"]["values"][0]
    Vf, Vs, Vb = mode.Vf(t), mode.Vs(t), mode.Vb(t)
    C_OH_f = p["COH_f"]["values"][0]
    C_OH_b = p["COH_b"]["values"][0]
    return C_OH_f * Vf + C_OH_b * Vb - (OH / V_l) * Vs

# Bundle all ODEs for composability
ODES: Dict[str, Callable[..., float]] = {
    "dSub_dt": dSub_dt,
    "dBiomass_dt": dBiomass_dt,
    "dCa_dt": dCa_dt, "dCl_dt": dCl_dt, "dCo_dt": dCo_dt, "dCu_dt": dCu_dt,
    "dFe_dt": dFe_dt, "dMg_dt": dMg_dt, "dMo_dt": dMo_dt, "dNa_dt": dNa_dt,
    "dZn_dt": dZn_dt, "dK_dt": dK_dt, "dNi_dt": dNi_dt, "dNH4_dt": dNH4_dt,
    "dP_dt": dP_dt, "dS_dt": dS_dt,
    "dCO2_l_dt": dCO2_l_dt, "dCO2_g_dt": dCO2_g_dt,
    "dO2_l_dt": dO2_l_dt,   "dO2_g_dt": dO2_g_dt,
    "dH_dt": dH_dt, "dOH_dt": dOH_dt
}

# Master RHS
def rhs(t: float, y: np.ndarray, params: dict, mode: Mode) -> np.ndarray:
    a  = ALGEBRAICS
    dy = np.zeros_like(y)
    
    dy[IDX["Sub"]]     = dSub_dt(t, y, params, a, mode)
    dy[IDX["Biomass"]] = dBiomass_dt(t, y, params, a, mode)
    dy[IDX["Ca"]]      = dCa_dt(t, y, params, a, mode)
    dy[IDX["Cl"]]      = dCl_dt(t, y, params, a, mode)
    dy[IDX["Co"]]      = dCo_dt(t, y, params, a, mode)
    dy[IDX["Cu"]]      = dCu_dt(t, y, params, a, mode)
    dy[IDX["Fe"]]      = dFe_dt(t, y, params, a, mode)
    dy[IDX["Mg"]]      = dMg_dt(t, y, params, a, mode)
    dy[IDX["Mo"]]      = dMo_dt(t, y, params, a, mode)
    dy[IDX["Na"]]      = dNa_dt(t, y, params, a, mode)
    dy[IDX["Zn"]]      = dZn_dt(t, y, params, a, mode)
    dy[IDX["K"]]       = dK_dt(t, y, params, a, mode)
    dy[IDX["Ni"]]      = dNi_dt(t, y, params, a, mode)
    dy[IDX["NH4"]]     = dNH4_dt(t, y, params, a, mode)
    dy[IDX["P"]]       = dP_dt(t, y, params, a, mode)
    dy[IDX["S"]]       = dS_dt(t, y, params, a, mode)
    dy[IDX["CO2_l"]]   = dCO2_l_dt(t, y, params, a, mode)
    dy[IDX["CO2_g"]]   = dCO2_g_dt(t, y, params, a, mode)
    dy[IDX["O2_l"]]    = dO2_l_dt(t, y, params, a, mode)
    dy[IDX["O2_g"]]    = dO2_g_dt(t, y, params, a, mode)
    dy[IDX["H"]]       = dH_dt(t, y, params, a, mode)
    dy[IDX["OH"]]      = dOH_dt(t, y, params, a, mode)
    return dy

# %%
# -------------------------------------------
# [E] Minimal parameter & initial dictionaries
# -------------------------------------------

def minimal_parameters() -> Dict[str, Dict[str, List]]:
    """
    Structure: {param_name: {"values": [possible_values], "unit": "..."}}.
    Values below are sensible placeholders; replace with your true settings.
    Units and symbols follow your PDF tables. :contentReference[oaicite:1]{index=1}
    """
    minimal_prm = {
        # Volumes (assumed constant unless you later model dV/dt)
        "V_l"  : {"values": [0.25],  "unit": "L"},
        "V_g"  : {"values": [0.105], "unit": "L"}, # volume gas phase
        "V_gas": {"values": [0.05],  "unit": "L"}, # volume gas phase (from exp)

        # Gas composition & thermodynamics
        "FractionO2" : {"values": [0.21],     "unit": "-"},
        "FractionCO2": {"values": [0.000407], "unit": "-"},
        "Pr"         : {"values": [101325.0], "unit": "Pa"},
        "H_O2"       : {"values": [1.3e-8],   "unit": "mol/(L·Pa)"},
        "H_CO2"      : {"values": [3.4e-7],   "unit": "mol/(L·Pa)"},
        "R"          : {"values": [8.314],    "unit": "J/(mol·K)"},
        "T_Scale"    : {"values": [273.15],   "unit": "K"}, # DD!
        "T_Celsius"  : {"values": [30.0],     "unit": "°C"},# DD!
        "T_Kelvin"   : {"values": [303.15],   "unit": "K"}, # def: 298.15K (273.15K + 25°C)  
        
        # Gas feed composition
        "O2_Gasin" : {"values": [0.21],   "unit": "mol/L_gas"},
        "CO2_Gasin": {"values": [0.00407],   "unit": "mol/L_gas"}, # def: 0.00

        # Liquid feeds (media/acid/base concentrations)
        "CSub_f" :  {"values": [0.000166]             , "unit": "mol/L"},
        "CCa_f"  :  {"values": [0.0]                  , "unit": "mol/L"},
        "CCl_f"  :  {"values": [0.000173]             , "unit": "mol/L"},
        "CCo_f"  :  {"values": [8.405819516967988e-07], "unit": "mol/L"},
        "CCu_f"  :  {"values": [5.865708131748498e-08], "unit": "mol/L"},
        "CFe_f"  :  {"values": [1.798470580618242e-06], "unit": "mol/L"},
        "CMg_f"  :  {"values": [1.622889229695627e-04], "unit": "mol/L"},
        "CMo_f"  :  {"values": [1.239887682707782e-07], "unit": "mol/L"},
        "CNa_f"  :  {"values": [0.009841767287091]    , "unit": "mol/L"},
        "CNa_b"  :  {"values": [4.0]                  , "unit": "mol/L"},
        "CZn_f"  :  {"values": [3.477547216397331e-07], "unit": "mol/L"},
        "CK_f"   :  {"values": [0.004408992875802]    , "unit": "mol/L"},
        "CNi_f"  :  {"values": [8.414299934452604e-08], "unit": "mol/L"},
        "CNH4_f" :  {"values": [0.006054221608728]    , "unit": "mol/L"},
        "CP_f"   :  {"values": [0.00863557426884131]  , "unit": "mol/L"},
        "CS_f"   :  {"values": [0.00319746255924881]  , "unit": "mol/L"},
        "CH_f"   :  {"values": [0.0144375593208705]   , "unit": "mol/L"},
        "CH_a"   :  {"values": [8.64]                 , "unit": "mol/L"},
        "COH_f"  :  {"values": [0.0]                  , "unit": "mol/L"},
        "COH_b"  :  {"values": [4.0]                  , "unit": "mol/L"},

        # Stoichiometries / yields
        "Y_Sub" : {"values": [1.0]          , "unit": "g_biomass/mol_Sub"},
        "Y_Ca"  : {"values": [1.0]          , "unit": "mol/mol"},
        "Y_Cl"  : {"values": [9.16]         , "unit": "mol/mol"},
        "Y_Co"  : {"values": [8.773]        , "unit": "mol/mol"},
        "Y_Cu"  : {"values": [7.73]         , "unit": "mol/mol"},
        "Y_Fe"  : {"values": [14.46]        , "unit": "mol/mol"},
        "Y_Mg"  : {"values": [10.633]       , "unit": "mol/mol"},
        "Y_Mo"  : {"values": [7.773]        , "unit": "mol/mol"},
        "Y_Na"  : {"values": [7.467]        , "unit": "mol/mol"},
        "Y_Zn"  : {"values": [0.00000002773], "unit": "mol/mol"},
        "Y_K"   : {"values": [0.156]        , "unit": "mol/mol"},
        "Y_Ni"  : {"values": [0.00000006773], "unit": "mol/mol"},
        "Y_NH4" : {"values": [0.006046]     , "unit": "mol/mol"},
        "Y_P"   : {"values": [0.01193]      , "unit": "mol/mol"},
        "Y_S"   : {"values": [0.0003809]    , "unit": "mol/mol"},
        "Y_CO2" : {"values": [20.43]        , "unit": "mol/g_biomass"},
        "Y_O2"  : {"values": [18.74]        , "unit": "mol/g_biomass"},
        "Y_H"   : {"values": [0.09618]      , "unit": "mol/g_biomass"},

        # Kinetics to estimate (Table 4)
        "mu_max": {"values": [0.0075], "unit": "1/min"},
        "K_subs": {"values": [0.005] , "unit": "mol/L"},
        "K_L_a" : {"values": [2.0]   , "unit": "1/min"}, # Gas transfer
    }

    # convertion updates (to comply with the units in the dict)
    # change T_Kelvin based on T_Celsius: T_Kelvin = T_Celsius + T_Scale 
    minimal_prm["T_Kelvin"]["values"][0] = minimal_prm["T_Scale"]["values"][0] + minimal_prm["T_Celsius"]["values"][0]
    # CFeeds divided by the right Vl: CF_i/V_l 
    for i in ["CSub_f","CCa_f","CCl_f","CCo_f","CCu_f","CFe_f","CMg_f","CMo_f","CNa_f","CZn_f","CK_f","CNi_f","CNH4_f","CP_f","CS_f","CH_f","COH_f"]:
        minimal_prm[i]["values"][0] = minimal_prm[i]["values"][0]/minimal_prm["V_l"]["values"][0] 
    # gas comp
    minimal_prm["CO2_Gasin"]["values"][0] = inventory_CO2_gas(minimal_prm)/minimal_prm["V_gas"]["values"][0]
    minimal_prm["O2_Gasin"]["values"][0]  = inventory_O2_gas(minimal_prm)/minimal_prm["V_gas"]["values"][0]

    return minimal_prm

def reset_and_update_inputs_params(initprms, inputs, prms: list):
    """
    change the specified parameters from an "inputs" structure
    """
    # convertion updates (to comply with the units in the dict)
    for prm in prms:
        if prm == "T_Celsius": # change T_Kelvin based on T_Celsius: T_Kelvin = T_Celsius + T_Scale 
            inputs["params"]["T_Kelvin"]["values"][0] = inputs["params"]["T_Scale"]["values"][0] + inputs["params"]["T_Celsius"]["values"][0]
        elif prm == "V_l":             # CFeeds divided by the right Vl: CF_i/V_l 
            for i in ["CSub_f","CCa_f","CCl_f","CCo_f","CCu_f","CFe_f","CMg_f","CMo_f","CNa_f","CZn_f","CK_f","CNi_f","CNH4_f","CP_f","CS_f","CH_f","COH_f"]:
                inputs["params"][i]["values"][0] = initprms[i]["values"][0]
                inputs["params"][i]["values"][0] = inputs["params"][i]["values"][0]/inputs["params"]["V_l"]["values"][0] 
        elif prm == "V_gas":         # gas comp
            inputs["params"]["CO2_Gasin"]["values"][0] = inventory_CO2_gas(inputs["params"])/inputs["params"]["V_gas"]["values"][0]
            inputs["params"]["O2_Gasin"]["values"][0]  = inventory_O2_gas(inputs["params"])/inputs["params"]["V_gas"]["values"][0]
    return inputs


def minimal_initials() -> Dict[str, Dict[str, List]]:
    """
    Structure: {state_name: {"values":[...], "unit":"..."}} for STATE_ORDER.
    Units match your table (mol or g). :contentReference[oaicite:2]{index=2}
    """
    inits = {}
    # Default small nonzero to avoid divide-by-zero edge cases
    defaults = {
        "Sub": 0.13, "Biomass": 0.04267,
        "Ca":0.0,"Cl":0.000173,"Co":8.405819516967988e-07,"Cu":5.865708131748498e-08,"Fe":1.798470580618242e-06,"Mg":1.622889229695627e-04,"Mo":1.239887682707782e-07,
        "Na":0.009841767287091,"Zn":3.477547216397331e-07,"K":0.004408992875802,"Ni":8.414299934452604e-08,"NH4":0.006054221608728,"P":0.00863557426884131,"S":0.00319746255924881,
        "CO2_l":3.5053383749999997e-06,"CO2_g":0.0017180378517745538,"O2_l":6.915431250000001e-05,"O2_g":0.8864568768369933,"H":0.0144375593208705,"OH":0.0
    }
    units = {
        "Sub":"mol","Biomass":"g","Ca":"mol","Cl":"mol","Co":"mol","Cu":"mol",
        "Fe":"mol","Mg":"mol","Mo":"mol","Na":"mol","Zn":"mol","K":"mol",
        "Ni":"mol","NH4":"mol","P":"mol","S":"mol","CO2_l":"mol","CO2_g":"mol",
        "O2_l":"mol","O2_g":"mol","H":"mol","OH":"mol"
    }
    for k in STATE_ORDER:
        inits[k] = {"values":[defaults[k]], "unit": units[k]}
    return inits

# %%
# -------------------------------------------
# [F] Convenience: pack y0 from initials dict
# -------------------------------------------
def pack_y0(initials: Dict[str, Dict[str, List]]) -> np.ndarray:
    y0 = np.zeros(len(STATE_ORDER))
    for name, idx in IDX.items():
        y0[idx] = float(initials[name]["values"][0])
    return y0

# %%
# -------------------------------------------
# [G] Run utility
# -------------------------------------------
def run_model(
    params: Dict[str, Dict[str, List]],
    initials: Dict[str, Dict[str, List]],
    mode_name: str = "fed-batch",
    mode_consts: dict = None,
    t_span: Tuple[float, float] = (0.0, 600.0),
    t_eval: np.ndarray = None,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9
):
    """
    Execute the bioprocess model.
    - mode_name: 'fed-batch' (default), 'batch', or 'continuous'
    - mode_consts: e.g. {"Vf_const": 0.01, "Vs_const": 0.0, "VGasIn_const": 0.1, "VOffGas_const": 0.1}
                   For 'continuous', provide {"Vs_const": D*V_l, "Vf_const": same} or pass your own schedules.
    """
    mode_consts = mode_consts or {}
    mode = make_mode(mode_name, mode_consts,params)

    keysprm = ["V_l","V_g", "V_gas","FractionO2","FractionCO2","Pr","H_O2","H_CO2","R","T_Scale","T_Celsius","T_Kelvin","O2_Gasin","CO2_Gasin","CSub_f" ,"CCa_f"  ,"CCl_f"  ,"CCo_f"  ,"CCu_f"  ,"CFe_f"  ,"CMg_f"  ,"CMo_f"  ,"CNa_f"  ,"CNa_b"  ,"CZn_f"  ,"CK_f"   ,"CNi_f"  ,"CNH4_f" ,"CP_f"   ,"CS_f"   ,"CH_f"   ,"CH_a"   ,"COH_f"  ,"COH_b","Y_Sub" ,"Y_Ca"  ,"Y_Cl"  ,"Y_Co"  ,"Y_Cu"  ,"Y_Fe"  ,"Y_Mg"  ,"Y_Mo"  ,"Y_Na"  ,"Y_Zn"  ,"Y_K"   ,"Y_Ni"  ,"Y_NH4" ,"Y_P"   ,"Y_S"   ,"Y_CO2" ,"Y_O2"  ,"Y_H","mu_max", "K_subs" ,"K_L_a"]

    # Sanity checks for required keys (minimal)
    for required in keysprm:#["V_l","V_g","mu_max","K_subs","K_L_a","FractionO2","FractionCO2","Pr","H_O2","H_CO2","R","T_Kelvin","O2_Gasin","CO2_Gasin","CSub_f","Y_Sub","Y_CO2","Y_O2","Y_H","Y_S"]:
        if required not in params:
            raise KeyError(f"Missing parameter: {required}")

    y0 = pack_y0(initials)
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params, mode),
        t_span=t_span, y0=y0, t_eval=t_eval,
        method=method, rtol=rtol, atol=atol
    )
    sol.state_names = STATE_ORDER  # attach for convenience
    return sol

# %%
# ---------------------------------------------------------
# [H] Save utilities (solution + metadata) in a neat format
# ---------------------------------------------------------
def save_solution(sol, params: dict, initials: dict, out_dir: str, basename: str = "run"):
    """
    Save solver output in two complementary formats:
      - NPZ (lossless, fast to reload): {t, Y, state_names}
      - CSV (human-friendly): wide table with columns: time + each state
    Also writes params/initials JSON next to them for reproducibility.

    Returns dict of written paths.
    """
    out = {}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Core arrays
    t = sol.t
    Y = sol.y  # shape (n_states, n_times)
    state_names = getattr(sol, "state_names", [f"x{i}" for i in range(Y.shape[0])])

    # NPZ
    npz_path = out_path / f"{basename}.npz"
    np.savez_compressed(npz_path, t=t, Y=Y, state_names=np.array(state_names, dtype=object))
    out["npz"] = str(npz_path)

    # CSV
    csv_path = out_path / f"{basename}.csv"
    header = ["time"] + list(state_names)
    data = np.column_stack([t, Y.T])  # shape (n_times, 1+n_states)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="", fmt="%.10g")
    out["csv"] = str(csv_path)

    # Params & initials (JSON)
    params_path = out_path / f"{basename}.params.json"
    with params_path.open("w") as f:
        json.dump(params, f, indent=2)
    out["params_json"] = str(params_path)

    initials_path = out_path / f"{basename}.initials.json"
    with initials_path.open("w") as f:
        json.dump(initials, f, indent=2)
    out["initials_json"] = str(initials_path)

    return out

# %%

# -----------------------------------------------------------------------
# [I] Plot utilities: choose any subset of states and save/display a plot
# -----------------------------------------------------------------------
def plot_timeseries(
    sol,
    variables,                 # list[str] or a single str
    out_dir: str = None,       # optional folder to save
    basename: str = "plot",    # filename stem (without extension)
    show: bool = True,         # display the figure
    save: bool = True,         # save the figure as PNG
    dpi: int = 160
):
    """
    Plot chosen variables from the solution.
      - variables: name or list of names from sol.state_names
      - If save=True and out_dir provided, saves PNG (and PDF if you want).
    Returns matplotlib Figure object.
    """
    if isinstance(variables, str):
        variables = [variables]

    state_names = getattr(sol, "state_names", None)
    if state_names is None:
        raise ValueError("Solution has no state_names attribute.")

    # Validate variables
    missing = [v for v in variables if v not in state_names]
    if missing:
        raise KeyError(f"Unknown variable(s): {missing}. Available: {state_names}")

    # Build the plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = sol.t
    for v in variables:
        idx = state_names.index(v)
        ax.plot(t, sol.y[idx, :], label=v)

    ax.set_xlabel("Time")
    ax.set_ylabel("Amount / Concentration (units per your model)")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()

    # Save if requested
    if save and out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        png_path = out_path / f"{basename}.png"
        fig.savefig(png_path, dpi=dpi)
        # Optional: also save vector version
        # fig.savefig(out_path / f"{basename}.pdf")
        # Return saved path on the axes for convenience
        ax.set_title(f"Saved: {png_path.name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

# %%
# -----------------------------------------------------------
# [J] End-to-end example: run, save the solution, plot & save
# -----------------------------------------------------------
def run_save_and_plot(
    mode_name: str = "fed-batch",
    mode_consts: dict = None,
    out_dir: str = "./playground_folders/outputs",
    run_name: str = "example_run",
    t_span=(0.0, 600.0),
    variables_to_plot=("Biomass", "Sub"),
    show_plot=True
):
    # Use your minimal builders (from the earlier code block)
    params = minimal_parameters()
    initials = minimal_initials()

    # Example tweak: gentle feed and gas flow for fed-batch
    if mode_consts is None and mode_name.lower() == "fed-batch":
        mode_consts = {"Vf_const": 0.01, "Vs_const": 0.0, "VGasIn_const": 0.1, "VOffGas_const": 0.1}

    # 1) Run
    sol = run_model(
        params=params,
        initials=initials,
        mode_name=mode_name,
        mode_consts=mode_consts,
        t_span=t_span
    )

    # 2) Save solution (+metadata) to out_dir/run_name.*
    written = save_solution(sol, params, initials, out_dir=Path(out_dir) / run_name, basename=run_name)

    # 3) Plot chosen variables and save the figure
    plot_timeseries(
        sol,
        variables=list(variables_to_plot),
        out_dir=Path(out_dir) / run_name,
        basename=f"{run_name}__{'+'.join(variables_to_plot)}",
        show=show_plot,
        save=True
    )

    return {"solution_paths": written, "state_names": getattr(sol, "state_names", None)}


# %%
# ---------------------------------------------------------
# [K] Save the model INPUTS (not the solution) to a folder
# ---------------------------------------------------------
def assemble_inputs(
    params: dict = None,
    initials: dict = None,
    mode_name: str = "fed-batch",
    mode_consts: dict = None,
    t_span: tuple = (0.0, 600.0),
    t_eval: np.ndarray | list | None = None,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9
) -> dict:
    """
    Build a single dictionary containing everything required to run the model.
    If params/initials are None, falls back to the minimal builders.
    """
    if params is None:
        params = minimal_parameters()
    if initials is None:
        initials = minimal_initials()
    payload = {
        "params": params,
        "initials": initials,
        "mode_name": mode_name,
        "mode_consts": mode_consts or {},
        "t_span": list(t_span),
        # store t_eval as list for JSON safety; None stays None
        "t_eval": None if t_eval is None else (list(t_eval) if isinstance(t_eval, (list, np.ndarray)) else [float(t_eval)]),
        "solver": {"method": method, "rtol": rtol, "atol": atol},
        # optional extras you may want to track later:
        "meta": {"state_order": STATE_ORDER}
    }
    return payload

def save_model_inputs(inputs: dict, out_dir: str, basename: str = "inputs") -> dict:
    """
    Writes a single JSON file with params, initials, schedules, and solver config.
    Returns paths.
    """
    out = {}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # basic validation
    required_top = ["params", "initials", "mode_name", "mode_consts", "t_span", "solver"]
    missing = [k for k in required_top if k not in inputs]
    if missing:
        raise KeyError(f"Missing keys in inputs: {missing}")

    json_path = out_path / f"{basename}.json"
    with json_path.open("w") as f:
        json.dump(inputs, f, indent=2)
    out["json"] = str(json_path)

    # also save lightweight README with quick tips
    readme_path = out_path / f"{basename}.README.txt"
    with readme_path.open("w") as f:
        f.write(
            "This folder contains model INPUTS only (no numerical solution).\n"
            f"Main file: {basename}.json\n"
            "Keys: params, initials, mode_name, mode_consts, t_span, t_eval, solver, meta.\n"
        )
    out["readme"] = str(readme_path)
    return out


# %%
# ---------------------------------------------------------
# [L] Load the model INPUTS from a folder
# ---------------------------------------------------------
def load_model_inputs(path_or_dir: str, basename: str = "inputs") -> dict:
    """
    Reads the JSON produced by save_model_inputs and reconstructs the inputs dict.
    Ensures t_eval is numpy.ndarray (or None), t_span is tuple, and returns
    the dict ready to pass into run_model(...).
    """
    p = Path(path_or_dir)
    json_path = p if p.suffix.lower() == ".json" else (p / f"{basename}.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Could not find inputs JSON at: {json_path}")

    with json_path.open("r") as f:
        inputs = json.load(f)

    # Normalize types
    t_eval_raw = inputs.get("t_eval", None)
    inputs["t_eval"] = None if t_eval_raw is None else np.asarray(t_eval_raw, dtype=float)
    inputs["t_span"] = tuple(inputs.get("t_span", (0.0, 1.0)))

    # A tiny sanity check for expected sub-keys
    for bucket in ("params", "initials"):
        if bucket not in inputs or not isinstance(inputs[bucket], dict):
            raise ValueError(f"Malformed inputs: '{bucket}' is missing or not a dict.")

    # Make sure required params exist (keeps friendly errors early)
    required_params = ["V_l","V_g", "V_gas","FractionO2","FractionCO2","Pr","H_O2","H_CO2","R","T_Scale","T_Celsius","T_Kelvin","O2_Gasin","CO2_Gasin","CSub_f" ,"CCa_f"  ,"CCl_f"  ,"CCo_f"  ,"CCu_f"  ,"CFe_f"  ,"CMg_f"  ,"CMo_f"  ,"CNa_f"  ,"CNa_b"  ,"CZn_f"  ,"CK_f"   ,"CNi_f"  ,"CNH4_f" ,"CP_f"   ,"CS_f"   ,"CH_f"   ,"CH_a"   ,"COH_f"  ,"COH_b","Y_Sub" ,"Y_Ca"  ,"Y_Cl"  ,"Y_Co"  ,"Y_Cu"  ,"Y_Fe"  ,"Y_Mg"  ,"Y_Mo"  ,"Y_Na"  ,"Y_Zn"  ,"Y_K"   ,"Y_Ni"  ,"Y_NH4" ,"Y_P"   ,"Y_S"   ,"Y_CO2" ,"Y_O2"  ,"Y_H","mu_max", "K_subs" ,"K_L_a"]#["V_l","V_g","mu_max","K_subs","K_L_a","FractionO2","FractionCO2","Pr","H_O2","H_CO2","R","T_Kelvin","O2_Gasin","CO2_Gasin","CSub_f","Y_Sub","Y_CO2","Y_O2","Y_H","Y_S"]
    for rp in required_params:
        if rp not in inputs["params"]:
            raise KeyError(f"Missing parameter in loaded inputs: {rp}")

    return inputs

# %%
# ---------------------------------------------------------
# [M] Convenience: run directly from the loaded inputs dict
# ---------------------------------------------------------
def run_from_inputs(inputs: dict):
    sol = run_model(
        params=inputs["params"],
        initials=inputs["initials"],
        mode_name=inputs.get("mode_name", "fed-batch"),
        mode_consts=inputs.get("mode_consts", {}),
        t_span=inputs.get("t_span", (0.0, 600.0)),
        t_eval=inputs.get("t_eval", None),
        method=inputs.get("solver", {}).get("method", "LSODA"),
        rtol=inputs.get("solver", {}).get("rtol", 1e-6),
        atol=inputs.get("solver", {}).get("atol", 1e-9),
    )
    return sol
#----------------------------------------------------------------------------------------------------------------------------
# DATASET UTILS
## DATASET GENERATION
def generate_pputida_dataset(
    # Source of model inputs (time in minutes; units from your files)
    inputs: Union[str, dict] = "../inputs/inputs.json",
    mode_name: str = "fed-batch",
    mode_consts: Optional[dict] = None,           # if None, use whatever is in inputs
    variables: Iterable[str] = ("Biomass", "Sub"),# any subset of your state names

    # Size / timing
    n_experiments: int = 6,
    n_timepoints: int = 60,
    t_span: Optional[Tuple[float, float]] = None, # fallback to inputs if None
    t_eval: list = None,

    # Between-experiment biological variability (relative CV on parameters/initials)
    param_cv: Dict[str, float] = None,            # e.g. {"mu_max":0.08, "K_subs":0.1, "K_L_a":0.15}
    initial_cv: Dict[str, float] = None,          # e.g. {"Biomass":0.15, "Sub":0.1}

    # Measurement effects
    noise_cv: float = 0.05,                       # relative Gaussian noise (5% default)
    noise_cv_per_var: Optional[Dict[str, float]] = None,
    aberrant_prob: float = 0.02,                  # per-point probability of outlier
    aberrant_multiplier_range: Tuple[float, float] = (2.5, 6.0),  # fold-change on outliers
    missing_prob: float = 0.03,                   # per-point probability of NaN

    # Output
    data_format: str = "wide",                    # "wide" or "long"
    save_dir: Optional[Union[str, Path]] = None,  # if not None, writes CSV + meta JSON
    basename: str = "synthetic_pputida",
    seed: Optional[int] = 42,
):
    """
    Generate a realistic P. putida fed-batch time-series dataset from your ODE model,
    with tunable noise, outliers, and missingness — while preserving model units.

    Returns
    -------
    df : pd.DataFrame
        - wide: columns = ["experiment","time", *variables]
        - long: columns = ["experiment","time","variable","value"]
    meta : dict
        Includes per-variable units and the model inputs used.
    """
    rng = np.random.default_rng(seed)

    # 1) Load inputs (your JSON format) or use provided dict
    if isinstance(inputs, str):
        inputs_dict = load_model_inputs(inputs)  # keeps t_span, solver, etc. consistent
    else:
        inputs_dict = deepcopy(inputs)

    # Lock to fed-batch unless explicitly overridden
    inputs_dict["mode_name"] = mode_name or inputs_dict.get("mode_name", "fed-batch")
    if mode_consts is not None:
        inputs_dict["mode_consts"] = mode_consts

    # 2) Time grid
    #t0, t1 = [0.0, 0.0]
    if (t_span is not None):# and (t_eval is None):
        inputs_dict["t_span"] = (float(t_span[0]), float(t_span[1]))
    t0, t1 = inputs_dict["t_span"]
    if t_eval is None:
        t_eval = np.linspace(t0, t1, int(n_timepoints))
   

    #elif (t_span is not None) and (t_eval is not None):
    #    t0, t1 = inputs_dict["t_span"]
        
        

    # 3) Default variability setups
    if param_cv is None:
        param_cv = {"mu_max": 0.08, "K_subs": 0.10, "K_L_a": 0.15}  # sensible defaults for kinetics
    if initial_cv is None:
        initial_cv = {"Biomass": 0.15, "Sub": 0.10}

    # 4) Units (from your initials dict); Biomass is "g", Sub is "mol", etc.
    units = {k: v["unit"] for k, v in inputs_dict["initials"].items()}  # uses your uploaded inputs.json
    # Sanity: make sure requested variables exist in the model
    base_state_names = inputs_dict.get("meta", {}).get("state_order", None)
    # If meta not present, we’ll accept variables and let run_model validate later.

    # 5) Prepare collectors
    wide_rows = []
    long_rows = []

    # 6) Generate experiments
    base_params = deepcopy(inputs_dict["params"])
    base_initials = deepcopy(inputs_dict["initials"])
    solver = inputs_dict.get("solver", {"method": "LSODA", "rtol": 1e-6, "atol": 1e-9})
    mode_consts_eff = inputs_dict.get("mode_consts", {})

    for exp in range(n_experiments):
        # 6a) Jitter parameters (relative Gaussian CV; floor at 0)
        params = deepcopy(base_params)
        for k, cv in (param_cv or {}).items():
            if k in params:
                v = float(params[k]["values"][0])
                params[k]["values"][0] = max(0.0, v * (1.0 + rng.normal(0.0, cv)))

        # 6b) Jitter initials
        initials = deepcopy(base_initials)
        for k, cv in (initial_cv or {}).items():
            if k in initials:
                v = float(initials[k]["values"][0])
                initials[k]["values"][0] = max(0.0, v * (1.0 + rng.normal(0.0, cv)))

        # 6c) Simulate the ODE (fed-batch), using your exact runner
        sol = run_model(
            params=params,
            initials=initials,
            mode_name=inputs_dict["mode_name"],
            mode_consts=mode_consts_eff,
            t_span=(t0, t1),
            t_eval=t_eval,
            method=solver.get("method", "LSODA"),
            rtol=solver.get("rtol", 1e-6),
            atol=solver.get("atol", 1e-9),
        )

        # 6d) Build a per-experiment table in "wide" form first
        row = {"experiment": exp, "time": sol.t}
        for var in variables:
            if var not in sol.state_names:
                raise KeyError(f"Variable '{var}' not in model states: {sol.state_names}")
            idx = sol.state_names.index(var)

            clean = sol.y[idx, :].copy()

            # Relative measurement noise (per variable or global)
            cv_var = (noise_cv_per_var or {}).get(var, noise_cv)
            noisy = clean * (1.0 + rng.normal(0.0, cv_var, size=clean.shape))
            noisy = np.clip(noisy, 0.0, None)  # non-negative physical amounts

            # Rare aberrant points (explode by a random factor)
            if aberrant_prob > 0.0:
                mask = rng.random(noisy.shape) < aberrant_prob
                if mask.any():
                    factors = rng.uniform(*aberrant_multiplier_range, size=mask.sum())
                    noisy[mask] = noisy[mask] * factors

            # Random missingness
            if missing_prob > 0.0:
                miss = rng.random(noisy.shape) < missing_prob
                noisy[miss] = np.nan

            row[var] = noisy

            # Also collect long-format rows on the fly
            long_rows.extend(
                {
                    "experiment": exp,
                    "time": float(t),
                    "variable": var,
                    "value": float(val) if np.isfinite(val) else np.nan,
                }
                for t, val in zip(sol.t, noisy)
            )

        # Convert row (arrays) to a DataFrame block for wide format
        # -> one row per timestamp for this experiment
        wide_block = pd.DataFrame({"experiment": row["experiment"], "time": row["time"]})
        for var in variables:
            wide_block[var] = row[var]
        wide_rows.append(wide_block)

    # 7) Concatenate and package
    df_wide = pd.concat(wide_rows, ignore_index=True)
    df_long = pd.DataFrame(long_rows)

    # Keep time units explicit (minutes) without changing your column names
    # Units are provided separately in meta to preserve your pipeline column names.
    meta = {
        "units": {var: units.get(var, "") for var in variables},
        "time_unit": "min",
        "mode_name": inputs_dict["mode_name"],
        "mode_consts": mode_consts_eff,
        "t_span": [t0, t1],
        "n_experiments": n_experiments,
        "n_timepoints": n_timepoints,
        "noise_cv": noise_cv,
        "noise_cv_per_var": noise_cv_per_var or {},
        "aberrant_prob": aberrant_prob,
        "aberrant_multiplier_range": aberrant_multiplier_range,
        "missing_prob": missing_prob,
        "param_cv": param_cv,
        "initial_cv": initial_cv,
        "solver": solver,
    }

    df = df_wide if data_format.lower() == "wide" else df_long

    # 8) Optional save
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / f"{basename}.{data_format.lower()}.csv"
        meta_path = save_dir / f"{basename}.meta.json"
        df.to_csv(csv_path, index=False)
        import json
        meta_to_write = {
            "columns": list(df.columns),
            "meta": meta
        }
        meta_path.write_text(json.dumps(meta_to_write, indent=2))
        # convenience: include paths in meta
        meta["saved_csv"] = str(csv_path)
        meta["saved_meta_json"] = str(meta_path)

    return df, meta
#----------------------------------------------------------------------------------------------------------------------------
# DATASET VISUALIZATION
def load_synthetic_dataset(csv_path, meta_path=None):
    """
    Load dataset and metadata generated by generate_pputida_dataset.
    Returns (df, meta).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if meta_path is None:
        meta_path = csv_path.with_suffix("").with_suffix(".meta.json")
    meta_path = Path(meta_path)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())["meta"]
    else:
        meta = {}
    return df, meta


def plot_spaghetti(df, meta, variables=None, ax=None, alpha=0.5):
    """
    Spaghetti plot: all experiments as faint lines.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if variables is None:
        variables = [c for c in df.columns if c not in ("experiment", "time")]

    for var in variables:
        unit = meta.get("units", {}).get(var, "")
        for exp, sub in df.groupby("experiment"):
            ax.plot(sub["time"], sub[var], alpha=alpha, label=f"{var}" if exp == 0 else "")
        ax.set_xlabel(f"Time [{meta.get('time_unit','')}]")
        ax.set_ylabel(f"{var} [{unit}]")
        ax.set_title(f"Spaghetti plot: {var}")
        ax.legend()
        plt.show()
    return ax


def plot_mean_std(df, meta, variables=None, ax=None, ci="sd"):
    """
    Plot mean ± std (default) or confidence interval per variable.
    Uses seaborn lineplot with grouping.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if variables is None:
        variables = [c for c in df.columns if c not in ("experiment", "time")]

    df_long = df.melt(id_vars=["experiment", "time"], value_vars=variables,
                      var_name="variable", value_name="value")

    sns.lineplot(data=df_long, x="time", y="value", hue="variable",
                 estimator="mean", errorbar=ci, ax=ax)

    ax.set_xlabel(f"Time [{meta.get('time_unit','')}]")
    ax.set_ylabel("Value")
    ax.set_title(f"Mean ± {ci}")
    return ax


def plot_boxplot(df, meta, variable, time_points=None, ax=None):
    """
    Boxplot of replicate distribution at selected time points for one variable.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if time_points is None:
        # choose a few evenly spaced times
        time_points = sorted(df["time"].unique())
        if len(time_points) > 10:  
            import numpy as np
            time_points = list(np.linspace(min(time_points), max(time_points), 8).astype(int))

    subset = df[df["time"].isin(time_points)]
    unit = meta.get("units", {}).get(variable, "")

    sns.boxplot(data=subset, x="time", y=variable, ax=ax)
    ax.set_xlabel(f"Time [{meta.get('time_unit','')}]")
    ax.set_ylabel(f"{variable} [{unit}]")
    ax.set_title(f"Boxplot of {variable}")
    return ax
#----------------------------------------------------------------------------------------------------------------------------
# DATASET INTERPOLATION
## Even interpolation
def interpolate_list(values, kind="linear", num_points=100):
    """
    Interpolates a list of floats sampled at evenly spaced positions.
    
    Parameters
    ----------
    values : list[float]
        List of float values (y values).
    kind : str, optional
        Interpolation type: 'linear', 'quadratic', 'cubic', etc. (scipy interp1d kinds).
    num_points : int, optional
        Number of interpolated points to return.
    
    Returns
    -------
    x_new : np.ndarray
        New x positions (normalized between 0 and len(values)-1).
    y_new : np.ndarray
        Interpolated y values.
    """
    x = np.arange(len(values))
    f = interp1d(x, values, kind=kind)
    
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_new = f(x_new)
    return x_new, y_new
# Uneven interpolations
def interpolate_points(x, y, x_new, kind="linear"):
    """
    Interpolates values at arbitrary new x positions.
    
    Parameters
    ----------
    x : list[float]
        Known x positions.
    y : list[float]
        Known y values.
    x_new : list[float] or np.ndarray
        Positions where you want interpolated values.
    kind : str, optional
        Interpolation type: 'linear', 'quadratic', 'cubic', etc.
    
    Returns
    -------
    y_new : np.ndarray
        Interpolated values at x_new.
    """
    f = interp1d(x, y, kind=kind, fill_value="extrapolate")
    return f(x_new)
#----------------------------------------------------------------------------------------------------------------------------
# PARAMETERS INFERENCE
# ---------- 0) Small helpers ----------
def _get_param(values_dict: Dict, name: str) -> float:
    return float(values_dict[name]["values"][0])

def _set_param(values_dict: Dict, name: str, value: float):
    values_dict[name]["values"][0] = float(value)

def _interp_series(t_src: np.ndarray, y_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Safe linear interpolation over model outputs onto observation times."""
    f = interp1d(t_src, y_src, kind="linear", fill_value="extrapolate", assume_sorted=True)
    return f(t_query)

# ---------- 1) Problem assembly ----------
def build_problem(
    csv_path: str | Path,
    inputs: dict = None, 
    observables: Sequence[str] = ("Biomass",),            # flexible: any observable(s) present in the CSV
    param_names: Sequence[str] = ("mu_max","K_subs","K_L_a"),  # flexible: any parameter names from inputs["params"]
    initial_names: Sequence[str] = (),                    # optional: e.g., ("Biomass","Sub")
    use_all_experiments: bool = True,                     # fit across all experiments by stacking residuals
    experiment_ids: Optional[Sequence[int]] = None,       # or fit a subset, e.g., [0,1]
) -> dict:
    """
    Loads data and inputs, and returns a dict bundling everything needed by the optimizer.
    """
    df, meta = load_synthetic_dataset(csv_path)  # wide format expected: ["experiment","time", *variables]

    # pick experiments
    if experiment_ids is None and use_all_experiments:
        experiment_ids = sorted(df["experiment"].unique().tolist())
    elif experiment_ids is None:
        experiment_ids = [int(df["experiment"].min())]

    # per-experiment time grids + observed vectors for requested observables
    obs = []  # list of dicts per experiment: {"exp":int, "time":array, "data":{var: array}}
    for exp in experiment_ids:
        sub = df[df["experiment"] == exp].sort_values("time")
        rec = {"exp": exp, "time": sub["time"].to_numpy(), "data": {}}
        for var in observables:
            if var not in sub.columns:
                raise KeyError(f"Observable '{var}' not found in dataset columns.")
            rec["data"][var] = sub[var].to_numpy()
        obs.append(rec)

    problem = {
        "inputs_base": inputs,      # untouched baseline inputs (we will deep-copy during evaluations)
        "observables": tuple(observables),
        "param_names": tuple(param_names),
        "initial_names": tuple(initial_names),
        "experiments": obs,         # list of {exp, time, data[var]}
        "meta": meta,
    }
    return problem

# ---------- 2) Packing / unpacking the decision vector ----------
def pack_theta(inputs: dict, param_names: Sequence[str], initial_names: Sequence[str]) -> np.ndarray:
    """Collect current parameter and (optional) initial values into a 1D vector."""
    theta = []
    for p in param_names:
        theta.append(_get_param(inputs["params"], p))
    for s in initial_names:
        theta.append(_get_param(inputs["initials"], s))
    return np.asarray(theta, dtype=float)

def apply_theta(inputs: dict, theta: np.ndarray, param_names: Sequence[str], initial_names: Sequence[str]) -> dict:
    """Write values from theta back into a fresh copy of inputs."""
    out = deepcopy(inputs)
    k = 0
    for p in param_names:
        _set_param(out["params"], p, float(theta[k])); k += 1
    for s in initial_names:
        _set_param(out["initials"], s, float(theta[k])); k += 1
    return out

def default_bounds(problem: dict, span: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very simple bounds: [value/span, value*span] around current seeds from inputs.json.
    Override with your own dict if needed.
    """
    base = problem["inputs_base"]
    pnames = problem["param_names"]
    inames = problem["initial_names"]
    seed = pack_theta(base, pnames, inames)
    lo = seed / span
    hi = seed * span
    # strictly positive parameters/initials
    lo = np.maximum(lo, 1e-12)
    return lo, hi

# ---------- 3) Simulation-to-data alignment ----------
def simulate_observables(inputs: dict, t_eval: np.ndarray, observables: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Run the model on t_eval and return a dict {var: y(t_eval)} for requested observables.
    """
    # Respect your run_from_inputs contract: it uses inputs["solver"], ["t_span"], ["t_eval"], etc.
    run_payload = deepcopy(inputs)
    run_payload["t_eval"] = list(t_eval)  # ensure list for JSON/contract
    sol = run_from_inputs(run_payload)    # returns scipy solution + .t, .y, and .state_names (in your utils)

    # Extract series for requested observables
    state_names = list(getattr(sol, "state_names", []))  # your utils attach this on the result
    out = {}
    for var in observables:
        if var not in state_names:
            raise KeyError(f"State '{var}' is not in model state_names.")
        idx = state_names.index(var)
        out[var] = sol.y[idx, :]  # already on t_eval
    return out

# ---------- 4) Residual builder ----------
def residuals(theta: np.ndarray, problem: dict) -> np.ndarray:
    """
    Stack residuals over all experiments and observables, aligned on each experiment's time grid.
    """
    base = problem["inputs_base"]
    pnames = problem["param_names"]
    inames = problem["initial_names"]
    observables = problem["observables"]
    exps = problem["experiments"]

    # Update inputs with the candidate theta
    trial_inputs = apply_theta(base, theta, pnames, inames)

    # For each experiment, simulate on its time vector and accumulate residuals
    res = []
    for rec in exps:
        t_obs = rec["time"]
        sim = simulate_observables(trial_inputs, t_obs, observables)
        for var in observables:
            y_model = np.asarray(sim[var], dtype=float)
            y_data  = np.asarray(rec["data"][var], dtype=float)
            # Optional scaling to stabilize magnitudes (simple variance normalization)
            scale = np.nanstd(y_data) if np.nanstd(y_data) > 0 else 1.0
            res.append( (y_model - y_data) / scale )
    return np.concatenate(res).ravel()

# ---------- 5) Public fit function ----------
def fit_parameters(
    csv_path: str | Path,
    inputs: dict = None, 
    observables: Sequence[str] = ("Biomass",),
    param_names: Sequence[str] = ("mu_max","K_subs","K_L_a"),
    initial_names: Sequence[str] = (),
    experiment_ids: Optional[Sequence[int]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    x0: Optional[np.ndarray] = None,
    verbose: int = 2,
) -> dict:
    """
    Run a bounded nonlinear least-squares fit (scipy.least_squares).
    - observables can be any subset present in the CSV.
    - param_names can include any model parameters in inputs.json.
    - initial_names can be empty, a few initials, or many (shared across all experiments).
    """
    problem = build_problem(
        csv_path=csv_path,
        inputs = inputs,
        observables=observables,
        param_names=param_names,
        initial_names=initial_names,
        experiment_ids=experiment_ids,
        use_all_experiments=(experiment_ids is None),
    )

    # seeds (x0) from inputs.json
    if x0 is None:
        x0 = pack_theta(problem["inputs_base"], param_names, initial_names)

    # bounds
    if bounds is None:
        bounds = default_bounds(problem, span=10.0)

    print(bounds)
    result = least_squares(
        fun=lambda th: residuals(th, problem),
        x0=x0,
        bounds=bounds,
        method="trf",     # robust and supports bounds
        verbose=verbose,
        max_nfev=200,
    )

    # Build a neat report
    fitted_inputs = apply_theta(problem["inputs_base"], result.x, param_names, initial_names)
    report = {
        "success": bool(result.success),
        "message": result.message,
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "theta_names": list(param_names) + list(initial_names),
        "theta_init": x0.tolist(),
        "theta_opt": result.x.tolist(),
        "bounds": (bounds[0].tolist(), bounds[1].tolist()),
        "fitted_inputs": fitted_inputs,  # you can reuse with run_from_inputs for forward sims
    }
    return report

# ---------- Data summary helpers ----------
def summarize_dataset(
    df: pd.DataFrame,
    variables: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """
    Return per-variable summary tables with columns:
    time, mean, std, n, min, max
    """
    summaries: Dict[str, pd.DataFrame] = {}
    # group by time across all experiments
    for var in variables:
        if var not in df.columns:
            raise KeyError(f"Variable '{var}' not found in dataset.")
        g = (
            df[["time", var]]
            .groupby("time", as_index=False)
            .agg(mean=(var, "mean"),
                 std=(var, "std"),
                 n=(var, "count"),
                 min=(var, "min"),
                 max=(var, "max"))
            .sort_values("time")
            .reset_index(drop=True)
        )
        # fill std when single observation
        g["std"] = g["std"].fillna(0.0)
        summaries[var] = g
    return summaries


# ---------- Simulation helpers ----------
def simulate_observables_on_grid(
    inputs: dict,
    observables: Sequence[str],
    t_eval: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Simulate on t_eval and extract the requested observables by name.
    """
    payload = deepcopy(inputs)
    payload["t_eval"] = list(np.asarray(t_eval, dtype=float))
    sol = run_from_inputs(payload)
    state_names = list(getattr(sol, "state_names", []))
    out = {}
    for var in observables:
        if var not in state_names:
            raise KeyError(f"State '{var}' not found in simulation state_names.")
        i = state_names.index(var)
        out[var] = np.asarray(sol.y[i, :], dtype=float)
    return out


# ---------- Main plotting function ----------
def plot_dataset_with_traces(
    csv_path: str | Path,
    inputs_init: dict | None = None,
    inputs_fit: dict | None = None,
    inputs_path: str | Path | None = None,     # if you prefer to load inputs from disk
    inputs_basename: str = "inputs",
    observables: Sequence[str] = ("Biomass",),
    envelope: str = "minmax",                  # "minmax" or "std" (mean ± std)
    t_dense: Optional[np.ndarray] = None,      # if None, build a dense grid from meta or inputs
    figsize: Tuple[int, int] = (9, 3),
    scatter_kwargs: dict | None = None,        # e.g., dict(s=25, alpha=0.9)
    init_kwargs: dict | None = None,           # e.g., dict(lw=2, ls="--")
    fit_kwargs: dict | None = None,            # e.g., dict(lw=2)
    alpha_envelope: float = 0.18,
    legend: bool = True,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Plot, for each observable (one subplot per observable):
      - mean scatter of the dataset,
      - shaded envelope (min–max or mean ± std),
      - initial trace (from inputs_init),
      - fitted trace (from inputs_fit).
    Returns (fig, axes_by_var).
    """
    csv_path = Path(csv_path)
    df, meta = load_synthetic_dataset(csv_path)

    # infer time unit and units for labels
    time_unit = meta.get("time_unit", "min")
    units = meta.get("units", {})  # dict var -> unit string

    # Build summaries (mean, std, min, max by time)
    summaries = summarize_dataset(df, observables)
    # Default dense grid
    if t_dense is None:
        # Prefer meta grid if present, else build from data range
        if "t_eval" in meta and isinstance(meta["t_eval"], (list, tuple)):
            base_grid = np.asarray(meta["t_eval"], dtype=float)
        else:
            tmin = max(0.0, float(df["time"].min()))
            tmax = float(df["time"].max())
            base_grid = np.linspace(tmin, tmax, 400)
        t_dense = base_grid

    # If inputs dicts not provided, optionally load from disk
    if inputs_init is None or inputs_fit is None:
        if inputs_path is None:
            # default: look next to the CSV under ./inputs/inputs.json
            inputs_dir = csv_path.parent / "inputs"
            inputs_path = inputs_dir / f"{inputs_basename}.json"
        else:
            inputs_path = Path(inputs_path)
        loaded = load_model_inputs(inputs_path.parent, inputs_path.stem)
        if inputs_init is None:
            inputs_init = loaded
        if inputs_fit is None:
            inputs_fit = loaded

    # Simulate traces on the same dense grid
    init_traces = simulate_observables_on_grid(inputs_init, observables, t_dense) if inputs_init else {}
    fit_traces  = simulate_observables_on_grid(inputs_fit,  observables, t_dense) if inputs_fit  else {}

    # Styling defaults
    scatter_kwargs = dict(s=22, alpha=0.9) | (scatter_kwargs or {})
    init_kwargs    = dict(lw=2.0, ls="--") | (init_kwargs or {})
    fit_kwargs     = dict(lw=2.2) | (fit_kwargs or {})

    # Prepare subplots (one row per observable)
    n = len(observables)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(figsize[0], figsize[1] * n), sharex=True)
    if n == 1:
        axes = [axes]
    axes_by_var = {}

    for ax, var in zip(axes, observables):
        summ = summaries[var]
        t = summ["time"].to_numpy()
        mu = summ["mean"].to_numpy()
        sd = summ["std"].to_numpy()
        vmin = summ["min"].to_numpy()
        vmax = summ["max"].to_numpy()

        # Envelope
        if envelope == "std":
            lo, hi = mu - sd, mu + sd
            label_env = "mean ± sd"
        elif envelope == "minmax":
            lo, hi = vmin, vmax
            label_env = "min–max"
        else:
            raise ValueError("envelope must be 'minmax' or 'std'.")

        ax.fill_between(t, lo, hi, alpha=alpha_envelope, label=label_env, linewidth=0)

        # Mean scatter
        ax.scatter(t, mu, label="mean (data)", **scatter_kwargs)

        # Initial trace
        if var in init_traces:
            ax.plot(t_dense, init_traces[var], label="initial trace", **init_kwargs)

        # Fitted trace
        if var in fit_traces:
            ax.plot(t_dense, fit_traces[var], label="fitted trace", **fit_kwargs)

        # Labels & cosmetics
        yunit = f" [{units.get(var, '')}]" if units.get(var) else ""
        ax.set_ylabel(f"{var}{yunit}")
        ax.grid(True, alpha=0.25)

        axes_by_var[var] = ax

    axes[-1].set_xlabel(f"time [{time_unit}]")
    if legend:
        # one legend for all; place on last axis by default
        handles, labels = axes[-1].get_legend_handles_labels()
        if not handles:  # collect from all axes if last is empty
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                handles += h; labels += l
        axes[-1].legend(handles, labels, loc="best", frameon=True)

    fig.tight_layout()
    return fig, axes_by_var
#----------------------------------------------------------------------------------------------------------------------------
# PINN related UTILITIES (for the Bioprocess model only) 
# =========================
# Residual dict for the bioprocess model
# =========================
# Expected inputs (flat scalars in `value`):
#   Volumes/flows: V_l, V_gas, Vf, Vs, VGasIn, VOffGas, Va, Vb
#   Kinetics: mu_max, K_subs, K_L_a
#   Gas props: FractionO2, FractionCO2, Pr, H_O2, H_CO2
#   Gas in comps: O2_Gasin, CO2_Gasin
#   Feeds (mol/L): CSub_f, CCa_f, CCl_f, CCo_f, CCu_f, CFe_f, CMg_f, CMo_f, CNa_f, CNa_b,
#                  CZn_f, CK_f, CNi_f, CNH4_f, CP_f, CS_f, CH_f, CH_a, COH_f, COH_b
#   Yields: Y_Sub, Y_Ca, Y_Cl, Y_Co, Y_Cu, Y_Fe, Y_Mg, Y_Mo, Y_Na, Y_Zn, Y_K, Y_Ni,
#           Y_NH4, Y_P, Y_S, Y_CO2, Y_O2, Y_H
#
# Notes:
# - Normalization follows the pattern of file1_tmp.py:
#     ( d/dt - RHS ) / (max - min)
# - We assume constant flows (Vf, Vs, VGasIn, VOffGas, Va, Vb) passed in value.
#   If you want time-varying schedules, pass “effective” values per batch/epoch.
# - The µ(Sub) and gas saturation formulas replicate the algebra in file2_tmp.py.
#   (See d*dt functions and algebraics.) :contentReference[oaicite:2]{index=2}
def _as_tensor_like(x, like):
    """Return x as a tensor on like's device/dtype."""
    if torch.is_tensor(x):
        return x.to(dtype=like.dtype, device=like.device)
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def _mu(Sub, value, eps=1e-12):
    """
    Tensor-safe Monod mu(Sub) = mu_max * Sub / (K_subs + Sub), with Sub >= 0.
    Works for batched Sub (shape [T]) and for scalar/tensor params.
    """
    # ensure tensor
    Sub_t = Sub if torch.is_tensor(Sub) else torch.as_tensor(Sub)

    # clamp Sub >= 0 elementwise
    Sub_eff = torch.clamp(Sub_t, min=0.0)

    # make params tensors on the same device/dtype
    mu_max = _as_tensor_like(value["mu_max"], Sub_eff)
    K      = _as_tensor_like(value["K_subs"],  Sub_eff)

    return mu_max * Sub_eff / (K + Sub_eff + eps)

def _Csat_O2(value):
    # partial pressure * Henry, as in file2_tmp (partial = Fraction * Pr)
    return value["FractionO2"] * value["Pr"] * value["H_O2"]

def _Csat_CO2(value):
    return value["FractionCO2"] * value["Pr"] * value["H_CO2"]

def _norm(var_name, num, min_var_dict, max_var_dict):
    return num / (max_var_dict[var_name] - min_var_dict[var_name])

def _trace_res(var_name, Cfeed_key, Y_key):
    # Generic Eq. form used for most trace elements in file2_tmp.py (Eqs. 3–15):
    # dX/dt = C_f*Vf - (X/V_l)*Vs - (1/Y)*mu(Sub)*Biomass  :contentReference[oaicite:3]{index=3}
    return (lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm(var_name,
                  d_dt_var_dict[var_name]
                  - ( value[Cfeed_key] * value["Vf"]
                      - (var_dict[var_name] / value["V_l"]) * value["Vs"]
                      - (1.0 / value[Y_key]) * _mu(var_dict["Sub"], value) * var_dict["Biomass"] ),
                  min_v, max_v))

ODE_residual_dict_Bioprocess = {
    # Biomass: dX/dt = µ(Sub)*X - (X/V_l)*Vs  (file2_tmp.py dBiomass_dt) :contentReference[oaicite:4]{index=4}
    "ode_Biomass":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("Biomass",
                  d_dt_var_dict["Biomass"]
                  - ( _mu(var_dict["Sub"], value) * var_dict["Biomass"]
                      - (var_dict["Biomass"] / value["V_l"]) * value["Vs"] ),
                  min_v, max_v),

    # Substrate: dSub/dt = CSub_f*Vf - (Sub/V_l)*Vs - (1/Y_Sub)*µ(Sub)*X (dSub_dt) :contentReference[oaicite:5]{index=5}
    "ode_Sub":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("Sub",
                  d_dt_var_dict["Sub"]
                  - ( value["CSub_f"] * value["Vf"]
                      - (var_dict["Sub"] / value["V_l"]) * value["Vs"]
                      - (1.0 / value["Y_Sub"]) * _mu(var_dict["Sub"], value) * var_dict["Biomass"] ),
                  min_v, max_v),

    # Trace elements with the standard pattern
    "ode_Ca":  _trace_res("Ca",  "CCa_f",  "Y_Ca"),
    "ode_Cl":  _trace_res("Cl",  "CCl_f",  "Y_Cl"),
    "ode_Co":  _trace_res("Co",  "CCo_f",  "Y_Co"),
    "ode_Cu":  _trace_res("Cu",  "CCu_f",  "Y_Cu"),
    "ode_Fe":  _trace_res("Fe",  "CFe_f",  "Y_Fe"),
    "ode_Mg":  _trace_res("Mg",  "CMg_f",  "Y_Mg"),
    "ode_Mo":  _trace_res("Mo",  "CMo_f",  "Y_Mo"),
    "ode_Zn":  _trace_res("Zn",  "CZn_f",  "Y_Zn"),
    "ode_K":   _trace_res("K",   "CK_f",   "Y_K"),
    "ode_Ni":  _trace_res("Ni",  "CNi_f",  "Y_Ni"),
    "ode_NH4": _trace_res("NH4", "CNH4_f", "Y_NH4"),
    "ode_P":   _trace_res("P",   "CP_f",   "Y_P"),

    # Sodium has base flow (dNa_dt): + CNa_b*Vb  (Eq. 10 logic) :contentReference[oaicite:6]{index=6}
    "ode_Na":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("Na",
                  d_dt_var_dict["Na"]
                  - ( value["CNa_f"] * value["Vf"]
                      + value["CNa_b"] * value["Vb"]
                      - (var_dict["Na"] / value["V_l"]) * value["Vs"]
                      - (1.0 / value["Y_Na"]) * _mu(var_dict["Sub"], value) * var_dict["Biomass"] ),
                  min_v, max_v),

    # Sulphate uses Y_S directly (dS_dt): ... - Y_S*µ*X  (Eq. 16) :contentReference[oaicite:7]{index=7}
    "ode_S":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("S",
                  d_dt_var_dict["S"]
                  - ( value["CS_f"] * value["Vf"]
                      - (var_dict["S"] / value["V_l"]) * value["Vs"]
                      - value["Y_S"] * _mu(var_dict["Sub"], value) * var_dict["Biomass"] ),
                  min_v, max_v),

    # CO2_l: KLa*(Csat_CO2 - CO2_l)*V_l + (1/Y_CO2)*µ*X  (dCO2_l_dt) :contentReference[oaicite:8]{index=8}
    "ode_CO2_l":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("CO2_l",
                  d_dt_var_dict["CO2_l"]
                  - ( value["K_L_a"] * (_Csat_CO2(value) - var_dict["CO2_l"]) * value["V_l"]
                      + (1.0 / value["Y_CO2"]) * _mu(var_dict["Sub"], value) * var_dict["Biomass"] ),
                  min_v, max_v),

    # CO2_g: CO2_Gasin*VGasIn - KLa*(Csat-CO2_l)*V_l - (CO2_g/V_gas)*VOffGas  (dCO2_g_dt) :contentReference[oaicite:9]{index=9}
    "ode_CO2_g":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("CO2_g",
                  d_dt_var_dict["CO2_g"]
                  - ( value["CO2_Gasin"] * value["VGasIn"]
                      - value["K_L_a"] * (_Csat_CO2(value) - var_dict["CO2_l"]) * value["V_l"]
                      - (var_dict["CO2_g"] / value["V_gas"]) * value["VOffGas"] ),
                  min_v, max_v),

    # O2_l: KLa*(Csat_O2 - O2_l)*V_l - (1/Y_O2)*µ*X  (dO2_l_dt) :contentReference[oaicite:10]{index=10}
    "ode_O2_l":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("O2_l",
                  d_dt_var_dict["O2_l"]
                  - ( value["K_L_a"] * (_Csat_O2(value) - var_dict["O2_l"]) * value["V_l"]
                      - (1.0 / value["Y_O2"]) * _mu(var_dict["Sub"], value) * var_dict["Biomass"] ),
                  min_v, max_v),

    # O2_g: O2_Gasin*VGasIn - KLa*(Csat-O2_l)*V_l - (O2_g/V_gas)*VOffGas (dO2_g_dt) :contentReference[oaicite:11]{index=11}
    "ode_O2_g":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("O2_g",
                  d_dt_var_dict["O2_g"]
                  - ( value["O2_Gasin"] * value["VGasIn"]
                      - value["K_L_a"] * (_Csat_O2(value) - var_dict["O2_l"]) * value["V_l"]
                      - (var_dict["O2_g"] / value["V_gas"]) * value["VOffGas"] ),
                  min_v, max_v),

    # Acid/base species (Eqs. 21–22) :contentReference[oaicite:12]{index=12}
    # H: CH_f*Vf + CH_a*Va - (H/V_l)*Vs + (1/Y_H)*µ*X
    "ode_H":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("H",
                  d_dt_var_dict["H"]
                  - ( value["CH_f"] * value["Vf"]
                      + value["CH_a"] * value["Va"]
                      - (var_dict["H"] / value["V_l"]) * value["Vs"]
                      + (1.0 / value["Y_H"]) * _mu(var_dict["Sub"], value) * var_dict["Biomass"] ),
                  min_v, max_v),

    # OH: COH_f*Vf + COH_b*Vb - (OH/V_l)*Vs
    "ode_OH":
        lambda var_dict, d_dt_var_dict, value, min_v, max_v:
            _norm("OH",
                  d_dt_var_dict["OH"]
                  - ( value["COH_f"] * value["Vf"]
                      + value["COH_b"] * value["Vb"]
                      - (var_dict["OH"] / value["V_l"]) * value["Vs"] ),
                  min_v, max_v),
}
