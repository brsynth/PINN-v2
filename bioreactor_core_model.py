# # Lorena's Bioprocess Model

# This model aims to describe a growth system of *Pseudomonas putida* in different
# set-ups. The model not only studies the microorganism’s growth dynamics
# but also the consumption of a carbon source (substrate) and other nutrients
# (chemical elements). Additionally, it examines the dynamics of gases that may
# be used in the process, including both those in the gas phase of the system and
# those dissolved in the liquid phase.
# 
# For further information, please check the [*BIOS_Pseudomonas_model.pdf*](misc/BIOS_Pseudomonas_model.pdf) document associated with this notebook in the *misc* folder.

# %%
# Lorena Bioprocess Model — Modular ODEs with flexible fed modes
# Requirements: numpy, scipy

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path
import json
import matplotlib.pyplot as plt


# %%
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
    Vg = params["V_g"]["values"][0]
    R = params["R"]["values"][0]
    TK = params["T_Kelvin"]["values"][0]
    return (Pr * Vg) / (R * TK)

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

def make_mode(mode: str, consts: dict) -> Mode:
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
            VGasIn=lambda t: VGasIn0, VOffGas=lambda t: VOffGas0,
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
    V_g = p["V_g"]["values"][0]
    KLa = p["K_L_a"]["values"][0]
    CO2_sat = a["Csat_CO2"](p)
    VGasIn, VOffGas = mode.VGasIn(t), mode.VOffGas(t)
    CO2_Gasin = p["CO2_Gasin"]["values"][0]
    CO2_l = y[IDX["CO2_l"]]
    return (CO2_Gasin * VGasIn
            - (KLa * (CO2_sat - CO2_l) * p["V_l"]["values"][0])
            - (CO2_g / V_g) * VOffGas)

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
    V_g = p["V_g"]["values"][0]
    KLa = p["K_L_a"]["values"][0]
    O2_sat = a["Csat_O2"](p)
    VGasIn, VOffGas = mode.VGasIn(t), mode.VOffGas(t)
    O2_Gasin = p["O2_Gasin"]["values"][0]
    O2_l = y[IDX["O2_l"]]
    return (O2_Gasin * VGasIn
            - (KLa * (O2_sat - O2_l) * p["V_l"]["values"][0])
            - (O2_g / V_g) * VOffGas)

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
    a = ALGEBRAICS
    dy = np.zeros_like(y)
    dy[IDX["Sub"]]     = dSub_dt(t, y, params, a, mode)
    dy[IDX["Biomass"]] = dBiomass_dt(t, y, params, a, mode)
    dy[IDX["Ca"]]  = dCa_dt(t, y, params, a, mode)
    dy[IDX["Cl"]]  = dCl_dt(t, y, params, a, mode)
    dy[IDX["Co"]]  = dCo_dt(t, y, params, a, mode)
    dy[IDX["Cu"]]  = dCu_dt(t, y, params, a, mode)
    dy[IDX["Fe"]]  = dFe_dt(t, y, params, a, mode)
    dy[IDX["Mg"]]  = dMg_dt(t, y, params, a, mode)
    dy[IDX["Mo"]]  = dMo_dt(t, y, params, a, mode)
    dy[IDX["Na"]]  = dNa_dt(t, y, params, a, mode)
    dy[IDX["Zn"]]  = dZn_dt(t, y, params, a, mode)
    dy[IDX["K"]]   = dK_dt(t, y, params, a, mode)
    dy[IDX["Ni"]]  = dNi_dt(t, y, params, a, mode)
    dy[IDX["NH4"]] = dNH4_dt(t, y, params, a, mode)
    dy[IDX["P"]]   = dP_dt(t, y, params, a, mode)
    dy[IDX["S"]]   = dS_dt(t, y, params, a, mode)
    dy[IDX["CO2_l"]] = dCO2_l_dt(t, y, params, a, mode)
    dy[IDX["CO2_g"]] = dCO2_g_dt(t, y, params, a, mode)
    dy[IDX["O2_l"]]  = dO2_l_dt(t, y, params, a, mode)
    dy[IDX["O2_g"]]  = dO2_g_dt(t, y, params, a, mode)
    dy[IDX["H"]]     = dH_dt(t, y, params, a, mode)
    dy[IDX["OH"]]    = dOH_dt(t, y, params, a, mode)
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
    return {
        # Volumes (assumed constant unless you later model dV/dt)
        "V_l":     {"values": [1.0],     "unit": "L"},
        "V_g":     {"values": [1.0],     "unit": "L"},

        # Gas transfer
        "K_L_a":   {"values": [50.0],    "unit": "1/min"},

        # Gas composition & thermodynamics
        "FractionO2":  {"values": [0.21], "unit": "-"},
        "FractionCO2": {"values": [0.00], "unit": "-"},
        "Pr":          {"values": [101325.0], "unit": "Pa"},
        "H_O2":        {"values": [1.3e-3],   "unit": "mol/(L·Pa)"},
        "H_CO2":       {"values": [3.4e-4],   "unit": "mol/(L·Pa)"},
        "R":           {"values": [8.314],    "unit": "J/(mol·K)"},
        "T_Kelvin":    {"values": [298.15],   "unit": "K"},

        # Gas feed composition
        "O2_Gasin":  {"values": [0.21],   "unit": "mol/L_gas"},
        "CO2_Gasin": {"values": [0.00],   "unit": "mol/L_gas"},

        # Liquid feeds (media/acid/base concentrations)
        "CSub_f": {"values": [0.5], "unit": "mol/L"},
        "CCa_f":  {"values": [0.0], "unit": "mol/L"},
        "CCl_f":  {"values": [0.0], "unit": "mol/L"},
        "CCo_f":  {"values": [0.0], "unit": "mol/L"},
        "CCu_f":  {"values": [0.0], "unit": "mol/L"},
        "CFe_f":  {"values": [0.0], "unit": "mol/L"},
        "CMg_f":  {"values": [0.0], "unit": "mol/L"},
        "CMo_f":  {"values": [0.0], "unit": "mol/L"},
        "CNa_f":  {"values": [0.0], "unit": "mol/L"},
        "CNa_b":  {"values": [0.0], "unit": "mol/L"},
        "CZn_f":  {"values": [0.0], "unit": "mol/L"},
        "CK_f":   {"values": [0.0], "unit": "mol/L"},
        "CNi_f":  {"values": [0.0], "unit": "mol/L"},
        "CNH4_f": {"values": [0.0], "unit": "mol/L"},
        "CP_f":   {"values": [0.0], "unit": "mol/L"},
        "CS_f":   {"values": [0.0], "unit": "mol/L"},
        "CH_f":   {"values": [0.0], "unit": "mol/L"},
        "CH_a":   {"values": [0.0], "unit": "mol/L"},
        "COH_f":  {"values": [0.0], "unit": "mol/L"},
        "COH_b":  {"values": [0.0], "unit": "mol/L"},

        # Stoichiometries / yields
        "Y_Sub": {"values": [0.5], "unit": "g_biomass/mol_Sub"},
        "Y_Ca":  {"values": [1.0], "unit": "mol/mol"},
        "Y_Cl":  {"values": [1.0], "unit": "mol/mol"},
        "Y_Co":  {"values": [1.0], "unit": "mol/mol"},
        "Y_Cu":  {"values": [1.0], "unit": "mol/mol"},
        "Y_Fe":  {"values": [1.0], "unit": "mol/mol"},
        "Y_Mg":  {"values": [1.0], "unit": "mol/mol"},
        "Y_Mo":  {"values": [1.0], "unit": "mol/mol"},
        "Y_Na":  {"values": [1.0], "unit": "mol/mol"},
        "Y_Zn":  {"values": [1.0], "unit": "mol/mol"},
        "Y_K":   {"values": [1.0], "unit": "mol/mol"},
        "Y_Ni":  {"values": [1.0], "unit": "mol/mol"},
        "Y_NH4": {"values": [1.0], "unit": "mol/mol"},
        "Y_P":   {"values": [1.0], "unit": "mol/mol"},
        "Y_S":   {"values": [0.05], "unit": "mol/mol"},
        "Y_CO2": {"values": [0.3],  "unit": "mol/g_biomass"},
        "Y_O2":  {"values": [0.2],  "unit": "mol/g_biomass"},
        "Y_H":   {"values": [0.5],  "unit": "mol/g_biomass"},

        # Kinetics to estimate (Table 4)
        "mu_max": {"values": [0.2], "unit": "1/min"},
        "K_subs": {"values": [0.1], "unit": "mol/L"},
    }

def minimal_initials() -> Dict[str, Dict[str, List]]:
    """
    Structure: {state_name: {"values":[...], "unit":"..."}} for STATE_ORDER.
    Units match your table (mol or g). :contentReference[oaicite:2]{index=2}
    """
    inits = {}
    # Default small nonzero to avoid divide-by-zero edge cases
    defaults = {
        "Sub": 0.1, "Biomass": 0.05,
        "Ca":0.0,"Cl":0.0,"Co":0.0,"Cu":0.0,"Fe":0.0,"Mg":0.0,"Mo":0.0,
        "Na":0.0,"Zn":0.0,"K":0.0,"Ni":0.0,"NH4":0.0,"P":0.0,"S":0.0,
        "CO2_l":0.0,"CO2_g":0.0,"O2_l":0.0,"O2_g":0.0,"H":0.0,"OH":0.0
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
    mode = make_mode(mode_name, mode_consts)

    # Sanity checks for required keys (minimal)
    for required in ["V_l","V_g","mu_max","K_subs","K_L_a","FractionO2","FractionCO2","Pr","H_O2","H_CO2","R","T_Kelvin","O2_Gasin","CO2_Gasin","CSub_f","Y_Sub","Y_CO2","Y_O2","Y_H","Y_S"]:
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
    required_params = ["V_l","V_g","mu_max","K_subs","K_L_a","FractionO2","FractionCO2","Pr","H_O2","H_CO2","R","T_Kelvin","O2_Gasin","CO2_Gasin","CSub_f","Y_Sub","Y_CO2","Y_O2","Y_H","Y_S"]
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

