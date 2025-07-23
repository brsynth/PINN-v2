# %% [markdown]
# # Optimal Fed-Batch Control of Induced Foreign Protein Production by Recombinant Bacteria

# %%
# ODE + Algebraic model template for fed-batch bioreactor (Monod + protein induction)

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# %%

# === Algebraic functions ===
def algebraic_functions(mode: str, state: np.ndarray, params: Dict[str, float]) -> Dict[str, float]:
    """
    Computes algebraic expressions from current state and parameters.
    
    Parameters:
        state: np.ndarray, [X, S, I, P, V]
        params: dict, model parameters (must include mode and possibly F)
        
    Returns:
        dict with computed algebraic terms like mu, r_P, D, F
    """

    X, S, I, P, V = state

    # Enforce non-negative substrate
    S = max(S, 0.0)

    # Determine feed rate F(t)
    if mode == "batch":
        F = 0.0
    elif mode == "fed-batch":
        F = params.get("F", 0.0)
    elif mode == "continuous":
        F = params.get("F", 0.0)  # Must be balanced with outflow
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'batch', 'fed-batch', or 'continuous'.")

    mu = params["mu_max"] * S / (params["K_S"] + S + 1e-8)
    r_P = params["k_P"] * X * I / (params["K_I"] + I + 1e-8)
    D = F / (V + 1e-8) # reamain positive in FB and C modes, is null in B mode 

    return {
        "mu": mu,
        "r_P": r_P,
        "D": D,
        "F": F
    }


# === ODE system ===
def fed_batch_odes(mode: str, t: float, state: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Defines the ODE system for batch, fed-batch, or continuous reactor.
    
    Parameters:
        t: float, current time
        state: np.ndarray, [X, S, I, P, V]
        params: dict, model parameters (must include mode and feed settings)
    
    Returns:
        dstate_dt: np.ndarray, derivatives [dX/dt, dS/dt, dI/dt, dP/dt, dV/dt]
    """
    # correcting S(t) and dS(t) to avoid negative S (1)
    state[1] = max(state[1], 0.0)
    X, S, I, P, V = state
    # correcting S(t) and dS(t) to avoid negative S (2)
    #S = max(S, 0.0)

    alg = algebraic_functions(mode, state, params) 

    dXdt = alg["mu"] * X - alg["D"] * X
    dSdt = - (1 / params["Y_XS"]) * alg["mu"] * X + alg["D"] * (params["S_in"] - S)

    # correcting S(t) and dS(t) to avoid negative S (3)
    #dSdt = 0.0 if S + dSdt < 0.0 else dSdt 

    dIdt = alg["D"] * (params["I_in"] - I) - params["k_d"] * I
    dPdt = alg["r_P"] - alg["D"] * P
    dVdt = alg["F"] if mode != "continuous" else 0.0  # Constant volume in continuous mode

    return np.array([dXdt, dSdt, dIdt, dPdt, dVdt])


# === Load Parameters and Initial Conditions from CSV ===
def load_model_inputs(param_file: str, init_file: str) -> Tuple[Dict[str, float], np.ndarray]:
    df_params = pd.read_csv(param_file)
    df_init = pd.read_csv(init_file)

    
    params = dict(zip(df_params["Parameter"], df_params["Value"]))
    
    init_values = np.array(df_init["InitialValue"])
    # special case for mode == "batch" (we need non-null inducer at t = 0 if we want to see changes)
    init_values[list(initial_conditions.keys()).index("I0")] = params["I_in"] if mode == "batch" else init_values[list(initial_conditions.keys()).index("I0")] 

    return params, init_values
# === Run Simulation ===
def simulate_fed_batch(mode: str,params: Dict[str, float], initial_conditions: np.ndarray,
                       t_span: Tuple[float, float], t_eval = None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(
        fun=lambda t, y: fed_batch_odes(mode,t, y, params),
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        method="RK45"
    )
    return sol

# === Plot and Save Results ===
def plot_fed_batch_results_separately(mode,result, save_figs = False):
    """Plot each state variable in a separate figure and save them."""
    t = result.t
    labels = ['Biomass', 'Substrate', 'Inducer', 'Protein', 'Volume'] 
    abbrvs = ["X","S","I","P","V"]
    units  = ["g/L","g/L","mM","g/L","L"] 
    variables = result.y

    for i, (unit,label,abbrv,data) in enumerate(zip(units,labels,abbrvs,variables)):
        plt.figure(figsize=(6, 4))
        plt.plot(t, data, label=abbrv, linewidth=2)
        plt.xlabel("Time ($h$)")
        plt.ylabel(f"${unit}$")
        plt.title(label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if save_figs:
            filename = f"{output_folder}{mode}_{label.split()[0].lower()}.png"
            plt.savefig(filename)

        plt.show()

# %%
# Data processing
# Update parameters dictionary to include (min, value, max) for each parameter
params_bounds = {
    "mu_max": (0.4, 0.6, 0.8),     # 1/h
    "K_S": (0.05, 0.1, 0.2),       # g/L
    "Y_XS": (0.4, 0.5, 0.6),       # g/g
    "k_P": (0.03, 0.05, 0.07),     # 1/h
    "K_I": (0.05, 0.1, 0.2),       # mM
    "k_d": (0.005, 0.01, 0.02),    # 1/h
    "S_in": (400.0, 500.0, 600.0), # g/L
    "I_in": (5.0, 10.0, 15.0),     # mM
    "F": (0.05, 0.1, 0.2)          # L/h
}

# Convert to DataFrame
df_params_bounds = pd.DataFrame([
    {"Parameter": key, "Min": v[0], "Value": v[1], "Max": v[2]}
    for key, v in params_bounds.items()
])

input_folder  = "./playground_folders/inputs/"
output_folder = "./playground_folders/outputs/"
# Save to CSV
df_params_bounds.to_csv(f"{input_folder}model_parameters_with_bounds.csv", index=False)


# Define initial conditions dictionary
initial_conditions = {
    "X0": 0.1,  # g/L
    "S0": 5.0,  # g/L
    "I0": 0.0 ,  # mM
    "P0": 0.0,  # g/L
    "V0": 1.0   # L
}

# Save to CSV files
df_init = pd.DataFrame(list(initial_conditions.items()), columns=["Variable", "InitialValue"])

# File output
df_init.to_csv(f"{input_folder}initial_conditions.csv", index=False)


# %%
## Preruning the model
mode = "fed-batch"
params, init_conds = load_model_inputs(f"{input_folder}model_parameters_with_bounds.csv",f"{input_folder}initial_conditions.csv")

# %%
# Runing the model
my_sol = simulate_fed_batch(mode,params, init_conds, (0.0,48.0))
plot_fed_batch_results_separately(mode,my_sol,save_figs=True)


