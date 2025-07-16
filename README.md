# Physics-Informed Neural Networks for Data-Driven Modeling of Metabolic Dynamics

Mathematical modeling is a powerful tool for quantitatively describing metabolic networks and predicting their behavior under varying environmental conditions—an essential aspect of metabolic engineering.

This repository contains a **hybrid model** based on the **Physics-Informed Neural Network (PINN)** framework, which integrates neural networks with ordinary differential equations (ODEs) to incorporate biological constraints into machine learning.

The model draws inspiration from:
- **Raissi et al. (2018)** (PINN methodology)
- **Yazdani et al. (2020)** (recent applications)
- **Dre. Giralt’s implementation**: [original code](https://github.com/brsynth/PINN/blob/main)

It was trained on experimental data from **P. Millard’s study** on acetate overflow in *Escherichia coli*. Despite using time-series data for only three metabolites, the model successfully reconstructed the system’s temporal dynamics with trends consistent with the original parametric model. However, accurate **kinetic parameter estimation** remains a challenge and is a key area for future improvement.

This work contributes to the field of hybrid modeling by evaluating how PINNs can capture complex metabolic dynamics. The next phase involves testing the model on broader datasets within a **whole-cell modeling** framework for *E. coli*, with the ultimate goal of supporting **mechanistically informed, data-driven predictions** in systems biology.

---

## Getting Started

1. **Clone the repository**  
    [How to clone a git repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

2. **Install Conda**  
   If not already installed, download and install a Conda distribution (e.g., Miniconda or Anaconda).

3. **Install required packages**  
   Refer to the `Package_specifics.ipynb` file for module dependencies.

---

## Repository Structure

- `ode_dict/`  
  Contains ODE model definitions and parameters in dictionary format. Includes a generator and SBML import functionality.

- `data/`  
  Contains all datasets used in example notebooks.

- `lib/`  
  Includes the core PINN class implementation and associated tools. A reference notebook for this version of the model is also included.

- `workflow_notebooks/`  
  Contains performance evaluations of the model on P. Millard et al.’s datasets.

- `workflow_exec/`  
  Server-executable versions of the notebooks.

- `archives/`  
  Contains complementary code and notebooks, including:
  - Runs on new *E. coli* data using the **IML1515** model
  - Previous versions of the model (“jinns”)

- `hands-on_LV.ipynb`  
  A guided tutorial notebook for using the code on a Lotka–Volterra equations.

- `PINN_proj.yml`
 Reacreating the same conda environment using this bash script in the current folder
 ```bash
 conda env create -f PINN_proj.yml
 conda activate PINN_proj
 ```

- `Additional specifications` 

    * pytorch : 2.6.0
    * optuna : 4.3.0
    * numpy : 2.0.2
    * tqdm : 4.67.1
    * scipy : 1.15.3
    * pandas : 2.2.2
    * matplotlib : 3.10.0
    * jax : 0.5.2

## Contributors
- **Lucie-Garance Barot**: Updating implementation from initial [PINN](https://github.com/brsynth/PINN) repository.

  Email: lbarot@clipper.ens.psl.eu