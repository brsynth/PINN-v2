#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:37:29 2025

@author: lucie-garance
"""

#Imports
import libsbml
import numpy as np
import cobra
import cobra.manipulation as manip
from cobra import Reaction, Metabolite, Model
from cobra.flux_analysis import pfba
import sympy as sp
import sys
from cobra.manipulation import delete
import pandas as pd
from collections import defaultdict

#Loading the sbml model
def load_sbml_model(filepath):
    reader = libsbml.SBMLReader()
    doc = reader.readSBML(filepath)
    if doc.getNumErrors() > 0:
        raise ValueError("SBML read error")
    return doc.getModel()

#Model reduction
def reduce_model_GPT(model, medium, measure, verbose=False):
    """
    Reduce an SBML model by removing reactions that:
    - Are not part of the medium or 'measure' list
    - Have zero flux under FBA
    
    Parameters:
    - model: cobra.Model object
    - medium: list of exchange reaction IDs that should be kept
    - measure: list of reaction IDs to preserve
    - verbose: if True, print info about removed reactions

    Returns:
    - Reduced cobra.Model object
    """
    # Solve the model to get fluxes
    solution = model.optimize()
    
    if solution.status != 'optimal':
        raise RuntimeError("Model optimization failed. Check constraints or medium setup.")
    
    fluxes = solution.fluxes
    remove = []

    for rxn in model.reactions:
        rxn_id = rxn.id
        if np.isclose(fluxes[rxn_id], 0.0) and rxn_id not in medium and rxn_id not in measure:
            remove.append(rxn)
            if verbose:
                print(f"Removing {rxn_id} (flux = 0)")

    # Remove reactions and prune
    model.remove_reactions(remove)
    delete.prune_unused_reactions(model)

    # Remove orphan metabolites
    orphan_mets = [m for m in model.metabolites if len(m.reactions) == 0]
    model.remove_metabolites(orphan_mets)
    delete.prune_unused_metabolites(model)

    if verbose:
        print(f"Reduced model: {len(model.metabolites)} metabolites, {len(model.reactions)} reactions.")

    return model, fluxes

#CSV reader
def read_csv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    filename += '.csv'
    dataframe = pd.read_csv(filename, header=0)
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:,:])
    return HEADER, DATA
    
#Ode builder
def build_odes_nolaws(model):
    odes = defaultdict(list)

    # Create symbolic variables for all species
    species_symbols = {
        s.getId(): sp.Symbol(s.getId())
        for s in model.getListOfSpecies()
    }

    for rxn in model.getListOfReactions():
        rid = rxn.getId()
        Vmax = sp.Symbol(f"v_max_{rid}")

        reactants = rxn.getListOfReactants()
        products = rxn.getListOfProducts()

        # Symbols for Km values
        KmS = [sp.Symbol(f"Km_{r.getSpecies()}_{rid}") for r in reactants]
        KmP = [sp.Symbol(f"Km_{p.getSpecies()}_{rid}") for p in products]

        # Convenience kinetics rate equation
        reactant_terms = [
            (species_symbols[r.getSpecies()] / KmS[i]) /
            (1 + (species_symbols[r.getSpecies()] / KmS[i]))
            for i, r in enumerate(reactants)
        ]

        product_terms = [
            1 / (1 + (species_symbols[p.getSpecies()] / KmP[i]))
            for i, p in enumerate(products)
        ]

        rate = Vmax * sp.Mul(*reactant_terms) * sp.Mul(*product_terms)

        # Build the ODEs
        for reactant in reactants:
            species = reactant.getSpecies()
            stoich = -reactant.getStoichiometry()
            odes[species].append(stoich * rate)

        for product in products:
            species = product.getSpecies()
            stoich = product.getStoichiometry()
            odes[species].append(stoich * rate)

    # Combine terms for each species
    return {s: sum(terms) for s, terms in odes.items()}


