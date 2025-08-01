#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:51:51 2025

@author: lucie-garance
"""

#Imports
import libsbml
import numpy as np
import cobra
import sympy as sp
import sys
from cobra.manipulation import delete
import pandas as pd
from collections import defaultdict

#----------------------------------------------------------------------------------------------

def load_sbml_model(filepath):
    """ Loads an SBML model
    Input
    ------------------------
    filepath : str
        path to the .xlm model
    
    Output
    ------------------------
    libsbml modifiable model
    """
    reader = libsbml.SBMLReader()
    doc = reader.readSBML(filepath)
    if doc.getNumErrors() > 0:
        raise ValueError("SBML read error")
    return doc.getModel()

#Model reduction
def reduce_model_GPT(model, medium, measure, verbose=False):
    """ Reduces an SBML model by removing reactions that:
    - Are not part of the medium or 'measure' list
    - Have zero flux under FBA
    
    Inputs:
    ------------------------------------------------------------
    model: cobra.Model object
    medium: list of exchange reaction IDs that should be kept
    measure: list of reaction IDs to preserve
    verbose: if True, print info about removed reactions

    Outputs:
    ------------------------------------------------------------
    Reduced cobra.Model object
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
    """ Function to read the csv files"""
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
    """Builds an ODE system for an SBML model
    Input:
    ------------------------------------------------
    model : an sbml model
    
    Output:
    ------------------------------------------------
    a dictionary of sympy ODE for the SBML model
    """
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

#----------------------------------------------------------------------------------------------

def SBML_ode_solving(sbml_file, asso_csv, curation=False, cur_params=None, reduce=False, red_params=None ,verbose=True):
    """Function for SBML importation to the PINN
    Input:
    -----------------------------------------------------------------
    sbml_file : str
        Referencement of the .xml file containing the sbml model
    asso_csv : str
        Referencement of the .csv file specifying the medium constraints for cobra fba
    curation : bool
        Need for adjusting the asso_csv file or not
    cur_params : str
        file to adjust the asso_csv file
    reduce : bool
        Reduction or not of the model
    red_params : list or None
        Additional parameters to add for the reduction (e.g. to keep some equations)
    verbose : bool
        Printing informations during run
    
    Output:
    -----------------------------------------------------------------
    model : sbml model
    ode_system : sympy equations
        System of ODE associated with the SBML model
    """
    
    #Importing the model
    model=load_sbml_model(sbml_file)
    #Getting the species Ids
    species_ids = [s.getId() for s in model.getListOfSpecies()]
    if verbose:
        print("Species:", species_ids)
    #Checking if the model already have kinetic laws
    reacts=[model.getReaction(rxn.getId()).getKineticLaw() for rxn in model.getListOfReactions()]
    kin_laws= [all(reacts)==None][0]
    if verbose : 
        print("Existing kinetic laws", kin_laws)
        print("Number of reactions",len([model.getReaction(rx.getId()) for rx in model.getListOfReactions()]))
        
    if curation:
        data=pd.read_csv(cur_params, sep=";")
        dat=pd.read_csv(asso_csv)
        #Data curation and addings
        data=data.drop([data.columns[0],data.columns[-1],data.columns[-2],data.columns[-3]], axis=1)
        data.columns = [col[2:] if col.startswith('R_') else col for col in data.columns]
        new_asso=dat.loc[:, dat.loc[0] == 1]
        for col in data.columns[:-1]:
            new_asso[col]=[100, 2.2,np.nan, np.nan] #Adding columns for all metabolites that are variable in the medium
        for col in dat.loc[:, dat.loc[0] == 1].columns:
            if col not in data.columns[:-1]:
                data[col]=[1 for i in range(280)] #Adding 1 columns for all necessary metabolites present in all mediums

        data.to_csv('curated_dataset.csv') 
        new_asso.to_csv('new_iML1515.csv')
    
    model = cobra.io.read_sbml_model(sbml_file)
    mediumname=asso_csv.split(".")[0]
    method="EXP"
    if curation:
        mediumname="new_iML1515"
    mediumsize=51
    H, M = read_csv(mediumname) #H:header ; M:data (fluxes)
    if 'EXP' in method : # Reading X, Y
        if mediumsize < 1:
            sys.exit('must indicate medium size with experimental dataset')
            medium = []
            for i in range(mediumsize):
                medium.append(H[i])
        else:
            medium = H[1:]  
    measure=[]  #[r.id for r in model.reactions if "BIOMASS" in r.id.upper()]
    
    reduced, fluxes = reduce_model_GPT(model, medium=medium, measure=measure, verbose=True)
    
    #Getting back an SBML model
    cobra.io.write_sbml_model(reduced, "reduced_model.xml")
    model=load_sbml_model("reduced_model.xml")
    
    #Adding biomass as a metabolite in the model
    if "biomass" not in [m.id for m in model.species]:
        biomass_species = model.createSpecies()
        biomass_species.setId("biomass")
        biomass_species.setName("Biomass")
        if model.getCompartment("c") is not None:
            biomass_species.setCompartment("c")
        else:
            print("Error : no cytosol compartment") #This is not supposed to happen with our model
        biomass_species.setInitialAmount(0.0)
        biomass_species.setBoundaryCondition(False)
        biomass_species.setHasOnlySubstanceUnits(False)
        biomass_species.setConstant(False)
    
    #Getting the biomass reactions
    biomass_reactions=[rxn.id for rxn in model.getListOfReactions() if "BIOMASS" in rxn.id.upper()]
    for id in biomass_reactions:
        biomass_rxn = model.getReaction(id)
        biomass_product = biomass_rxn.createProduct()
        biomass_product.setSpecies("biomass")
        biomass_product.setStoichiometry(1.0)
        biomass_product.setConstant(True)
    
    #Checking that the changes have occured
    if verbose:
        print("biomass" in [s.getId() for s in model.getListOfSpecies()])

    for id in biomass_reactions:
        biomass_rxn = model.getReaction(id)
        for prod in biomass_rxn.getListOfProducts():
            if verbose:
                print(f"{prod.getStoichiometry()} {prod.getSpecies()}")
    
    ode_system = build_odes_nolaws(model)
    if verbose:
        for species, ode in ode_system.items():
            print(f"d{species}/dt = {ode}")
    
    return model, ode_system

































