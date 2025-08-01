#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:44:08 2025

@author: lucie-garance
"""

#Imports
import sympy as sp

#----------------------------------------------------------------------------------------------

#Building dictionnary – Parameters
def get_all_symbols(ode_exprs):
    """ getting all the symbols of a set of ode expressions in sympy"""
    symbols = set()
    for expr in ode_exprs.values():
        symbols.update(expr.free_symbols)
    return sorted(symbols, key=lambda s: str(s))
#all_symbs=get_all_symbols(ode_system)

def split_species_and_params(expr, species_ids):
    "splitting the symbols of a sympy expression between variables and parameters"
    expr_symbols = expr.free_symbols
    species_syms = {sp.Symbol(sid) for sid in species_ids}
    
    species = sorted(expr_symbols & species_syms, key=str)
    parameters = sorted(expr_symbols - species_syms, key=str)

    return species, parameters


#Building the parameters dict
def ode_parameters(model, ode_system, verbose=True):
    """ Creating the ode paramters dict
    Input:
    --------------------------------------------------------------------------------------
    model : sbml model
    ode_system : dictionary of the ODE of the model
    
    Output:
    --------------------------------------------------------------------------------------
    ode_parameters_ranges_dict, variables_standard_dev_dict: dictionaries
        dictionaries of the ranges for parameters search and variables standard deviation
    """
    #Ranges dictionnary and Variables std
    ode_parameter_ranges_dict={}
    variables_standard_dev_dict={}
    species_ids = [s.getId() for s in model.getListOfSpecies()]
    for species, expr in ode_system.items():
        ode_vars, params = split_species_and_params(expr, species_ids)
        for p in params:
            ode_parameter_ranges_dict[p]=(1e-5,1e5)
        for v in ode_vars:
            if verbose:
                print(v)
            variables_standard_dev_dict[v]=1
    #Adding some complementary parameters
    ode_parameter_ranges_dict["volume"]=(1e-5,1e5)
    
    return ode_parameter_ranges_dict, variables_standard_dev_dict

#True values dictionnary
#ode_parameters_dict={}
#for symb in all_symbs:
#  ode_parameters_dict[symb]= value
#ode_parameters_dict

#----------------------------------------------------------------------------------------------

#Building dictionnary – Equations 

def build_lambda_dict(ode_exprs):
    """ building a dictionary of lambda functions out of ode expressions """
    lambda_dict = {}
    symbol_dict = {}
    ode_dict={}

    for species, expr in ode_exprs.items():
        symbols = sorted(expr.free_symbols, key=lambda s: str(s))
        symbol_dict[species] = symbols
        lambda_dict[species] = sp.lambdify(symbols, expr, modules='numpy')
        ode_dict[species]=expr

    return lambda_dict, symbol_dict, ode_dict

def sort_symbols_by_name(symbols, names):
    """ sorting symbols in alphabetical order from their names """
    # To sort the symbols by name 
    sorted_symbols = [sym for _, sym in sorted(zip(names, symbols), key=lambda pair: pair[0])]
    return sorted_symbols


def create_dict_lambda(expression_func, var_keys, d_dt_keys, value_keys, min_keys, max_keys):
    """ Creates a dictionary of lambda functions using dictionaries """
    # Create symbols
    all_keys = {
        "var_dict": var_keys,
        "d_dt_var_dict": d_dt_keys,
        "value": value_keys,
        "min_var_dict": min_keys,
        "max_var_dict": max_keys,
    }
    
    symbol_map = {}
    for dict_name, keys in all_keys.items():
        for key in keys:
            symbol = sp.Symbol(f"{dict_name}__{key}")
            symbol_map[(dict_name, key)] = symbol
    
    # Build the symbolic expression
    expr = expression_func(symbol_map)

    # Create the ordered list of variables to lambdify
    ordered_symbols = [symbol_map[(dict_name, key)]
                       for dict_name, keys in all_keys.items()
                       for key in keys]

    # Create a lambda function using sympy.lambdify
    f = sp.lambdify(ordered_symbols, expr, modules="numpy")

    # Define the final lambda wrapper that takes 5 dictionaries
    def final_lambda(var_dict, d_dt_var_dict, value, min_var_dict, max_var_dict):
        lookup = {
            "var_dict": var_dict,
            "d_dt_var_dict": d_dt_var_dict,
            "value": value,
            "min_var_dict": min_var_dict,
            "max_var_dict": max_var_dict
        }
        args = [lookup[dict_name][key] for dict_name, keys in all_keys.items() for key in keys]
        return f(*args)

    return final_lambda

def build_residual_dict(model, ode_system):
    """ Residuals dictionary builder
    Inputs:
    ---------------------------------------------------------------
    model : sbml model
    ode_system : dictionary of the ODE associated with the SBML model
    
    Output:
    ---------------------------------------------------------------
    The residuals dictionary
    """
    
    lambda_dict, symbol_dict, ode_d = build_lambda_dict(ode_system)
    ode_dict={}
    list_spes=list(lambda_dict.keys())
    species_ids = [s.getId() for s in model.getListOfSpecies()]

    for i in range(len(list_spes)):
      new_symbol_var=list_spes[i] 
      list_nsv=[new_symbol_var] #d_dt_keys, min_var_keys, max_var_keys
      expr_=ode_d[list_spes[i]]
      vars, params =split_species_and_params(expr_, species_ids)
      symbs_var= [str(v) for v in vars if str(v) != list_spes[i]]
      var_keys=symbs_var + list_nsv
      if 'biomass' not in var_keys:
        var_keys = var_keys + ["biomass"] #var_keys  
      symbs_params=[str(p) for p in params] 
      values_keys=symbs_params + ["volume"] #values_keys

      #Building the function for symbolic expression 
      def expr_func(symbol_map):
        get = lambda d, k: symbol_map[(d, k)]
        names_args=[sv for sv in symbs_var] + [sp for sp in symbs_params] + [new_symbol_var]
        args=[get("var_dict",sv) for sv in symbs_var] + [get("value",sp) for sp in symbs_params]+[get("var_dict",new_symbol_var)]
        args=sort_symbols_by_name(args, names_args)
        args=tuple(args)
        return get("d_dt_var_dict", new_symbol_var) - ((lambda_dict[list_spes[i]](*args))*get("var_dict","biomass")*get("value","volume"))/(get("max_var_dict",new_symbol_var)-get("min_var_dict",new_symbol_var))

      ode_dict[f"ode_{i}"] = create_dict_lambda(expr_func, var_keys, list_nsv, values_keys, list_nsv, list_nsv)

    return ode_dict