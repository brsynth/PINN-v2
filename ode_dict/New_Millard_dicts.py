#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:08:39 2025

@author: lucie-garance
"""
## Millard_dict version with all parameters set as unknown


# Dictionary with all the parameters known, from the literature or found for Millard's model
ode_parameters_dict = {"v_max_Pta":9565521.7634551357,
                       "Km_ACCOA_Pta":0.2,
                       "Km_P":2.6,
                       "Ki_P":2.6,
                       "Ki_ACP":0.2,
                       "Km_COA":0.029,
                       "Km_ACP_Pta":0.7,
                       "P":10,
                       "COA":1.22,
                       "Keq_Pta":0.0281,
                       "v_max_AckA":336940.01763224584,
                       "ADP":0.606,
                       "Km_ACP_AckA":0.16,
                       "Km_ADP":0.5,
                       "Km_ACE_AckA":7,
                       "ATP":2.4,
                       "Km_ATP":0.07,
                       "Keq_AckA":174,
                       "v_max_TCA_cycle":740859.87349901034,
                       "Km_ACCOA_TCA_cycle":24.759179607035055,
                       "v_sink":39553.964309360796,
                       "Y":9.980425734506176e-05,
                       "volume":0.00177,
                       "Ki_ACE_TCA_cycle":2.3261654310710522,
                       "Km_ACE_acetate_exchange":33.153580532659561,
                       "v_max_acetate_exchange":480034.55363364267,
                       "v_growth_rate":3.947654032948833,
                       "r_TRP_Ki":60,
                       "v_glc_uptake":9.80047500380069,
                       "v_ace_net":22.324528765957282,
                       "D":0, ##
                       "v_feed":0, ##
                       "_X_conc_pulse":10000,##
                       "Initial for Y":9.98042573450618e-05, ##
                       "Initial for _X_conc_pulse":10000, ##
                       "Initial for _dilution_rate":0, ##
                       "Initial for _feed":0, ##
                       "Initial for volume":0.00177, ##
                       "v_max_glycolysis":5557.64,
                       "Km_GLC":0.02,
                       "Ki_ACE_glycolysis":36.6776,
                       "Keq_acetate_exchange":1
                       }

# Dictionary with the parameter ranges used in Millard's paper
ode_parameter_ranges_dict = {"v_max_Pta":(1e3,1e7),
                             "Km_ACCOA_Pta":(1e-2,1.0),
                             "Km_P":(1e-1,1e1),
                             "Ki_P":(1e-1,1e1),
                             "Ki_ACP":(1e-2,1.0),
                             "Km_COA":(1e-3,1e-1),
                             "Km_ACP_Pta":(1e-1,1e1),
                             "P":(1.0,1e2),
                             "COA":(1e-1,1e1),
                             "Keq_Pta":(1e-3,1e-1),
                             "v_max_AckA":(1e3,1e7),
                             "ADP":(1e-1,1e1),
                             "Km_ACP_AckA":(1e-2,1.0),
                             "Km_ADP":(1e-1,1e1),
                             "Km_ACE_AckA":(1.0,1e1),
                             "ATP":(1.0,1e1),
                             "Km_ATP":(1e-3,1e-1),
                             "Keq_AckA":(1e2,1e3),
                             "v_max_TCA_cycle":(1e3,1e8),
                             "Km_ACCOA_TCA_cycle":(0.1,1e3),
                             "v_sink":(1e4,1e6),
                             "Y":(1e-5,1e-3),
                             "volume":(1e-4,1e-2),
                             "Ki_ACE_TCA_cycle":(0.1,1e3),
                             "Km_ACE_acetate_exchange":(0.1,1e3),
                             "v_max_acetate_exchange":(1e3,1e8),
                             "v_growth_rate":(1e-1,1e1),
                             "r_TRP_Ki":(1.0,1e2),
                             "v_glc_uptake":(1.0,1e2),
                             "v_ace_net":(1.0,1e2),
                              "v_max_glycolysis":(1e3,1e7),
                              "Km_GLC":(1e-3,1e-1),
                              "Ki_ACE_glycolysis":(0.1,1e3),
                              "Keq_acetate_exchange":(1e-1,1e1) }

# Standard deviations used to calculate the SSR
variable_standard_deviations_dict = {"GLC":0.5,
                                     "ACE_env":0.2,
                                     "X":0.045,
                                     "ACCOA":1, # made-up standard deviation only to test on Millard_pinn_toy    
                                     "ACP":1, # made-up standard deviation only to test on Millard_pinn_toy
                                     "ACE_cell":1 # made-up standard deviation only to test on Millard_pinn_toy
                                     }