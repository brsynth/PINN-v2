''' ODE system specification 
        deriv_Millard : function specifying the ODE system for odeint integration
        ODE_residual_dict_Millard : dictionary of lambda functions to compute the residual for each ODE
'''

#A set of lambda functions to simplify ODE specification
r1 = lambda V, S, Km, I, Ki : V*S/((Km+S)*(1+I/Ki))

v_glycolysis = lambda v_max_glycolysis, GLC, Km_GLC, ACE_env, Ki_ACE_glycolysis : r1(v_max_glycolysis, GLC, Km_GLC, ACE_env, Ki_ACE_glycolysis)

v_TCA_cycle = lambda v_max_TCA_cycle, ACCOA, Km_ACCOA_TCA_cycle, ACE_env, Ki_ACE_TCA_cycle : r1(v_max_TCA_cycle, ACCOA, Km_ACCOA_TCA_cycle, ACE_env, Ki_ACE_TCA_cycle)

v_AckA = lambda v_max_AckA, ACP, ADP, ACE_cell, ATP, Keq_AckA, Km_ACP_AckA, Km_ADP, Km_ATP, Km_ACE_AckA : \
   v_max_AckA*(ACP*ADP-ACE_cell*ATP/Keq_AckA)/(Km_ACP_AckA*Km_ADP)/((1+ACP/Km_ACP_AckA+ACE_cell/Km_ACE_AckA)*(1+ADP/Km_ADP+ATP/Km_ATP))

v_Pta = lambda v_max_Pta, ACCOA, P, ACP, COA, Keq_Pta, Km_ACP_Pta, Km_P, Km_ACCOA_Pta, Ki_P, Ki_ACP, Km_COA : \
   v_max_Pta*(ACCOA*P-ACP*COA/Keq_Pta)/(Km_ACCOA_Pta*Km_P)/(1+ACCOA/Km_ACCOA_Pta+P/Ki_P+ACP/Ki_ACP+COA/Km_COA+ACCOA*P/(Km_ACCOA_Pta*Km_P)+ACP*COA/(Km_ACP_Pta*Km_COA))

v_acetate_exchange = lambda v_max_acetate_exchange, ACE_cell, ACE_env, Keq_acetate_exchange, Km_ACE_acetate_exchange : \
   v_max_acetate_exchange*(ACE_cell-ACE_env/Keq_acetate_exchange)/Km_ACE_acetate_exchange/(1+ACE_cell/Km_ACE_acetate_exchange+ACE_env/Km_ACE_acetate_exchange)

###############################################################################################

#Only used to solve the ODE system with odeint
def deriv_Millard(t, y, DICT): 
    
    """ Function specifying the ODE system for solving it using odeint
    Input
    --------------
    t : float
        time of integration
    y : np.array
        value of the different variables at time t
    DICT : diction
        dictionary of the values of all ODE parameters
    
    Output
    --------------
    list of the values of the different variables at time t
    """
    
    GLC, ACE_env, X, ACCOA, ACP, ACE_cell = y

    dGLCdt = - v_glycolysis(v_max_glycolysis=DICT["v_max_glycolysis"],
                            GLC=GLC,
                            Km_GLC=DICT["Km_GLC"],
                            ACE_env=ACE_env,
                            Ki_ACE_glycolysis=DICT["Ki_ACE_glycolysis"]) \
               *X \
               *(DICT["volume"]) \
               +(DICT["v_feed"]-DICT["D"]*GLC)
    
    dACE_envdt = v_acetate_exchange(v_max_acetate_exchange=DICT["v_max_acetate_exchange"],
                                    ACE_cell=ACE_cell,
                                    ACE_env=ACE_env,
                                    Keq_acetate_exchange=DICT["Keq_acetate_exchange"],
                                    Km_ACE_acetate_exchange=DICT["Km_ACE_acetate_exchange"]) \
               *X \
               *(DICT["volume"]) \
               +(-DICT["D"]*ACE_env)
    
    dXdt = X \
            *v_TCA_cycle(v_max_TCA_cycle=DICT["v_max_TCA_cycle"],
                         ACCOA=ACCOA,
                         Km_ACCOA_TCA_cycle=DICT["Km_ACCOA_TCA_cycle"],
                         ACE_env=ACE_env,
                         Ki_ACE_TCA_cycle=DICT["Ki_ACE_TCA_cycle"]) \
            *DICT["Y"] \
            +(-DICT["D"]*X)

    dACCOAdt = 1.4*v_glycolysis(v_max_glycolysis=DICT["v_max_glycolysis"],
                                GLC=GLC,
                                Km_GLC=DICT["Km_GLC"],
                                ACE_env=ACE_env,
                                Ki_ACE_glycolysis=DICT["Ki_ACE_glycolysis"]) \
            - v_Pta(v_max_Pta=DICT["v_max_Pta"],
                    ACCOA=ACCOA,
                    P=DICT["P"],
                    ACP=ACP,
                    COA=DICT["COA"],
                    Keq_Pta=DICT["Keq_Pta"],
                    Km_ACP_Pta=DICT["Km_ACP_Pta"],
                    Km_P=DICT["Km_P"],
                    Km_ACCOA_Pta=DICT["Km_ACCOA_Pta"],
                    Ki_P=DICT["Ki_P"],
                    Ki_ACP=DICT["Ki_ACP"],
                    Km_COA=DICT["Km_COA"]) \
            - v_TCA_cycle(v_max_TCA_cycle=DICT["v_max_TCA_cycle"],
                          ACCOA=ACCOA, 
                          Km_ACCOA_TCA_cycle=DICT["Km_ACCOA_TCA_cycle"],
                          ACE_env=ACE_env,
                          Ki_ACE_TCA_cycle=DICT["Ki_ACE_TCA_cycle"])
    
    dACPdt = v_Pta(v_max_Pta=DICT["v_max_Pta"],
                   ACCOA=ACCOA,
                   P=DICT["P"],
                   ACP=ACP,
                   COA=DICT["COA"],
                   Keq_Pta=DICT["Keq_Pta"],
                   Km_ACP_Pta=DICT["Km_ACP_Pta"],
                   Km_P=DICT["Km_P"], 
                   Km_ACCOA_Pta=DICT["Km_ACCOA_Pta"],
                   Ki_P=DICT["Ki_P"],
                   Ki_ACP=DICT["Ki_ACP"],
                   Km_COA=DICT["Km_COA"]) \
            - v_AckA(v_max_AckA=DICT["v_max_AckA"],
                     ACP=ACP,
                     ADP=DICT["ADP"],
                     ACE_cell=ACE_cell,
                     ATP=DICT["ATP"],
                     Keq_AckA=DICT["Keq_AckA"],
                     Km_ACP_AckA=DICT["Km_ACP_AckA"],
                     Km_ADP=DICT["Km_ADP"],
                     Km_ATP=DICT["Km_ATP"],
                     Km_ACE_AckA=DICT["Km_ACE_AckA"])
      
    dACE_celldt = v_AckA(v_max_AckA=DICT["v_max_AckA"],
                         ACP=ACP,
                         ADP=DICT["ADP"],
                         ACE_cell=ACE_cell,
                         ATP=DICT["ATP"],
                         Keq_AckA=DICT["Keq_AckA"],
                         Km_ACP_AckA=DICT["Km_ACP_AckA"], 
                         Km_ADP=DICT["Km_ADP"],
                         Km_ATP=DICT["Km_ATP"],
                         Km_ACE_AckA=DICT["Km_ACE_AckA"]) \
                  - v_acetate_exchange(v_max_acetate_exchange=DICT["v_max_acetate_exchange"],
                                       ACE_cell=ACE_cell,
                                       ACE_env=ACE_env,
                                       Keq_acetate_exchange=DICT["Keq_acetate_exchange"],
                                       Km_ACE_acetate_exchange=DICT["Km_ACE_acetate_exchange"])


    return [dGLCdt, dACE_envdt, dXdt, dACCOAdt, dACPdt, dACE_celldt]


###############################################################################################


#Dictionary for calculating the residuals for each ODE
""" Dictionary of lambda functions
-------------------------------------
Input of the lambda functions :
    var_dict : dictionary containing the values of each variable
    d_dt_var : dictionary containing the value of the automatic differentiation for each variable
    value : dictionary containing the values of each parameter
    min_var_dict (resp. max_var_dict) : dictionary containing the min (resp. max) value for each variable
--------------------------------------
Direct use by the Pinn for the determination of the residual loss at each step of optimisation
"""

ODE_residual_dict_Millard = {
                     "ode_1" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["GLC"] - (-v_glycolysis(v_max_glycolysis=value["v_max_glycolysis"],
                                                            GLC=var_dict["GLC"],
                                                            Km_GLC=value["Km_GLC"],
                                                            ACE_env=var_dict["ACE_env"],
                                                            Ki_ACE_glycolysis=value["Ki_ACE_glycolysis"])
                                             *var_dict["X"]
                                             *(value["volume"])
                                             +(value["v_feed"]-value["D"]*var_dict["GLC"])
                                             )/(max_var_dict["GLC"] - min_var_dict["GLC"]),
                     "ode_2" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["ACE_env"] - (v_acetate_exchange(v_max_acetate_exchange=value["v_max_acetate_exchange"],
                                                                       ACE_cell=var_dict["ACE_cell"],
                                                                       ACE_env=var_dict["ACE_env"],
                                                                       Keq_acetate_exchange=value["Keq_acetate_exchange"],
                                                                       Km_ACE_acetate_exchange=value["Km_ACE_acetate_exchange"])
                                                   *var_dict["X"]
                                                   *(value["volume"])
                                                   +(-value["D"]*var_dict["ACE_env"])
                                                   )/(max_var_dict["ACE_env"] - min_var_dict["ACE_env"]),
                     "ode_3" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["X"] - (var_dict["X"]
                                             *v_TCA_cycle(v_max_TCA_cycle=value["v_max_TCA_cycle"],
                                                          ACCOA=var_dict["ACCOA"],
                                                          Km_ACCOA_TCA_cycle=value["Km_ACCOA_TCA_cycle"],
                                                          ACE_env=var_dict["ACE_env"],
                                                          Ki_ACE_TCA_cycle=value["Ki_ACE_TCA_cycle"])
                                             *value["Y"]
                                             +(-value["D"]*var_dict["X"])
                                             )/(max_var_dict["X"] - min_var_dict["X"]),
                     "ode_4" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["ACCOA"] - (1.4*v_glycolysis(v_max_glycolysis=value["v_max_glycolysis"],
                                                                   GLC=var_dict["GLC"],
                                                                   Km_GLC=value["Km_GLC"],
                                                                   ACE_env=var_dict["ACE_env"],
                                                                   Ki_ACE_glycolysis=value["Ki_ACE_glycolysis"]) 
                                                   - v_Pta(v_max_Pta=value["v_max_Pta"],
                                                           ACCOA=var_dict["ACCOA"],
                                                           P=value["P"],
                                                           ACP=var_dict["ACP"],
                                                           COA=value["COA"],
                                                           Keq_Pta=value["Keq_Pta"],
                                                           Km_ACP_Pta=value["Km_ACP_Pta"],
                                                           Km_P=value["Km_P"],
                                                           Km_ACCOA_Pta=value["Km_ACCOA_Pta"],
                                                           Ki_P=value["Ki_P"],
                                                           Ki_ACP=value["Ki_ACP"],
                                                           Km_COA=value["Km_COA"]) 
                                                   - v_TCA_cycle(v_max_TCA_cycle=value["v_max_TCA_cycle"],
                                                                 ACCOA=var_dict["ACCOA"], 
                                                                 Km_ACCOA_TCA_cycle=value["Km_ACCOA_TCA_cycle"],
                                                                 ACE_env=var_dict["ACE_env"],
                                                                 Ki_ACE_TCA_cycle=value["Ki_ACE_TCA_cycle"])
                                                   )/(max_var_dict["ACCOA"] - min_var_dict["ACCOA"]),
                     "ode_5" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["ACP"] - (v_Pta(v_max_Pta=value["v_max_Pta"],
                                                      ACCOA=var_dict["ACCOA"],
                                                      P=value["P"],
                                                      ACP=var_dict["ACP"],
                                                      COA=value["COA"],
                                                      Keq_Pta=value["Keq_Pta"],
                                                      Km_ACP_Pta=value["Km_ACP_Pta"],
                                                      Km_P=value["Km_P"], 
                                                      Km_ACCOA_Pta=value["Km_ACCOA_Pta"],
                                                      Ki_P=value["Ki_P"],
                                                      Ki_ACP=value["Ki_ACP"],
                                                      Km_COA=value["Km_COA"]) 
                                                - v_AckA(v_max_AckA=value["v_max_AckA"],
                                                         ACP=var_dict["ACP"],
                                                         ADP=value["ADP"],
                                                         ACE_cell=var_dict["ACE_cell"],
                                                         ATP=value["ATP"],
                                                         Keq_AckA=value["Keq_AckA"],
                                                         Km_ACP_AckA=value["Km_ACP_AckA"],
                                                         Km_ADP=value["Km_ADP"],
                                                         Km_ATP=value["Km_ATP"],
                                                         Km_ACE_AckA=value["Km_ACE_AckA"])
                                                )/(max_var_dict["ACP"] - min_var_dict["ACP"]),
                     "ode_6" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["ACE_cell"] - (v_AckA(v_max_AckA=value["v_max_AckA"],
                                                            ACP=var_dict["ACP"],
                                                            ADP=value["ADP"],
                                                            ACE_cell=var_dict["ACE_cell"],
                                                            ATP=value["ATP"],
                                                            Keq_AckA=value["Keq_AckA"],
                                                            Km_ACP_AckA=value["Km_ACP_AckA"], 
                                                            Km_ADP=value["Km_ADP"],
                                                            Km_ATP=value["Km_ATP"],
                                                            Km_ACE_AckA=value["Km_ACE_AckA"]) 
                                                   - v_acetate_exchange(v_max_acetate_exchange=value["v_max_acetate_exchange"],
                                                                        ACE_cell=var_dict["ACE_cell"],
                                                                        ACE_env=var_dict["ACE_env"],
                                                                        Keq_acetate_exchange=value["Keq_acetate_exchange"],
                                                                        Km_ACE_acetate_exchange=value["Km_ACE_acetate_exchange"])
                                                   )/(max_var_dict["ACE_cell"] - min_var_dict["ACE_cell"]),
                    }