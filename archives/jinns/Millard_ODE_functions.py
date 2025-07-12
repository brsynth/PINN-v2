#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:10:59 2025

@author: lucie-garance
"""

#Creating the PINN equation

from jinns.loss._DynamicLossAbstract import ODE
from jinns.parameters import Params
from typing import Dict
from jaxtyping import Array, Float
import equinox as eqx
import jax
import jax.numpy as jnp


class SBINN_metabo(ODE):

    def equation(
        self,
        t: Float[Array, "1"],
        u: Dict[str, eqx.Module],
        params: Params,
    ) -> Float[Array, "6"]:

        #import pdb; pdb.set_trace()
        GLC, dGLC_dt = jax.value_and_grad(lambda t: jnp.squeeze(u(t, params)[0]), 0)(t) #[0]
        ACE_env, dACE_env_dt = jax.value_and_grad(lambda t: jnp.squeeze(u(t, params)[1]), 0)(t)
        X, dX_dt = jax.value_and_grad(lambda t: jnp.squeeze(u(t, params)[2]), 0)(t)
        ACCOA, dACCOA_dt = jax.value_and_grad(lambda t: jnp.squeeze(u(t, params)[3]), 0)(t)
        ACP, dACP_dt = jax.value_and_grad(lambda t: jnp.squeeze(u(t, params)[4]), 0)(t)
        ACE_cell, dACE_cell_dt = jax.value_and_grad(lambda t: jnp.squeeze(u(t, params)[5]), 0)(t)

        #eq1
        r1 = lambda V, S, Km, I, Ki : V*S/((Km+S)*(1+I/Ki))

        v_glycolysis = lambda v_max_glycolysis, GLC, Km_GLC, ACE_env, Ki_ACE_glycolysis : r1(v_max_glycolysis, GLC, Km_GLC, ACE_env, Ki_ACE_glycolysis)

        fGLC = - v_glycolysis(v_max_glycolysis=params.eq_params["v_max_glycolysis"],
                            GLC=GLC,
                            Km_GLC=params.eq_params["Km_GLC"],
                            ACE_env=ACE_env,
                            Ki_ACE_glycolysis=params.eq_params["Ki_ACE_glycolysis"]) \
               *X \
               *(params.eq_params["volume"]) \
               +(params.eq_params["v_feed"]-params.eq_params["D"]*GLC)

        #eq2

        v_acetate_exchange = lambda v_max_acetate_exchange, ACE_cell, ACE_env, Keq_acetate_exchange, Km_ACE_acetate_exchange : \
           v_max_acetate_exchange*(ACE_cell-ACE_env/Keq_acetate_exchange)/Km_ACE_acetate_exchange/(1+ACE_cell/Km_ACE_acetate_exchange+ACE_env/Km_ACE_acetate_exchange)

        fACE_env = v_acetate_exchange(v_max_acetate_exchange=params.eq_params["v_max_acetate_exchange"],
                                    ACE_cell=ACE_cell,
                                    ACE_env=ACE_env,
                                    Keq_acetate_exchange=params.eq_params["Keq_acetate_exchange"],
                                    Km_ACE_acetate_exchange=params.eq_params["Km_ACE_acetate_exchange"]) \
               *X \
               *(params.eq_params["volume"]) \
               +(-params.eq_params["D"]*ACE_env)

        #eq3
        r1 = lambda V, S, Km, I, Ki : V*S/((Km+S)*(1+I/Ki))
        v_TCA_cycle = lambda v_max_TCA_cycle, ACCOA, Km_ACCOA_TCA_cycle, ACE_env, Ki_ACE_TCA_cycle : r1(v_max_TCA_cycle, ACCOA, Km_ACCOA_TCA_cycle, ACE_env, Ki_ACE_TCA_cycle)

        fX = X \
            *v_TCA_cycle(v_max_TCA_cycle=params.eq_params["v_max_TCA_cycle"],
                         ACCOA=ACCOA,
                         Km_ACCOA_TCA_cycle=params.eq_params["Km_ACCOA_TCA_cycle"],
                         ACE_env=ACE_env,
                         Ki_ACE_TCA_cycle=params.eq_params["Ki_ACE_TCA_cycle"]) \
            *params.eq_params["Y"] \
            +(-params.eq_params["D"]*X)

        #eq4
        r1 = lambda V, S, Km, I, Ki : V*S/((Km+S)*(1+I/Ki))

        v_glycolysis = lambda v_max_glycolysis, GLC, Km_GLC, ACE_env, Ki_ACE_glycolysis : r1(v_max_glycolysis, GLC, Km_GLC, ACE_env, Ki_ACE_glycolysis)

        v_TCA_cycle = lambda v_max_TCA_cycle, ACCOA, Km_ACCOA_TCA_cycle, ACE_env, Ki_ACE_TCA_cycle : r1(v_max_TCA_cycle, ACCOA, Km_ACCOA_TCA_cycle, ACE_env, Ki_ACE_TCA_cycle)

        v_Pta = lambda v_max_Pta, ACCOA, P, ACP, COA, Keq_Pta, Km_ACP_Pta, Km_P, Km_ACCOA_Pta, Ki_P, Ki_ACP, Km_COA : \
           v_max_Pta*(ACCOA*P-ACP*COA/Keq_Pta)/(Km_ACCOA_Pta*Km_P)/(1+ACCOA/Km_ACCOA_Pta+P/Ki_P+ACP/Ki_ACP+COA/Km_COA+ACCOA*P/(Km_ACCOA_Pta*Km_P)+ACP*COA/(Km_ACP_Pta*Km_COA))

        fACCOA = 1.4*v_glycolysis(v_max_glycolysis=params.eq_params["v_max_glycolysis"],
                                GLC=GLC,
                                Km_GLC=params.eq_params["Km_GLC"],
                                ACE_env=ACE_env,
                                Ki_ACE_glycolysis=params.eq_params["Ki_ACE_glycolysis"]) \
            - v_Pta(v_max_Pta=params.eq_params["v_max_Pta"],
                    ACCOA=ACCOA,
                    P=params.eq_params["P"],
                    ACP=ACP,
                    COA=params.eq_params["COA"],
                    Keq_Pta=params.eq_params["Keq_Pta"],
                    Km_ACP_Pta=params.eq_params["Km_ACP_Pta"],
                    Km_P=params.eq_params["Km_P"],
                    Km_ACCOA_Pta=params.eq_params["Km_ACCOA_Pta"],
                    Ki_P=params.eq_params["Ki_P"],
                    Ki_ACP=params.eq_params["Ki_ACP"],
                    Km_COA=params.eq_params["Km_COA"]) \
            - v_TCA_cycle(v_max_TCA_cycle=params.eq_params["v_max_TCA_cycle"],
                          ACCOA=ACCOA,
                          Km_ACCOA_TCA_cycle=params.eq_params["Km_ACCOA_TCA_cycle"],
                          ACE_env=ACE_env,
                          Ki_ACE_TCA_cycle=params.eq_params["Ki_ACE_TCA_cycle"])

        #eq5
        v_AckA = lambda v_max_AckA, ACP, ADP, ACE_cell, ATP, Keq_AckA, Km_ACP_AckA, Km_ADP, Km_ATP, Km_ACE_AckA : \
           v_max_AckA*(ACP*ADP-ACE_cell*ATP/Keq_AckA)/(Km_ACP_AckA*Km_ADP)/((1+ACP/Km_ACP_AckA+ACE_cell/Km_ACE_AckA)*(1+ADP/Km_ADP+ATP/Km_ATP))

        v_Pta = lambda v_max_Pta, ACCOA, P, ACP, COA, Keq_Pta, Km_ACP_Pta, Km_P, Km_ACCOA_Pta, Ki_P, Ki_ACP, Km_COA : \
           v_max_Pta*(ACCOA*P-ACP*COA/Keq_Pta)/(Km_ACCOA_Pta*Km_P)/(1+ACCOA/Km_ACCOA_Pta+P/Ki_P+ACP/Ki_ACP+COA/Km_COA+ACCOA*P/(Km_ACCOA_Pta*Km_P)+ACP*COA/(Km_ACP_Pta*Km_COA))

        fACP = v_Pta(v_max_Pta=params.eq_params["v_max_Pta"],
                   ACCOA=ACCOA,
                   P=params.eq_params["P"],
                   ACP=ACP,
                   COA=params.eq_params["COA"],
                   Keq_Pta=params.eq_params["Keq_Pta"],
                   Km_ACP_Pta=params.eq_params["Km_ACP_Pta"],
                   Km_P=params.eq_params["Km_P"],
                   Km_ACCOA_Pta=params.eq_params["Km_ACCOA_Pta"],
                   Ki_P=params.eq_params["Ki_P"],
                   Ki_ACP=params.eq_params["Ki_ACP"],
                   Km_COA=params.eq_params["Km_COA"]) \
            - v_AckA(v_max_AckA=params.eq_params["v_max_AckA"],
                     ACP=ACP,
                     ADP=params.eq_params["ADP"],
                     ACE_cell=ACE_cell,
                     ATP=params.eq_params["ATP"],
                     Keq_AckA=params.eq_params["Keq_AckA"],
                     Km_ACP_AckA=params.eq_params["Km_ACP_AckA"],
                     Km_ADP=params.eq_params["Km_ADP"],
                     Km_ATP=params.eq_params["Km_ATP"],
                     Km_ACE_AckA=params.eq_params["Km_ACE_AckA"])

        #eq6
        v_AckA = lambda v_max_AckA, ACP, ADP, ACE_cell, ATP, Keq_AckA, Km_ACP_AckA, Km_ADP, Km_ATP, Km_ACE_AckA : \
           v_max_AckA*(ACP*ADP-ACE_cell*ATP/Keq_AckA)/(Km_ACP_AckA*Km_ADP)/((1+ACP/Km_ACP_AckA+ACE_cell/Km_ACE_AckA)*(1+ADP/Km_ADP+ATP/Km_ATP))

        v_acetate_exchange = lambda v_max_acetate_exchange, ACE_cell, ACE_env, Keq_acetate_exchange, Km_ACE_acetate_exchange : \
           v_max_acetate_exchange*(ACE_cell-ACE_env/Keq_acetate_exchange)/Km_ACE_acetate_exchange/(1+ACE_cell/Km_ACE_acetate_exchange+ACE_env/Km_ACE_acetate_exchange)

        fACE_cell = v_AckA(v_max_AckA=params.eq_params["v_max_AckA"],
                         ACP=ACP,
                         ADP=params.eq_params["ADP"],
                         ACE_cell=ACE_cell,
                         ATP=params.eq_params["ATP"],
                         Keq_AckA=params.eq_params["Keq_AckA"],
                         Km_ACP_AckA=params.eq_params["Km_ACP_AckA"],
                         Km_ADP=params.eq_params["Km_ADP"],
                         Km_ATP=params.eq_params["Km_ATP"],
                         Km_ACE_AckA=params.eq_params["Km_ACE_AckA"]) \
                  - v_acetate_exchange(v_max_acetate_exchange=params.eq_params["v_max_acetate_exchange"],
                                       ACE_cell=ACE_cell,
                                       ACE_env=ACE_env,
                                       Keq_acetate_exchange=params.eq_params["Keq_acetate_exchange"],
                                       Km_ACE_acetate_exchange=params.eq_params["Km_ACE_acetate_exchange"])


#        return jnp.array([dGLC_dt + self.Tmax[0] * (-(fGLC)), dACE_env_dt + self.Tmax[1] * (-fACE_env), dX_dt + self.Tmax[2] * (-fX),
#                          dACCOA_dt + self.Tmax[3] * (-fACCOA), dACP_dt + self.Tmax[4] * (-fACP), dACE_cell_dt + self.Tmax[5] * (-fACE_cell) ])

        return jnp.concatenate([dGLC_dt + self.Tmax[0] * (-(fGLC)), dACE_env_dt + self.Tmax[1] * (-fACE_env), dX_dt + self.Tmax[2] * (-fX),
                                 dACCOA_dt + self.Tmax[3] * (-fACCOA), dACP_dt + self.Tmax[4] * (-fACP), dACE_cell_dt + self.Tmax[5] * (-fACE_cell) ],axis=0)
