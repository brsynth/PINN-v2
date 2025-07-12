#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:53:45 2025

@author: lucie-garance
"""

import jinns
import numpy as np
#import pandas as pd
import jax
from jax import random, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from numpy import genfromtxt
from jinns.data import append_obs_batch
from sklearn.metrics import r2_score
import Utils as utils

class PINN_model(object):
    
    def __init__(self,KeyInitVal, DataFile, DataGeneratorParams, NNStructure, layers_params, BiologicalModelAndParameters):
        """ Function that initialises a PINN_model object

        Parameters
        ------------
        KeyInitVal :  float
            Gives the initialisation of the random key for jax
            
        DataFile : str
            path to the file of data observation.
        
        DataGeneratorParams : list [nt,batch_size,method,tmin,tmax,Tmax,n]
            The parameters necessary to programm the DataGenerator of jinns.

        NNStructure : equinox-like modulus
            The structure of the NN

        BiologicalModelAndParameters : BiologicalModel object
            Gives the ODE and the parameters of the problem

        Output
        ------------
        gp_model : PINN_model object type
            Attributes :    key, subkey
                            data_generated
                            data_generated_t
                            values
                            train_data
                            obs_data
                            u
                            init_nn_params
                            init_params
                            derivative_keys_
                            tracked_params
            
            Associated functions :  set_loss
                                    testing_loss_function
                                    train
                                    loss_plotting
                                    integration_plotting
                                    integration_results
                                    parameters_plotting
                                    parameters_results
        """

        self.key = random.PRNGKey(KeyInitVal)
        self.key, self.subkey = random.split(self.key)
        self.BiologicalModelAndParameters=BiologicalModelAndParameters
        self.layers_params=layers_params
        
        #Importing the data
        self.data_generated = genfromtxt(os.path.join(DataFile), delimiter=',')
        self.data_generated_t = self.data_generated[0]
        self.values = self.data_generated[1:]
        
        #Data generation and amplification with batches
        self.nt,self.batch_size,self.method,self.tmin,self.tmax,self.Tmax,self.n = DataGeneratorParams
        self.key, self.subkey = random.split(self.key)
        self.train_data = jinns.data.DataGeneratorODE(
            key=self.subkey, nt=self.nt, tmin=self.tmin, tmax=self.tmax, temporal_batch_size=self.batch_size, method=self.method
            )

#        self.key, self.subkey = random.split(self.key)
#        idx = jnp.append(
#            jax.random.choice(self.subkey, jnp.arange(1, n - 1), shape=(n // 5,), replace=False),
#            jnp.array([0, n - 1]),
#            )

        self.key, self.subkey = random.split(self.key)
        obs_batch_size = self.batch_size  # must be equal to time_batch_size !
        #import pdb; pdb.set_trace()
        self.obs_data = jinns.data.DataGeneratorObservations(
            obs_batch_size=obs_batch_size,
            observed_pinn_in=self.data_generated_t.reshape(-1,1)/self.Tmax,
            observed_values= self.data_generated[1:].T,
            key=self.subkey,
            )
        

        self.u, self.init_nn_params = jinns.nn.PINN_MLP.create(
            key=self.subkey,
            eqx_list=NNStructure,
            eq_type="ODE",
            )
        
        self.init_params = jinns.parameters.Params(nn_params=self.init_nn_params,eq_params=BiologicalModelAndParameters.init_eq_params)
        
        self.derivative_keys_ = jinns.parameters.DerivativeKeysODE.from_str(
            dyn_loss=jinns.parameters.Params(
                nn_params=True, 
                eq_params=BiologicalModelAndParameters.target_params
                ),
            initial_condition="nn_params", 
            observations="nn_params",
            params=self.init_params
            )
        
        self.tracked_params = jinns.parameters.Params(
            nn_params=None, 
            eq_params=BiologicalModelAndParameters.tracked_params
            )
        

    def train(self,TrainingProcess, indep_trainings=False, testing_loss_function=False, missing_data=False):
        
        """ Function to train the PINN_model object

        Parameters
        ------------
        TrainingProcess :  -
            
        indep_trainings : bool (default=False)
            if the trainings of the training process are or not independant
        
        testing_loss_function : bool (default=False)
            to print the first value of the different components of the loss before training

        Output
        ------------
        BiologicalModel : BiologicalModel object type
            Attributes :    process_memory
                            loss_weights
                            loss
                            AIC
                            params
                            total_loss_list
                            loss_by_term
                            train_data
                            lopt_state
                            stored_params
                            validation_crit_values
                            best_val_params
                            
        """
        
        self.process_memory=[]
        
        def testing_loss_function(self):
            """ Function to print the first value of the loss components """
            losses_and_grad = jax.value_and_grad(self.loss.evaluate, 0, has_aux=True)
            losses, grads = losses_and_grad(
                self.init_params,
                append_obs_batch(
                    self.train_data.get_batch()[1],
                    self.obs_data.get_batch()[1]
                    )
                )
            l_tot, d = losses
            print(f"total loss: {l_tot}") #Value of the total loss
            print(f"Individual losses: { {key: f'{val:.2f}' for key, val in d.items()} }") #Value of each component of the loss
            return None
        
        self.loss_weights = jinns.loss.LossWeightsODE(
            dyn_loss= TrainingProcess[0].dyn_loss_weight,
            initial_condition=TrainingProcess[0].init_cond_weight,
            observations=TrainingProcess[0].obs_weight,
            )
        if missing_data:
            self.loss = utils.CustomLoss(
                u=self.u,
                loss_weights=self.loss_weights,
                dynamic_loss=self.BiologicalModelAndParameters.sbinn_metabo,
                derivative_keys=self.derivative_keys_,
                initial_condition=self.BiologicalModelAndParameters.init_metabo,
                params=self.init_params
                )
        else:
            self.loss = jinns.loss.LossODE(
                u=self.u,
                loss_weights=self.loss_weights,
                dynamic_loss=self.BiologicalModelAndParameters.sbinn_metabo,
                derivative_keys=self.derivative_keys_,
                initial_condition=self.BiologicalModelAndParameters.init_metabo,
                params=self.init_params
                )
        
        if testing_loss_function==True:
            testing_loss_function()
        
        self.key, self.subkey = random.split(self.key)
        self.params, self.total_loss_list, self.loss_by_term, self.train_data, self.loss, self.opt_state, self.stored_params, self.validation_crit_values, self.best_val_params= jinns.solve(
            n_iter=TrainingProcess[0].n_iter,
            init_params=self.init_params,
            data=self.train_data,
            loss=self.loss,
            tracked_params=self.tracked_params,
            obs_data=self.obs_data,
            optimizer=TrainingProcess[0].optimizer
            )
        
        self.AIC=2*self.layers_params+2*self.total_loss_list[-1]
        
        self.process_memory.append([self.params, self.total_loss_list, self.loss_by_term, self.train_data, self.loss, self.opt_state, self.stored_params, self.validation_crit_values, self.best_val_params, self.AIC])
        
        if len(TrainingProcess)>1:
            for step in TrainingProcess[1:]:
                
                self.loss_weights = jinns.loss.LossWeightsODE(
                    dyn_loss= step.dyn_loss_weight,
                    initial_condition=step.init_cond_weight,
                    observations=step.obs_weight,
                    )
                
                if indep_trainings==True:
                    
                    if missing_data:
                        self.loss = utils.CustomLoss(
                            u=self.u,
                            loss_weights=self.loss_weights,
                            dynamic_loss=self.BiologicalModelAndParameters.sbinn_metabo,
                            derivative_keys=self.derivative_keys_,
                            initial_condition=self.BiologicalModelAndParameters.init_metabo,
                            params=self.init_params
                            )
                    else:
                        self.loss = jinns.loss.LossODE(
                            u=self.u,
                            loss_weights=self.loss_weights,
                            dynamic_loss=self.BiologicalModelAndParameters.sbinn_metabo,
                            derivative_keys=self.derivative_keys_,
                            initial_condition=self.BiologicalModelAndParameters.init_metabo,
                            params=self.init_params
                            )
                    
                    if testing_loss_function==True:
                        testing_loss_function()
                    
                    self.key, self.subkey = random.split(self.key)
                    self.params, self.total_loss_list, self.loss_by_term, self.train_data, self.loss, self.opt_state, self.stored_params, self.validation_crit_values, self.best_val_params= jinns.solve(
                        n_iter=step.n_iter,
                        init_params=self.init_params,
                        data=self.train_data,
                        loss=self.loss,
                        tracked_params=self.tracked_params,
                        obs_data=self.obs_data,
                        optimizer=step.optimizer,
                        )
                
                else:
                    if missing_data:
                        self.loss = utils.CustomLoss(
                            u=self.u,
                            loss_weights=self.loss_weights,
                            dynamic_loss=self.BiologicalModelAndParameters.sbinn_metabo,
                            derivative_keys=self.derivative_keys_,
                            initial_condition=self.BiologicalModelAndParameters.init_metabo,
                            params=self.params
                            )
                    else:
                        self.loss = jinns.loss.LossODE(
                            u=self.u,
                            loss_weights=self.loss_weights,
                            dynamic_loss=self.BiologicalModelAndParameters.sbinn_metabo,
                            derivative_keys=self.derivative_keys_,
                            initial_condition=self.BiologicalModelAndParameters.init_metabo,
                            params=self.params
                            )
                    
                    if testing_loss_function==True:
                        testing_loss_function()
                    
                    self.key, self.subkey = random.split(self.key)
                    self.params, self.total_loss_list, self.loss_by_term, self.train_data, self.loss, self.opt_state, self.stored_params, self.validation_crit_values, self.best_val_params= jinns.solve(
                        n_iter=step.n_iter,
                        init_params=self.params,
                        data=self.train_data,
                        loss=self.loss,
                        tracked_params=self.tracked_params,
                        obs_data=self.obs_data,
                        optimizer=step.optimizer,
                        opt_state=self.opt_state,
                        )
                
                self.AIC=2*self.layers_params+2*self.total_loss_list[-1]
        
                self.process_memory.append([self.params, self.total_loss_list, self.loss_by_term, self.train_data, self.loss, self.opt_state, self.stored_params, self.validation_crit_values, self.best_val_params, self.AIC]) 

        return None
    
    def loss_plotting(self,step):
        
        """ Function to plot the loss function at a given step of the training process 
        ------------------
        Inputs :    step : int
                        step of the TrainingProcess at which the extraction occurs
        ------------------
        Output :    None – displays the results
        """
        
        loss_by_term=self.process_memory[step][2]
        for loss_name, loss_values in loss_by_term.items():
            plt.plot(jnp.log10(loss_values), label=loss_name)
        plt.plot(jnp.log10(self.total_loss_list), label="total loss")
        plt.legend()
        plt.show()
        return None
    
    def integration_plotting(self,step, ValidationSet):
        
        """ Function to plot the metabolite concentration wrt time at a given step of the training process
        ------------------
        Inputs :    step : int
                        step of the TrainingProcess at which the extraction occurs
                    ValidationSet : ValidationSet object
                        gives the validation set to confront the data
        ------------------
        Output :    None – displays the results
        """
        
        params=self.process_memory[step][0]
        self.key, self.subkey = random.split(self.key, 2)
        val_data = jinns.data.DataGeneratorODE(key=self.subkey, nt=self.nt, tmin=self.tmin, tmax=self.tmax, temporal_batch_size=self.batch_size, method=self.method)

        fig, axes = plt.subplots(3, 2)
        labels = self.BiologicalModelAndParameters.list_metabo
        k=0
        for i in range(3):
            for j in range(2):
                idx = jnp.ravel_multi_index((i, j), (3, 2))
                axes[i, j].plot(ValidationSet.time_series * self.Tmax, ValidationSet.values[:, idx], label=labels[idx]+str('_true_vals'))
                u_est_ij = vmap(lambda t:self.u(t, jinns.parameters.Params(nn_params=params.nn_params, eq_params=params.eq_params)), 0, 0)
                axes[i, j].plot(val_data.times.sort(axis=0) * self.Tmax, u_est_ij(val_data.times.sort(axis=0))[:,k], '--' ,label=labels[k]+str('_predicted'))
                axes[i, j].legend()
                k+=1           
        return None
    
    def integration_results(self,step,ValidationSet):
        
        """ Function to get the R2 score of the model for the training and validation sets at a given step of the training process 
        ------------------
        Inputs :    step : int
                        step of the TrainingProcess at which the extraction occurs
                    ValidationSet : ValidationSet object
                        gives the validation set to confront the data
        ------------------
        Outputs :   R2_train : list
                        list of the R2 scores of the model for each metabolite compared to the training set
                    R2_val : list
                        list of the R2 scores of the model for each metabolite compared to the validation set
            
        """
        
        params=self.process_memory[step][0]
        u_est_ij = vmap(lambda t:self.u(t, jinns.parameters.Params(nn_params=params.nn_params, eq_params=params.eq_params)), 0, 0)
        
        R2_train,R2_val=[[],[]]
        for k in range(6):
            #import pdb; pdb.set_trace()
            if True in jnp.isnan(self.data_generated[1:][k]):
                    R2_train.append(jnp.nan)
            else:
                R2_train.append( r2_score(
                    self.data_generated[1:][k], 
                    u_est_ij(self.data_generated_t)[:,k], 
                    sample_weight=None, 
                    multioutput='uniform_average', 
                    force_finite=True
                    ) )
            R2_val.append( r2_score(
                ValidationSet.values[:,k], 
                u_est_ij(ValidationSet.time_series)[:,k],
                sample_weight=None,
                multioutput='uniform_average',
                force_finite=True
                 ))
        
        return R2_train,R2_val
        
    
    def parameters_plotting(self,step):
        
        """ Function to plot the percentage of error for each parameter of the model for the training and validation sets at a given step of the training process 
        ------------------
        Input :     step : int
                        step of the TrainingProcess at which the extraction occurs
        ------------------
        Output :    None – displays the plot
        """
        
        stored_params=self.process_memory[step][6]
        plt.figure(1)
        plt.plot(jnp.linspace(0,30,100), np.zeros((100)),'k--')
        for i in range(len(list(self.BiologicalModelAndParameters.true_vals.keys()))):
            #import pdb; pdb.set_trace()
            keys=list(self.BiologicalModelAndParameters.true_vals.keys())
            plt.plot(i, (abs(self.BiologicalModelAndParameters.true_vals[keys[i]] - stored_params.eq_params[keys[i]][-1])/self.BiologicalModelAndParameters.true_vals[keys[i]])*100,'o', label=keys[i])
        #plt.legend()
        plt.show()

    def parameters_results(self,step, print_res=False):
        
        """ Function to get the infos of estimation for each parameter of the model for the training and validation sets at a given step of the training process 
        -----------------
        Inputs :    step : int
                        step of the TrainingProcess at which the extraction occurs
                    print_res (default = False) :bool
                        if True, the results are printed
        -----------------
        Output :    pandas.DataFrame 
            ODEParams updated with the estimated value, the error percentage, the evolution during training and the variance of the evolution for each parameter
        """
        
        stored_params=self.process_memory[step][6]
        vals,err_per,evol,var=[[],[],[],[]]
        for i in list(self.BiologicalModelAndParameters.true_vals.keys()):
            vals.append(float(stored_params.eq_params[i][-1]))
            if float(self.BiologicalModelAndParameters.true_vals[i])==0:
              err_per.append(float(self.BiologicalModelAndParameters.true_vals[i] - stored_params.eq_params[i][-1]))
            else:
              err_per.append(float((abs(self.BiologicalModelAndParameters.true_vals[i] - stored_params.eq_params[i][-1])/self.BiologicalModelAndParameters.true_vals[i])*100))
            evol.append(float(stored_params.eq_params[i][-1]-stored_params.eq_params[i][0]))
            var.append(float(np.var(stored_params.eq_params[i])))
        self.BiologicalModelAndParameters.ODEParams['estimated_value_model'+str(step)]=vals
        self.BiologicalModelAndParameters.ODEParams['error_percentage_model'+str(step)]=err_per
        self.BiologicalModelAndParameters.ODEParams['evol_param_model'+str(step)]=evol
        self.BiologicalModelAndParameters.ODEParams['var_param_model'+str(step)]=var
        return self.BiologicalModelAndParameters.ODEParams
        
        
class BiologicalModel(object):
    
    def __init__(self,ODEFunction,DataFile,MetabolitesNames,ODEParams,init_time,init_metabo_conc):
        """ Function that initialises a BiologicalModel object

        Parameters
        ------------
        ODEFunction :  jinns ODE PINN object
            Gives the organised ODE and returns a list of attributes
            
        DataFile : str
            path to the file of data observation.
        
        MetabolitesNames : list of str
            list of the names of the metabolites, for referencing
            
        ODEParams : pandas.DataFrame ; columns: [names;	true_values;initialisation;	equation;targetting;tracked]
            dataframe giving the names, true values, initial values, and equation referencing for all parameters. 
            Columns targetting and tracked are to specify if the value of the parameter should be evaluated using the model, and if the value of the parameter should be tracked through training.

        init_time : jnp.array of shape (1,)
            inital time point

        init_metabo_conc : jnp.array of shape (nb of metabolites,)
            initial concentration of the metabolites

        Output
        ------------
        BiologicalModel : BiologicalModel object type
            Attributes :    sbinn_metabo
                            init_metabo
                            init_eq_params
                            target_params
                            true_vals
                            list_metabo
        
        """
        data_generated = genfromtxt(os.path.join(DataFile), delimiter=',')
        #Tmax_ = [1/(max(data_generated[1:][k])-min(data_generated[1:][k])) for k in range(6)]
        Tmax_=[1 for k in range(6)]
        self.sbinn_metabo = ODEFunction(Tmax=Tmax_)
        self.init_metabo=(init_time,init_metabo_conc)
        
        dnames=ODEParams['names'].to_dict()
        self.init_eq_params=ODEParams["initialisation"].to_dict()
        self.target_params=ODEParams["targetting"].to_dict()
        self.tracked_params=ODEParams["tracked"].to_dict()
        self.init_eq_params=dict((dnames[key], value) for (key, value) in self.init_eq_params.items())
        self.target_params=dict((dnames[key], value) for (key, value) in self.target_params.items())
        self.tracked_params=dict((dnames[key], value) for (key, value) in self.tracked_params.items())
        
        self.true_vals=ODEParams["true_values"].to_dict()
        self.true_vals=dict((dnames[key], value) for (key, value) in self.true_vals.items())
        
        self.list_metabo=MetabolitesNames
        self.ODEParams=ODEParams
        
class ValidationSet(object):
    
    """ Function that initialises a ValidationSet object

    Parameters
    ------------
    time_series :  numpy.ndarray of shape (nb_time_points, 1)
        array containing the time points at which the metabolites concentrations are given
        
    values : numpy.ndarray of shape (nb_time_points, nb_metabolites)
        array containing the values of the metabolites concentrations at each time point

    Output
    ------------
    ValidationSet : ValidationSet object type
        Attributes :    time_series
                        values
    
    """
    
    
    def __init__(self,time_series,values):
        self.time_series=time_series
        self.values=values
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        