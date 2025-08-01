#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:20:39 2025

@author: lucie-garance
"""

#Getting the linear modulus of equinox to get the right shape for the bias (because it doesn't work)
import math
from typing import Any, Literal, Optional, TypeVar, Union

import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from equinox._misc import default_floating_dtype
from equinox._module import field, Module
from equinox.nn._misc import default_init, named_scope

class Linear(Module, strict=True):
    """Performs a linear transformation."""

    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.

        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        if in_features_ == 0:
            lim = 1.0
        else:
            lim = 1 / math.sqrt(in_features_)
        wshape = (out_features_, in_features_)
        self.weight = default_init(wkey, wshape, dtype, lim)
        print('shape_weight',self.weight.shape)
        bshape = (out_features_,) #1??
        self.bias = default_init(bkey, bshape, dtype, lim) if use_bias else None
        print('shape_bias',self.bias.shape)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    @named_scope("eqx.nn.Linear")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`. (Or shape `()` if
        `out_features="scalar"`.)
        """
        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        #print('shape_x_bf',x.shape)
        #print('x_bf',x)
        x = self.weight @ x
        #print('shape_x_+weight',x.shape)

        if self.bias is not None:

            if len(x.shape)==1:
                x = x + self.bias
                #print('shape_x_+bias_if',x.shape)
            else:
                x = x + self.bias.reshape(-1, 1) #reshaping the bias to make sure that the correct broadcasting rule is applied (for x.shape = (.,1))
                #print('shape_x_+bias_resh',x.shape)
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        #print('x_final',x)
        #print('shape_x_final',x.shape)
        return x


_T = TypeVar("_T")


class Identity(Module, strict=True):
    """Identity operation that does nothing. Sometimes useful as a placeholder for
    another Module.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Consumes arbitrary `*args` and `**kwargs` but ignores them."""

    @named_scope("eqx.nn.Identity")
    def __call__(self, x: _T, *, key: Optional[PRNGKeyArray] = None) -> _T:
        """**Arguments:**

        - `x`: The input, of any type.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The input, unchanged.
        """
        return x
    

class IndivTrainingProcess(object):
    
    """ Function that initialises a IndivTrainingProcess object

    Parameters
    ------------
    dyn_loss_weight :  float or jnp.array of shape (nb_of_metabolites,)
        array containing the weight of the ODE system (either one weight for all or one per ODE) in the loss function
        
    init_cond_weight : float
        weight of the initial conditions-related term of the loss function
        
    obs_weight : float
        weight of the observation-related term of the loss function
        
    n_iter : int
        maximum number of epochs of the training
        
    optimizer : optax optimiser
        the optimiser that will be used in the solve function of jinns

    Output
    ------------
    IndivTrainingProcess : IndivTrainingProcess object type
        Attributes :    dyn_loss_weight
                        init_cond_weight
                        obs_weight
                        n_iter
                        optimizer
    
    """
    
    def __init__(self,dyn_loss_weight,init_cond_weight,obs_weight,n_iter,optimizer):
        self.dyn_loss_weight=dyn_loss_weight
        self.init_cond_weight=init_cond_weight
        self.obs_weight=obs_weight
        self.n_iter=n_iter
        self.optimizer=optimizer

#Création d'une loss custom
from jinns.loss._LossODE import _LossODEAbstract
import jax
import equinox as eqx
from dataclasses import InitVar, fields
from typing import TYPE_CHECKING, Dict
import abc
import warnings
import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from jaxtyping import Float, Array, Int
from jinns.data._DataGenerators import append_obs_batch
from jinns.loss._loss_utils import (
    observations_loss_apply,
    constraints_system_loss_apply,
)
from jinns.parameters._params import (
    _get_vmap_in_axes_params,
    _update_eq_params_dict,
)
from jinns.parameters._derivative_keys import _set_derivatives, DerivativeKeysODE
from jinns.loss._loss_weights import LossWeightsODE, LossWeightsODEDict
from jinns.loss._DynamicLossAbstract import ODE
from jinns.nn._pinn import PINN


from jinns.parameters._params import Params
from jinns.data._Batchs import ODEBatch
from jinns.loss._DynamicLossAbstract import DynamicLoss

if TYPE_CHECKING:
    from jinns.utils._types import *

from typing import TYPE_CHECKING, Callable, Dict
import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from jaxtyping import Float, Array, PyTree

from jinns.loss._boundary_conditions import (
    _compute_boundary_loss,
)
from jinns.utils._utils import _subtract_with_check, get_grid
from jinns.data._DataGenerators import append_obs_batch, make_cartesian_product
from jinns.parameters._params import _get_vmap_in_axes_params
from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN
from jinns.nn._hyperpinn import HyperPINN
from jinns.data._Batchs import *
from jinns.parameters._params import Params, ParamsDict

if TYPE_CHECKING:
    from jinns.utils._types import *


def dynamic_loss_apply(
    dyn_loss: DynamicLoss,
    u: eqx.Module,
    batch: (
        Float[Array, "batch_size 1"]
        | Float[Array, "batch_size dim"]
        | Float[Array, "batch_size 1+dim"]
    ),
    params: Params | ParamsDict,
    vmap_axes: tuple[int | None, ...],
    loss_weight: float | Float[Array, "dyn_loss_dimension"],
    u_type: PINN | HyperPINN | None = None,
) -> float:
    """
    Sometimes when u is a lambda function a or dict we do not have access to
    its type here, hence the last argument
    """
    if u_type == PINN or u_type == HyperPINN or isinstance(u, (PINN, HyperPINN)):
        v_dyn_loss = vmap(
            lambda batch, params: dyn_loss(
                batch, u, params  # we must place the params at the end
            ),
            vmap_axes,
            0,
        )
        residuals = v_dyn_loss(batch, params)
        mse_dyn_loss = jnp.nanmean(jnp.nansum(loss_weight * residuals**2, axis=-1))
    elif u_type == SPINN or isinstance(u, SPINN):
        residuals = dyn_loss(batch, u, params)
        mse_dyn_loss = jnp.nanmean(jnp.nansum(loss_weight * residuals**2, axis=-1))
    else:
        raise ValueError(f"Bad type for u. Got {type(u)}, expected PINN or SPINN")

    return mse_dyn_loss

class CustomLoss(_LossODEAbstract):
    u: eqx.Module
    dynamic_loss: DynamicLoss | None
    vmap_in_axes: tuple[Int] = eqx.field(init=False, static=True)

    def __post_init__(self, params=None):
        super().__post_init__(params=params)
        self.vmap_in_axes = (0,)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, params: Params, batch: ODEBatch) -> tuple[Float[Array, "1"], dict[str, float]]:
        temporal_batch = batch.temporal_batch

        if batch.param_batch_dict is not None:
            params = _update_eq_params_dict(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        if self.dynamic_loss is not None:
            mse_dyn_loss = dynamic_loss_apply(
                self.dynamic_loss.evaluate,
                self.u,
                temporal_batch,
                _set_derivatives(params, self.derivative_keys.dyn_loss),
                self.vmap_in_axes + vmap_in_axes_params,
                self.loss_weights.dyn_loss,
            )
        else:
            mse_dyn_loss = jnp.array(0.0)

        if self.initial_condition is not None:
            vmap_in_axes = (None,) + vmap_in_axes_params
            if not jax.tree_util.tree_leaves(vmap_in_axes):
                v_u = self.u
            else:
                v_u = jax.vmap(self.u, (None,) + vmap_in_axes_params)
            t0, u0 = self.initial_condition
            t0 = jnp.array([t0])
            u0 = jnp.array(u0)
            mask_ic = ~jnp.isnan(u0)
            u_pred = v_u(
                t0,
                _set_derivatives(params, self.derivative_keys.initial_condition),
            )
            u_pred = jnp.nan_to_num(u_pred)
            u0 = jnp.nan_to_num(u0)
            mse_initial_condition = jnp.mean(
                self.loss_weights.initial_condition * jnp.sum(((u_pred - u0) * mask_ic) ** 2, axis=-1)
            ) / jnp.maximum(jnp.sum(mask_ic), 1.0)
        else:
            mse_initial_condition = jnp.array(0.0)

        if batch.obs_batch_dict is not None:
            params = _update_eq_params_dict(params, batch.obs_batch_dict["eq_params"])
            obs_values = batch.obs_batch_dict["val"]
            obs_values_masked = jnp.nan_to_num(obs_values)
            #print('ovm',obs_values_masked)
            obs_mask = ~jnp.isnan(obs_values)
            #print('om',obs_mask)
            #import pdb; pdb.set_trace()

            raw_obs_loss = observations_loss_apply(
                self.u,
                (batch.obs_batch_dict["pinn_in"],),
                _set_derivatives(params, self.derivative_keys.observations),
                self.vmap_in_axes + vmap_in_axes_params,
                obs_values_masked, #obs_values,#
                self.loss_weights.observations,
                self.obs_slice,
            )
            #print('rol',raw_obs_loss)

            # Apply the mask and normalize
            obs_mask_sum = jnp.sum(obs_mask)
            #print('oms',obs_mask_sum)
            mse_observation_loss= jax.lax.cond(obs_mask_sum > 0, lambda _: raw_obs_loss, lambda _: jnp.array(0.0),operand=None) #jnp.where(obs_mask_sum > 0, jnp.sum(obs_mask * raw_obs_loss) / obs_mask_sum, 0.0)
            #print('mol',mse_observation_loss)
#            if obs_mask_sum > 0:
#                mse_observation_loss = jnp.sum(obs_mask * raw_obs_loss) / obs_mask_sum
#            else:
#                mse_observation_loss = jnp.array(0.0)
            #mse_observation_loss=raw_obs_loss
        else:
            mse_observation_loss = jnp.array(0.0)

        total_loss = mse_dyn_loss + mse_initial_condition + mse_observation_loss
        return total_loss, {
            "dyn_loss": mse_dyn_loss,
            "initial_condition": mse_initial_condition,
            "observations": mse_observation_loss,
        }
    
    
    
    
    
    
    