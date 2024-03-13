from typing import List, Dict, Callable, Any, Tuple, Union

import tensorflow as tf
import sys, os, logging, time
import tensorflow_probability as tfp

import numpy as np
from pinnstf2.utils import fwd_gradient, gradient
from pinnstf2.utils import (
    fix_extra_variables,
    mse,
    relative_l2_error,
    sse
)

class PINNModule:
    loss_data_before = 0.0
    loss_pde_before = 0.0
    loss_data_difference = 1.0
    loss_pde_difference = 1.0
    def __init__(
        self,
        net,
        pde_fn: Callable[[Any, ...], tf.Tensor],
        pde_fn_cp: Callable[[Any, ...], tf.Tensor],
        optimizer: tf.keras.optimizers.Nadam = tf.keras.optimizers.Nadam,
        #optimizer: tfp.optimizer.lbfgs_minimize = tfp.optimizer.lbfgs_minimize,
        loss_fn: str = "sse",
        extra_variables: Dict[str, Any] = None,
        output_fn: Callable[[Any, ...], tf.Tensor] = None,
        runge_kutta=None,
        jit_compile: bool = True,
        amp: bool = False,
        dtype: str = 'float32'
    ) -> None:
        """
        #optimizer: tfp.optimizer.lbfgs_minimize = tfp.optimizer.lbfgs_minimize,
        Initialize a `PINNModule`.

        :param net: The neural network model to be used for approximating solutions.
        :param pde_fn: The partial differential equation (PDE) function defining the PDE to solve.
        :param optimizer: The optimizer used for training the neural network.
        :param loss_fn: The name of the loss function to be used. Default is 'sse' (sum of squared errors).
        :param extra_variables: Additional variables used in the model, provided as a dictionary. Default is None.
        :param output_fn: A function applied to the output of the network, for post-processing or transformations.
        :param runge_kutta: An optional Runge-Kutta method implementation for solving discrete problems. Default is None.
        :param jit_compile: If True, TensorFlow's JIT compiler will be used for optimizing computations. Default is True.
        :param amp: Automatic mixed precision (amp) for optimizing training performance. Default is False.
        :param dtype: Data type to be used for the computations. Default is 'float32'.
        """
        super().__init__()
        
        self.net = net
        self.tf_dtype = tf.as_dtype(dtype)
        
        if hasattr(self.net, 'model'):
            self.trainable_variables = self.net.model.trainable_variables
        else:
            self.trainable_variables = self.net.trainable_variables
        (self.trainable_variables,
         self.extra_variables) = fix_extra_variables(self.trainable_variables, extra_variables, self.tf_dtype)
        self.output_fn = output_fn
        self.rk = runge_kutta
        self.pde_fn = pde_fn
        self.pde_fn_cp = pde_fn_cp
        self.opt = optimizer()
        self.amp = amp
        if self.amp:
            self.opt = tf.keras.mixed_precision.LossScaleOptimizer(self.opt)

        if jit_compile:
            self.train_step = tf.function(self.train_step, jit_compile=True)
            self.eval_step = tf.function(self.eval_step, jit_compile=True)
        else:
            self.train_step = tf.function(self.train_step, jit_compile=False)
            self.eval_step = tf.function(self.eval_step, jit_compile=False)
        
        if loss_fn == "sse":
            self.loss_fn = sse
        elif loss_fn == "mse":
            self.loss_fn = mse
        
        self.time_list = []
        self.functions = {
            "runge_kutta": self.rk,
            "forward": self.forward,
            "pde_fn": self.pde_fn,
            "pde_fn_cp": self.pde_fn_cp,
            "output_fn": self.output_fn,
            "extra_variables": self.extra_variables,
            "loss_fn": self.loss_fn,
        }

    def forward(self, spatial: List[tf.Tensor], time: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Perform a forward pass through the model `self.net`.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A tensor of solutions.
        """
        outputs = self.net(spatial, time)
        if self.output_fn:
            outputs = self.output_fn(outputs, *spatial, time)
        return outputs
    
    def model_step(
        self,
        batch: Dict[
            str,
            Union[
                Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]
            ],
        ],
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the
        input tensor of different conditions and data.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A dictionary of predictions.
        """
        loss = 0.0
        for loss_fn_name, data in batch.items():
            loss, preds = self.function_mapping[loss_fn_name](data, loss, self.functions)
        return loss, preds
    
    def model_step_test(
        self,
        batch: Dict[
            str,
            Union[
                Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]
            ],
        ],
        adaptive_weight
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the
        input tensor of different conditions and data.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A dictionary of predictions.
        """
        loss1 = 0.0 ## data
        loss2 = 0.0 ## collection
        loss3 = 0.0 ## non-slip
        loss4 = 0.0 ## outlet
        loss5 = 0.0 ## inlet
        loss6 = 0.0 ## choroid plexus
        count = 0
        for loss_fn_name, data in batch.items():
            if count==0: ##collection and data loss
                loss1, loss2, preds_test = self.function_mapping[loss_fn_name](data, adaptive_weight, loss1, loss2, self.functions)
            if count==1: ##non_slip_boundary
                loss3, preds = self.function_mapping[loss_fn_name](data, loss3, self.functions)
            if count==2: ##pressure_boundary_condition
                loss4, preds = self.function_mapping[loss_fn_name](data, loss4, self.functions)
            if count==3: ##inlet_boundary_condition
                loss5, preds = self.function_mapping[loss_fn_name](data, loss4, self.functions)
            if count==4: ##choroid_plexus_boundary_condition
                loss6, preds_test = self.function_mapping[loss_fn_name](data, loss6, self.functions)
            count+=1
        
        return loss1, loss2, loss3, loss4, loss5, preds_test
    
    def model_step_test_eval(
        self,
        batch: Dict[
            str,
            Union[
                Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]
            ],
        ],
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the
        input tensor of different conditions and data.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A dictionary of predictions.
        """
        loss1 = 0.0 ## data
        loss2 = 0.0 ## collection
        loss3 = 0.0 ## non-slip
        loss4 = 0.0 ## outlet
        loss5 = 0.0 ## inlet
        loss6 = 0.0 ## choroid plexus
        adaptive_weight = 1.0
        count = 0
        for loss_fn_name, data in batch.items():
            try:
                if count==0: ##collection and data loss
                    loss1, loss2, preds = self.function_mapping[loss_fn_name](data, adaptive_weight, loss1, loss2, self.functions)
                if count==1: ##non_slip_and_inlet_boundary
                    loss3, preds = self.function_mapping[loss_fn_name](data, loss3, self.functions)
                if count==2: ##pressure_boundary_condition
                    loss4, preds = self.function_mapping[loss_fn_name](data, loss4, self.functions)
                if count==3: ##inlet_boundary_condition
                    loss5, preds = self.function_mapping[loss_fn_name](data, loss4, self.functions)
                if count==4: ##choroid_plexus_boundary_condition
                    loss6, preds = self.function_mapping[loss_fn_name](data, loss6, self.functions)
            except:
                loss6, preds = self.function_mapping[loss_fn_name](data, loss6, self.functions)
            count+=1
            return loss1, loss2, loss3, loss4, loss5, preds
    
    def boundary_input_return(
        self,
        batch: Dict[
            str,
            Union[
                Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]
            ],
        ],
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        count=0
        for loss_fn_name, data in batch.items():
            print(loss_fn_name)
            if count==0:
                data_boundary = data
            count+=1
        return data_boundary
    
    def give_boundary_data(self, batch):
        with tf.GradientTape() as tape:
            data_boundary = self.boundary_input_return(batch)
            return data_boundary

    def train_step(self, batch, adaptive_weight):
        """
        Performs a single training step, including forward and backward passes.
    
        :param batch: The input batch of data for training.
        :return: The calculated loss and any extra variables to be used outside.
        """
        # Use GradientTape for automatic differentiation - to record operations for the forward pass
        with tf.GradientTape() as tape:
            #loss, pred = self.model_step(batch)
            loss1, loss2, loss3, loss4, loss5, preds = self.model_step_test(batch, adaptive_weight)    
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            # If automatic mixed precision (amp) is enabled, scale the loss to prevent underflow
            if self.amp:
                scaled_loss = self.opt.get_scaled_loss(loss)
        # If amp is enabled, compute gradients w.r.t the scaled loss
        if self.amp:
            scaled_grad = tape.gradient(scaled_loss, self.trainable_variables)
            gradients = self.opt.get_unscaled_gradients(scaled_grad)
        else:
            gradients = tape.gradient(loss, self.trainable_variables)

        # Apply the calculated gradients to the model's trainable parameters
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))   

        return loss, loss1, loss2, loss3, loss4, loss5, self.extra_variables,preds

    def eval_step(
        self, batch
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """Perform a single evaluation step on a batch of data.

        :param batch: A batch of data containing input tensors and conditions.
        :return: A tuple containing loss, error dictionary, and predictions.
        """

        x, t, u = list(batch.values())[0]
                
        #loss, preds = self.model_step(batch)
        loss1, loss2, loss3, loss4, loss5, preds = self.model_step_test_eval(batch)   
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        try:
            if self.rk:
                error_dict = {
                    solution_name: relative_l2_error(
                        preds[solution_name][:, -1][:, None], u[solution_name]
                    )
                    for solution_name in self.val_solution_names
                }
            else:
                error_dict = {
                    solution_name: relative_l2_error(preds[solution_name], u[solution_name])
                    for solution_name in self.val_solution_names
                }    
            return loss, error_dict, preds
        except:
            return loss, preds
   
    def validation_step(self, batch):
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict).
        :param batch_idx: The index of the current batch.
        """
        
        loss, error_dict, _ = self.eval_step(batch)
        
        return loss, error_dict    
    
    def test_step(self, batch):
        """Perform a single predict step on a batch of data from the test set.

        :param batch: A batch of data (a dict).
        :param batch_idx: The index of the current batch.
        """
        
        loss, error, _ = self.eval_step(batch)

        return loss, error
                
    def predict_step(self, batch):
        """Perform a single predict step on a batch of data from the prediction set.

        :param batch: A batch of data (a dict).
        :param batch_idx: The index of the current batch.
        """
        try:
            _, _, preds = self.eval_step(batch)

            return preds
        except:
            _, preds = self.eval_step(batch)
            return preds
                