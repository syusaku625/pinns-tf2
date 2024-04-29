from typing import Dict, List

import numpy as np
import tensorflow as tf

from .sampler_base import SamplerBase
import pandas as pd 

L = 3e-4
U = 3e-4
class MeshSampler(SamplerBase):
    """Sample from Mesh for continuous mode."""

    def __init__(
        self,
        mesh,
        idx_t: int = None,
        num_sample: int = None,
        solution: List = None,
        collection_points: List = None,
        use_lhs: bool = True,
        dtype: str = 'float32'
    ):
        """Initialize a mesh sampler for collecting training data.

        :param mesh: Instance of the mesh used for sampling.
        :param idx_t: Index of the time step.
        :param num_sample: Number of samples to generate.
        :param solution: Names of the solution outputs.
        :param collection_points: Collection points mode.
        :param use_lhs: Whether use lhs or not for generating collection points.
        """

        super().__init__(dtype)

        self.solution_names = solution
        self.collection_points_names = collection_points
        self.idx_t = idx_t

        # On a time step.
        if self.idx_t:
            flatten_mesh = mesh.on_initial_boundary(self.solution_names, self.idx_t)

        # All time steps.
        elif self.solution_names is not None:
            flatten_mesh = mesh.flatten_mesh(self.solution_names)

        if self.solution_names:
            (
                self.spatial_domain_sampled,
                self.time_domain_sampled,
                self.solution_sampled,
            ) = self.sample_mesh(num_sample, flatten_mesh)

            self.solution_sampled = tf.split(self.solution_sampled,
                                         num_or_size_splits=self.solution_sampled.shape[1],
                                         axis=1)
        
        # Collection Points only.
        else:
            (self.spatial_domain_sampled, self.time_domain_sampled) = self.convert_to_tensor(
                mesh.collection_points(num_sample, use_lhs)
            )

            self.solution_sampled = None

        self.spatial_domain_sampled = tf.split(self.spatial_domain_sampled,
                                               num_or_size_splits=self.spatial_domain_sampled.shape[1],
                                               axis=1)
        
        
    def loss_fn(self, inputs, adaptive_weight, loss1, loss2, functions):
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs

        outputs = functions["forward"](x, t)

        if self.collection_points_names:
            if functions["extra_variables"]:
                outputs = functions["pde_fn"](outputs, *x, t, functions["extra_variables"])
            else:
                outputs = functions["pde_fn"](outputs, *x, t)
        judge = "else"
        loss1 = functions["loss_fn"](loss1, outputs, judge, keys=self.collection_points_names)
        judge = "vector"
        loss2 = functions["loss_fn"](loss2, outputs, judge, u, keys=self.solution_names)
        #adaptive_weight = tf.cast(adaptive_weight, dtype=tf.float32)
        loss1 = loss1
        loss2 = loss2
        return loss1, loss2, outputs

class ChoroidPlexusMeshSampler(SamplerBase):
    """Sample from Mesh for continuous mode."""

    def __init__(
        self,
        mesh,
        idx_t: int = None,
        num_sample: int = None,
        solution: List = None,
        collection_points: List = None,
        use_lhs: bool = True,
        dtype: str = 'float32'
    ):
        """Initialize a mesh sampler for collecting training data.

        :param mesh: Instance of the mesh used for sampling.
        :param idx_t: Index of the time step.
        :param num_sample: Number of samples to generate.
        :param solution: Names of the solution outputs.
        :param collection_points: Collection points mode.
        :param use_lhs: Whether use lhs or not for generating collection points.
        """

        super().__init__(dtype)

        self.solution_names = solution
        self.collection_points_names = collection_points
        self.idx_t = idx_t

        self.spatial_domain_sampled = []
        self.solution_sampled = []
        filename = 'data/mouse33_velocity_collection/cp_node.csv'
        time_filename = 'data/mouse33_velocity_collection/Time_interpolated_tmp.csv'
        tmp_df = pd.read_csv(filename)
        time_df = pd.read_csv(time_filename)
        tmp_time = time_df['Time'].to_numpy()
        tmp_time = tmp_time * (U / L)
        tmp_x = tmp_df['Points_0'].to_numpy()
        tmp_y = tmp_df['Points_1'].to_numpy()
        tmp_z = tmp_df['Points_2'].to_numpy()

        # On a time step.
        if self.idx_t:
            tmp_time = tmp_time[idx_t].reshape(-1,1)
            
            numofnode = len(tmp_x)
            numoftime = len(tmp_time)
            tmp_time = np.vstack([tmp_time] * numofnode)

            tmp_x = np.repeat(tmp_x, numoftime).reshape(-1,1) / L
            tmp_y = np.repeat(tmp_y, numoftime).reshape(-1,1) / L
            tmp_z = np.repeat(tmp_z, numoftime).reshape(-1,1) / L

            tmp_time = tf.convert_to_tensor(tmp_time, dtype=tf.float32)
            tmp_x = tf.convert_to_tensor(tmp_x, dtype=tf.float32)
            tmp_y = tf.convert_to_tensor(tmp_y, dtype=tf.float32)
            tmp_z = tf.convert_to_tensor(tmp_z, dtype=tf.float32)

            self.spatial_domain_sampled.append(tmp_x)
            self.spatial_domain_sampled.append(tmp_y)
            self.spatial_domain_sampled.append(tmp_z)

            self.time_domain_sampled = tmp_time

            self.solution_sampled = None

        # All time steps.
        else:
            numofnode = len(tmp_x)
            numoftime = len(tmp_time)

            tmp_time = tmp_time.reshape(-1,1)
            tmp_time = np.vstack([tmp_time] * numofnode)

            tmp_x = np.repeat(tmp_x, numoftime).reshape(-1,1)
            tmp_y = np.repeat(tmp_y, numoftime).reshape(-1,1)
            tmp_z = np.repeat(tmp_z, numoftime).reshape(-1,1)

            tmp_x = tmp_x / L
            tmp_y = tmp_y / L
            tmp_z = tmp_z / L

            tmp_time = tf.convert_to_tensor(tmp_time, dtype=tf.float32)
            tmp_x = tf.convert_to_tensor(tmp_x, dtype=tf.float32)
            tmp_y = tf.convert_to_tensor(tmp_y, dtype=tf.float32)
            tmp_z = tf.convert_to_tensor(tmp_z, dtype=tf.float32)

            self.spatial_domain_sampled.append(tmp_x)
            self.spatial_domain_sampled.append(tmp_y)
            self.spatial_domain_sampled.append(tmp_z)

            self.time_domain_sampled = tmp_time

            self.solution_sampled = None

        
    def loss_fn(self, inputs, loss5, functions):
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs

        outputs = functions["forward"](x, t)

        if self.collection_points_names:
            if functions["extra_variables"]:
                outputs = functions["pde_fn_cp"](outputs, *x, t, functions["extra_variables"])
            else:
                outputs = functions["pde_fn_cp"](outputs, *x, t)
        judge = "else"
        loss5 = functions["loss_fn"](loss5, outputs, judge, keys=self.collection_points_names)
        loss5 = loss5
        return loss5, outputs


class DiscreteMeshSampler(SamplerBase):
    """Sample from Mesh for discrete mode."""

    def __init__(
        self,
        mesh,
        idx_t: int,
        num_sample: int = None,
        solution: List = None,
        collection_points: List = None,
        dtype: str = 'float32'
    ):
        """Initialize a mesh sampler for collecting training data in discrete mode.

        :param mesh: Instance of the mesh used for sampling.
        :param idx_t: Index of the time step for discrete mode.
        :param num_sample: Number of samples to generate.
        :param solution: Names of the solution outputs.
        :param collection_points: Collection points mode.
        """
        super().__init__(dtype)

        self.solution_names = solution
        self.collection_points_names = collection_points
        self.idx_t = idx_t
        self._mode = None

        flatten_mesh = mesh.on_initial_boundary(self.solution_names, self.idx_t)

        (
            self.spatial_domain_sampled,
            self.time_domain_sampled,
            self.solution_sampled,
        ) = self.sample_mesh(num_sample, flatten_mesh)

        self.spatial_domain_sampled = tf.split(self.spatial_domain_sampled,
                                               num_or_size_splits=self.spatial_domain_sampled.shape[1],
                                               axis=1)
        self.time_domain_sampled = None
        self.solution_sampled = tf.split(self.solution_sampled,
                                         num_or_size_splits=self.solution_sampled.shape[1],
                                         axis=1)

    @property
    def mode(self):
        """Get the current mode for RungeKutta class.

        :return: The current mode value.
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        """Set the mode value by PINNDataModule for RungeKutta class.

        :param value: The mode value to be set.
        """
        self._mode = value

    def loss_fn(self, inputs, loss, functions):
        """Compute the loss function based on inputs and functions. _mode is assigned in
        PINNDataModule class. It can be `inverse_discrete_1`, `inverse_discrete_2`, or
        `forward_discrete`

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs

        outputs = functions["forward"](x, t)

        if self._mode:
            if functions["extra_variables"]:
                outputs = functions["pde_fn"](outputs, *x, functions["extra_variables"])
            else:
                outputs = functions["pde_fn"](outputs, *x)
            outputs = functions["runge_kutta"](
                outputs,
                mode=self._mode,
                solution_names=self.solution_names,
                collection_points_names=self.collection_points_names,
            )
        loss = functions["loss_fn"](loss, outputs, u, keys=self.solution_names)

        return loss, outputs
