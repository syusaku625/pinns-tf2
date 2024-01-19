import numpy as np
import tensorflow as tf
import pandas as pd
from .sampler_base import SamplerBase

#L = 3e-4
#U = 0.1
#rho = 1e3
#mu = 1e-3

L = 3e-4
U = 3e-4
rho = 1e3
mu = 1e-3

class InitialCondition(SamplerBase):
    """Initialize initial boundary condition."""

    def __init__(self, mesh, num_sample=None, solution=None, initial_fun=None, dtype: str = 'float32'):
        """Initialize an InitialCondition object for sampling initial condition data.

        :param mesh: Mesh object containing spatial and time domain information.
        :param num_sample: Number of samples.
        :param solution: List of solution variable names.
        :param initial_fun: Function to generate initial conditions (optional).
        """
        super().__init__(dtype)

        self.solution_names = solution

        (self.spatial_domain, self.time_domain, self.solution) = mesh.on_initial_boundary(
            self.solution_names
        )

        print(type(self.spatial_domain))
        print(type(self.time_domain))
        print(self.solution)
        #exit(1)
        self.spatial_domain_sampled = []
        self.solution_sampled = []
        filename = 'data/mouse_mesh_version2/initial_contrast.csv'
        tmp_df = pd.read_csv(filename)
        tmp_x = tmp_df['Points_0'].to_numpy()
        tmp_y = tmp_df['Points_1'].to_numpy()
        tmp_z = tmp_df['Points_2'].to_numpy()
        tmp_contrast = tmp_df['contrast'].to_numpy()
        tmp_x = np.repeat(tmp_x, 2)
        tmp_y = np.repeat(tmp_y, 2)
        tmp_z = np.repeat(tmp_z, 2)
        tmp_contrast = np.repeat(tmp_contrast, 2)
        tmp_x = tmp_x.reshape(-1,1)
        tmp_y = tmp_y.reshape(-1,1)
        tmp_z = tmp_z.reshape(-1,1)
        tmp_contrast = tmp_contrast.reshape(-1,1)

        tmp_x = tmp_x / L
        tmp_y = tmp_y / L
        tmp_z = tmp_z / L

        tmp_x = tf.convert_to_tensor(tmp_x, dtype=tf.float32)
        tmp_y = tf.convert_to_tensor(tmp_y, dtype=tf.float32)
        tmp_z = tf.convert_to_tensor(tmp_z, dtype=tf.float32)
        tmp_contrast = tf.convert_to_tensor(tmp_contrast, dtype=tf.float32)

        self.spatial_domain_sampled.append(tmp_x)
        self.spatial_domain_sampled.append(tmp_y)
        self.spatial_domain_sampled.append(tmp_z)
        self.solution_sampled.append(tmp_contrast)

        tmp_time = np.zeros(tmp_contrast.shape)
        tmp_time = tf.convert_to_tensor(tmp_time, dtype=tf.float32)
        self.time_domain_sampled = tmp_time

        #print(self.solution_sampled)
        #print(self.time_domain_sampled)
        #print(self.spatial_domain_sampled)
        #exit(1)
        #if initial_fun:
        #    self.solution = initial_fun(self.spatial_domain)
#
        #(
        #    self.spatial_domain_sampled,
        #    self.time_domain_sampled,
        #    self.solution_sampled,
        #) = self.sample_mesh(num_sample, (self.spatial_domain, self.time_domain, self.solution))
#
        #self.spatial_domain_sampled = tf.split(self.spatial_domain_sampled,
        #                                       num_or_size_splits=self.spatial_domain_sampled.shape[1],
        #                                       axis=1)
        #self.solution_sampled = tf.split(self.solution_sampled,
        #                                 num_or_size_splits=self.solution_sampled.shape[1],
        #                                 axis=1)

    def sample_mesh(self, num_sample, flatten_mesh):
        """Sample the mesh data for training. If num_sample is not defined the whole points will be
        selected.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: Flattened mesh data.
        :return: Sampled spatial, time, and solution data.
        """
        flatten_mesh = list(flatten_mesh)
        concatenated_solutions = [
            flatten_mesh[2][solution_name] for solution_name in self.solution_names
        ]
        flatten_mesh[2] = np.concatenate(concatenated_solutions, axis=-1)

        if num_sample is None:
            return self.convert_to_tensor(flatten_mesh)
        else:
            idx = np.random.choice(range(flatten_mesh[0].shape[0]), num_sample, replace=False)
            return self.convert_to_tensor(
                (flatten_mesh[0][idx, :], flatten_mesh[1][idx, :], flatten_mesh[2][idx, :])
            )

    def loss_fn(self, inputs, loss, functions):
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs

        outputs = functions["forward"](x, t)
        loss = functions["loss_fn"](loss, outputs, u, keys=self.solution_names)

        return loss, outputs
