import numpy as np
import tensorflow as tf

from pinnstf2 import utils
import pandas as pd
from .sampler_base import SamplerBase

L = 3e-4
U = 3e-4
rho = 1e3
mu = 7.0e-4

class DirichletBoundaryCondition(SamplerBase):
    """Initialize Dirichlet boundary condition."""

    def __init__(
        self,
        mesh,
        solution,
        num_sample: int = None,
        idx_t: int = None,
        boundary_fun=None,
        discrete: bool = False,
        dtype: str = 'float32'
    ):
        """Initialize a mesh sampler for collecting training data in upper and lower boundaries for
        Dirichlet boundary condition.

        :param mesh: Instance of the mesh used for sampling.
        :param solution: Names of the solution outputs.
        :param num_sample: Number of samples to generate.
        :param idx_t: Index of the time step for discrete mode.
        :param boundary_fun: A function can apply on boundary data.
        :param discrete: It is a boolean that is true when problem is discrete.
        """
        

        super().__init__(dtype)

        self.solution_names = solution
        self.discrete = discrete

        spatial_upper_bound, time_upper_bound, solution_upper_bound = mesh.on_upper_boundary(
            self.solution_names
        )
        spatial_lower_bound, time_lower_bound, solution_lower_bound = mesh.on_lower_boundary(
            self.solution_names
        )

        spatial_bound = np.vstack([spatial_upper_bound, spatial_lower_bound])
        
        self.idx_t = idx_t

        self.spatial_domain_sampled = []
        self.solution_sampled = []

        filename = 'data/mouse33_velocity_collection/non_slip_x.csv'
        time_filename = 'data/mouse33_velocity_collection/Time_interpolated_tmp.csv'
        sample_number = 61440

        tmp_df = pd.read_csv(filename)
        time_df = pd.read_csv(time_filename)

        tmp_time = time_df['Time'].to_numpy()

        # 各配列の最大絶対値を計算
        max_abs_arr1 = np.max(np.abs(tmp_df['Points_0'].to_numpy()))
        max_abs_arr2 = np.max(np.abs(tmp_df['Points_1'].to_numpy()))
        max_abs_arr3 = np.max(np.abs(tmp_df['Points_2'].to_numpy()))
        # 3つの最大絶対値の中で最大のものを見つける
        L_max = max(max_abs_arr1, max_abs_arr2, max_abs_arr3)

        tmp_time = tmp_time * (U / L)
        tmp_x = tmp_df['Points_0'].to_numpy() / L
        tmp_y = tmp_df['Points_1'].to_numpy() / L
        tmp_z = tmp_df['Points_2'].to_numpy() / L

        # ランダムにインデックスを選択（重複なしで選択する場合はreplace=Falseを設定）
        indexes = np.random.choice(len(tmp_x), size=sample_number, replace=False)

        # 選択されたインデックスに基づいてx, y, z座標をサンプリング
        tmp_x = tmp_x[indexes]
        tmp_y = tmp_y[indexes]
        tmp_z = tmp_z[indexes]

        numofwall = len(tmp_x)
        numoftime = len(tmp_time)
        numofnode = len(tmp_x)

        tmp_time = tmp_time.reshape(-1,1)
        tmp_time = np.vstack([tmp_time] * numofnode)

        tmp_x = np.repeat(tmp_x, numoftime)
        tmp_y = np.repeat(tmp_y, numoftime)
        tmp_z = np.repeat(tmp_z, numoftime)
        tmp_x = tmp_x.reshape(-1,1)
        tmp_y = tmp_y.reshape(-1,1)
        tmp_z = tmp_z.reshape(-1,1)

        tmp_u = np.zeros(tmp_time.shape)
        tmp_v = np.zeros(tmp_time.shape)
        tmp_w = np.zeros(tmp_time.shape)

        tmp_time = tf.convert_to_tensor(tmp_time, dtype=tf.float32)
        tmp_x = tf.convert_to_tensor(tmp_x, dtype=tf.float32)
        tmp_y = tf.convert_to_tensor(tmp_y, dtype=tf.float32)
        tmp_z = tf.convert_to_tensor(tmp_z, dtype=tf.float32)

        tmp_u = tf.convert_to_tensor(tmp_u, dtype=tf.float32)
        tmp_v = tf.convert_to_tensor(tmp_v, dtype=tf.float32)
        tmp_w = tf.convert_to_tensor(tmp_w, dtype=tf.float32)

        self.spatial_domain_sampled.append(tmp_x)
        self.spatial_domain_sampled.append(tmp_y)
        self.spatial_domain_sampled.append(tmp_z)

        self.solution_sampled.append(tmp_u)
        self.solution_sampled.append(tmp_v)
        self.solution_sampled.append(tmp_w)

        tmp_time = tf.convert_to_tensor(tmp_time, dtype=tf.float32)
        self.time_domain_sampled = tmp_time

    def sample_mesh(self, num_sample, flatten_mesh):
        """Sample the mesh data for training. If idx_t is defined, only points on that time will be
        selected. If num_sample is not defined the whole points will be selected.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: Flattened mesh data.
        :return: Sampled spatial, time, and solution data.
        """

    def loss_fn(self, inputs, loss, functions):
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs

        # In discrete mode, we do not use time.
        if self.discrete:
            t = None

        outputs = functions["forward"](x, t)
        judge = "else"
        loss = functions["loss_fn"](loss, outputs, judge, u, keys=self.solution_names)

        loss = loss

        return loss, outputs
    
class PressureBoundaryCondition(SamplerBase):
    """Initialize Dirichlet boundary condition."""

    def __init__(
        self,
        mesh,
        solution,
        num_sample: int = None,
        idx_t: int = None,
        boundary_fun=None,
        discrete: bool = False,
        dtype: str = 'float32'
    ):  

        super().__init__(dtype)

        self.solution_names = solution
        self.discrete = discrete

        spatial_upper_bound, time_upper_bound, solution_upper_bound = mesh.on_upper_boundary(
            self.solution_names
        )
        spatial_lower_bound, time_lower_bound, solution_lower_bound = mesh.on_lower_boundary(
            self.solution_names
        )

        spatial_bound = np.vstack([spatial_upper_bound, spatial_lower_bound])
#
        self.idx_t = idx_t
        
        self.spatial_domain_sampled = []
        self.solution_sampled = []

        filename = 'data/mouse33_velocity_collection/outlet_x.csv'
        time_filename = 'data/mouse33_velocity_collection/Time_interpolated_tmp.csv'

        tmp_df = pd.read_csv(filename)
        time_df = pd.read_csv(time_filename)

        # 各配列の最大絶対値を計算
        max_abs_arr1 = np.max(np.abs(tmp_df['Points_0'].to_numpy()))
        max_abs_arr2 = np.max(np.abs(tmp_df['Points_1'].to_numpy()))
        max_abs_arr3 = np.max(np.abs(tmp_df['Points_2'].to_numpy()))
        # 3つの最大絶対値の中で最大のものを見つける
        L_max = max(max_abs_arr1, max_abs_arr2, max_abs_arr3)

        tmp_time = time_df['Time'].to_numpy()
        tmp_time = tmp_time * (U / L)
        tmp_x = tmp_df['Points_0'].to_numpy() / L
        tmp_y = tmp_df['Points_1'].to_numpy() / L
        tmp_z = tmp_df['Points_2'].to_numpy() / L

        numofnode = len(tmp_x)
        numoftime = len(tmp_time)

        tmp_time = tmp_time.reshape(-1,1)
        tmp_time = np.vstack([tmp_time] * numofnode)

        tmp_x = np.repeat(tmp_x, numoftime)
        tmp_y = np.repeat(tmp_y, numoftime)
        tmp_z = np.repeat(tmp_z, numoftime)
        tmp_x = tmp_x.reshape(-1,1)
        tmp_y = tmp_y.reshape(-1,1)
        tmp_z = tmp_z.reshape(-1,1)

        tmp_x = tmp_x
        tmp_y = tmp_y
        tmp_z = tmp_z

        tmp_time = tf.convert_to_tensor(tmp_time, dtype=tf.float32)
        tmp_x = tf.convert_to_tensor(tmp_x, dtype=tf.float32)
        tmp_y = tf.convert_to_tensor(tmp_y, dtype=tf.float32)
        tmp_z = tf.convert_to_tensor(tmp_z, dtype=tf.float32)

        self.spatial_domain_sampled.append(tmp_x)
        self.spatial_domain_sampled.append(tmp_y)
        self.spatial_domain_sampled.append(tmp_z)

        tmp_p = np.zeros(tmp_time.shape)
        tmp_p = tf.convert_to_tensor(tmp_p, dtype=tf.float32)

        self.solution_sampled.append(tmp_p)

        tmp_time = tf.convert_to_tensor(tmp_time, dtype=tf.float32)
        self.time_domain_sampled = tmp_time

    def sample_mesh(self, num_sample, flatten_mesh):
        """Sample the mesh data for training. If idx_t is defined, only points on that time will be
        selected. If num_sample is not defined the whole points will be selected.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: Flattened mesh data.
        :return: Sampled spatial, time, and solution data.
        """

    def loss_fn(self, inputs, loss, functions):
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs

        # In discrete mode, we do not use time.
        if self.discrete:
            t = None

        outputs = functions["forward"](x, t)
        judge = "else"
        loss = functions["loss_fn"](loss, outputs, judge, u, keys=self.solution_names)

        loss = loss
        return loss, outputs

class InletBoundaryCondition(SamplerBase):
    """Initialize Dirichlet boundary condition."""

    def __init__(
        self,
        mesh,
        solution,
        num_sample: int = None,
        idx_t: int = None,
        boundary_fun=None,
        discrete: bool = False,
        dtype: str = 'float32'
    ):  

        super().__init__(dtype)

        self.solution_names = solution
        self.discrete = discrete

        spatial_upper_bound, time_upper_bound, solution_upper_bound = mesh.on_upper_boundary(
            self.solution_names
        )
        spatial_lower_bound, time_lower_bound, solution_lower_bound = mesh.on_lower_boundary(
            self.solution_names
        )

        spatial_bound = np.vstack([spatial_upper_bound, spatial_lower_bound])
#
        self.idx_t = idx_t
        
        self.spatial_domain_sampled = []
        self.solution_sampled = []

        time_filename = 'data/mouse33_velocity_collection/Time_interpolated_tmp.csv'
        inlet_node_file = 'data/mouse33_velocity_collection/inlet_coordinate.csv'
        area_value = 9.07502e-08
        injecttion_temporal_number = 6
        flow_rate = 0.25 #micro l /min
        flow_rate = flow_rate * 1e-6 # l/min
        flow_rate = flow_rate * 1e-3 #m3/min
        flow_rate = flow_rate / 60.0 #m3/s
        inlet_velocity = flow_rate / area_value
        inlet_velocity = inlet_velocity / U

        time_df = pd.read_csv(time_filename)
        tmp_df = pd.read_csv(inlet_node_file)

        tmp_time = time_df['Time'].to_numpy()

        # 各配列の最大絶対値を計算
        max_abs_arr1 = np.max(np.abs(tmp_df['Points_0'].to_numpy()))
        max_abs_arr2 = np.max(np.abs(tmp_df['Points_1'].to_numpy()))
        max_abs_arr3 = np.max(np.abs(tmp_df['Points_2'].to_numpy()))
        # 3つの最大絶対値の中で最大のものを見つける
        L_max = max(max_abs_arr1, max_abs_arr2, max_abs_arr3)

        tmp_time = tmp_time * (U / L)
        tmp_x = tmp_df['Points_0'].to_numpy() / L
        tmp_y = tmp_df['Points_1'].to_numpy() / L
        tmp_z = tmp_df['Points_2'].to_numpy() / L

        numofnode = len(tmp_x)
        numoftime = len(tmp_time)

        tmp_time = tmp_time.reshape(-1,1)
        tmp_time = np.vstack([tmp_time] * numofnode)

        tmp_x = np.repeat(tmp_x, numoftime)
        tmp_y = np.repeat(tmp_y, numoftime)
        tmp_z = np.repeat(tmp_z, numoftime)
        tmp_x = tmp_x.reshape(-1,1)
        tmp_y = tmp_y.reshape(-1,1)
        tmp_z = tmp_z.reshape(-1,1)

        tmp_time = tf.convert_to_tensor(tmp_time, dtype=tf.float32)
        tmp_x = tf.convert_to_tensor(tmp_x, dtype=tf.float32)
        tmp_y = tf.convert_to_tensor(tmp_y, dtype=tf.float32)
        tmp_z = tf.convert_to_tensor(tmp_z, dtype=tf.float32)

        self.spatial_domain_sampled.append(tmp_x)
        self.spatial_domain_sampled.append(tmp_y)
        self.spatial_domain_sampled.append(tmp_z)

        tmp_u = np.zeros(tmp_time.shape)
        tmp_v = np.zeros(tmp_time.shape)
        tmp_w = np.zeros(tmp_time.shape)
        tmp_c = np.zeros(tmp_time.shape)

        #inpose inlet velocity
        for i in range(0, numofnode):
            for j in range(0, injecttion_temporal_number):
                tmp_u[i*numoftime+j] = -inlet_velocity
                tmp_c[i*numoftime+j] = 1e0

        tmp_u = tf.convert_to_tensor(tmp_u, dtype=tf.float32)
        tmp_v = tf.convert_to_tensor(tmp_v, dtype=tf.float32)
        tmp_w = tf.convert_to_tensor(tmp_w, dtype=tf.float32)
        tmp_c = tf.convert_to_tensor(tmp_c, dtype=tf.float32)

        self.solution_sampled.append(tmp_u)
        self.solution_sampled.append(tmp_v)
        self.solution_sampled.append(tmp_w)
        self.solution_sampled.append(tmp_c)

        tmp_time = tf.convert_to_tensor(tmp_time, dtype=tf.float32)
        self.time_domain_sampled = tmp_time

    def sample_mesh(self, num_sample, flatten_mesh):
        """Sample the mesh data for training. If idx_t is defined, only points on that time will be
        selected. If num_sample is not defined the whole points will be selected.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: Flattened mesh data.
        :return: Sampled spatial, time, and solution data.
        """

    def loss_fn(self, inputs, loss, functions):
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs

        # In discrete mode, we do not use time.
        if self.discrete:
            t = None

        outputs = functions["forward"](x, t)
        judge = "else"
        loss = functions["loss_fn"](loss, outputs, judge, u, keys=self.solution_names)

        loss = loss

        return loss, outputs

class PeriodicBoundaryCondition(SamplerBase):
    """Initialize Periodic boundary condition."""

    def __init__(
        self,
        mesh,
        solution,
        idx_t: int = None,
        num_sample: int = None,
        derivative_order: int = 0,
        discrete: bool = False,
        dtype: str = 'float32'
    ):
        super().__init__(dtype)
        """Initialize a mesh sampler for collecting training data in upper and lower boundaries for
        periodic boundary condition.

        :param mesh: Instance of the mesh used for sampling.
        :param solution: Names of the solution outputs.
        :param num_sample: Number of samples to generate.
        :param idx_t: Index of the time step for discrete mode.
        :param boundary_fun: A function can apply on boundary data.
        :param discrete: It is a boolean that is true when problem is discrete.
        """

        self.derivative_order = derivative_order
        self.idx_t = idx_t
        self.solution_names = solution

        spatial_upper_bound, time_upper_bound, _ = mesh.on_upper_boundary(self.solution_names)
        spatial_lower_bound, time_lower_bound, _ = mesh.on_lower_boundary(self.solution_names)

        self.discrete = discrete

        (self.spatial_domain_sampled, self.time_domain_sampled) = self.sample_mesh(
            num_sample,
            (spatial_upper_bound, time_upper_bound, spatial_lower_bound, time_lower_bound),
        )

        self.mid = len(self.time_domain_sampled) // 2
        self.spatial_domain_sampled = tf.split(self.spatial_domain_sampled,
                                               num_or_size_splits=self.spatial_domain_sampled.shape[1],
                                               axis=1)


    def sample_mesh(self, num_sample, flatten_mesh):
        """Sample the mesh data for training.

        :param num_sample: Number of samples to generate.
        :param flatten_mesh: Flattened mesh data.
        :return: Sampled spatial, time, and solution data.
        """

        if self.discrete:
            flatten_mesh = [
                flatten_mesh_[self.idx_t : self.idx_t + 1, :] for flatten_mesh_ in flatten_mesh
            ]

        if num_sample is None:
            return self.convert_to_tensor(
                (
                    np.vstack((flatten_mesh[0], flatten_mesh[2])),
                    np.vstack((flatten_mesh[1], flatten_mesh[3])),
                )
            )
        else:
            idx = np.random.choice(range(flatten_mesh[0].shape[0]), num_sample, replace=False)
            return self.convert_to_tensor(
                (
                    np.vstack((flatten_mesh[0][idx, :], flatten_mesh[2][idx, :])),
                    np.vstack((flatten_mesh[1][idx, :], flatten_mesh[3][idx, :])),
                )
            )

    def loss_fn(self, inputs, loss, functions):
        """Compute the loss function based on inputs and functions.

        :param inputs: Input data for computing the loss.
        :param loss: Loss variable.
        :param functions: Additional functions required for loss computation.
        :return: Loss variable and outputs dict from the forward pass.
        """

        x, t, u = inputs

        # In discrete mode, we do not use time.
        if self.discrete:
            t = None

        outputs = functions["forward"](x, t)

        if self.derivative_order > 0:
            for solution_name in self.solution_names:
                if self.discrete:
                    outputs["tmp"] = utils.fwd_gradient(outputs[solution_name], x)
                else:
                    outputs["tmp"] = utils.gradient(outputs[solution_name], x)

                loss = functions["loss_fn"](loss, outputs, keys=["tmp"], mid=self.mid)

        loss = functions["loss_fn"](loss, outputs, keys=self.solution_names, mid=self.mid)

        return loss, outputs
