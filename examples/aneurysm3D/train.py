from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
import numpy as np
import tensorflow as tf
import pandas as pd
import pinnstf2

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omegaconf import DictConfig

L = 3e-4
U = 3e-4
D = 1e-9
rho = 1e3
mu = 7.0e-4


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data in the form of a PointCloudData object.
    """

    ##second train
    time_file = root_path + 'mouse33_velocity_collection/Time_interpolated_tmp.csv'
    ##first train
    #time_file = root_path + 'mouse33_velocity_collection/Time_interpolated.csv'
    #time_file = root_path + 'data_fine/Time.csv'
    coordinate_file = root_path + 'mouse33_velocity_collection/coordinate.csv'

    time = pd.read_csv(time_file)
    time = time['Time'].to_numpy()
    time = time.reshape(-1,1)

    coordinate = pd.read_csv(coordinate_file)
    x = coordinate['Points_0'].to_numpy()
    y = coordinate['Points_1'].to_numpy()
    z = coordinate['Points_2'].to_numpy()
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    z = z.reshape(-1,1)

    u = np.array([])
    v = np.array([])
    w = np.array([])
    p = np.array([])
    c = np.array([])

    #for i in range (19, 180, 2):
    for i in range (0, 81):
    #for i in range (0, 860, 5):
        #second train
        filename = root_path + 'mouse33_velocity_collection/contrast_interpolated_' + str(i) + '.csv'
        #filename = root_path + 'data_fine/test_' + str(i) + '.csv'
        #first train
        #filename = root_path + 'mouse33_velocity_collection/contrast_' + str(i) + '.csv'
        tmp_df = pd.read_csv(filename)
        tmp_c = tmp_df['concentration[-]'].to_numpy()
        #tmp_c = tmp_df['PassiveScalar'].to_numpy()
        #tmp_u = tmp_df['Velocityi'].to_numpy()
        #tmp_v = tmp_df['Velocityj'].to_numpy()
        #tmp_w = tmp_df['Velocityk'].to_numpy()
        #tmp_c = np.zeros(tmp_c.shape)
        tmp_u = np.zeros(tmp_c.shape)
        tmp_v = np.zeros(tmp_c.shape)
        tmp_w = np.zeros(tmp_c.shape)
        tmp_p = np.zeros(tmp_c.shape)
        p = np.append(p, tmp_p)
        c = np.append(c, tmp_c)
        u = np.append(u, tmp_u)
        v = np.append(v, tmp_v)
        w = np.append(w, tmp_w)

    u = u.reshape(-1,len(x))
    v = v.reshape(-1,len(x))
    w = w.reshape(-1,len(x))
    p = p.reshape(-1,len(x))
    c = c.reshape(-1,len(x))

    max_value = np.max(c)
    # 各配列の最大絶対値を計算
    max_abs_arr1 = np.max(np.abs(x))
    max_abs_arr2 = np.max(np.abs(y))
    max_abs_arr3 = np.max(np.abs(z))
    # 3つの最大絶対値の中で最大のものを見つける
    L_max = max(max_abs_arr1, max_abs_arr2, max_abs_arr3)

    u = u.T
    v = v.T
    w = w.T
    p = p.T
    c = c.T

    t_star = time * (U / L)
    x_star = x / L
    y_star = y / L
    z_star = z / L
    U_star = u / U
    V_star = v / U
    W_star = w / U
    P_star = p / (rho * U * U)
    C_star = c / max_value
    #C_star = c

    return pinnstf2.data.PointCloudData(
        spatial=[x_star, y_star, z_star],
        time=[t_star],
        solution={"u": U_star, "v": V_star, "w": W_star, "p": P_star, "c": C_star},
    )

def pde_fn(outputs: Dict[str, tf.Tensor],
           x: tf.Tensor,
           y: tf.Tensor,
           z: tf.Tensor,
           t: tf.Tensor,
           extra_variables: Dict[str, tf.Tensor]):   
    """Define the partial differential equations (PDEs).

    :param outputs: Dictionary containing the network outputs for different variables.
    :param x: Spatial coordinate x.
    :param y: Spatial coordinate y.
    :param z: Spatial coordinate z.
    :param t: Temporal coordinate t.
    :param extra_variables: Additional variables if available (optional).
    :return: Dictionary of computed PDE terms for each variable.
    """

    Y = tf.stack([outputs["c"], outputs["u"], outputs["v"], outputs["w"], outputs["p"]], axis=1)    
    shape = tf.shape(Y)
    Y = tf.reshape(Y, [shape[0], -1])
    
    Y_x = pinnstf2.utils.fwd_gradient(Y, x)
    Y_y = pinnstf2.utils.fwd_gradient(Y, y)
    Y_z = pinnstf2.utils.fwd_gradient(Y, z)
    Y_t = pinnstf2.utils.fwd_gradient(Y, t)

    Y_xx = pinnstf2.utils.fwd_gradient(Y_x, x)
    Y_yy = pinnstf2.utils.fwd_gradient(Y_y, y)
    Y_zz = pinnstf2.utils.fwd_gradient(Y_z, z)

    Pec = 297.8632
    Rey = rho * L * U / mu

    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    w = Y[:,3:4]
    p = Y[:,4:5]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    w_t = Y_t[:,3:4]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    w_x = Y_x[:,3:4]
    p_x = Y_x[:,4:5]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    w_y = Y_y[:,3:4]
    p_y = Y_y[:,4:5]
       
    c_z = Y_z[:,0:1]
    u_z = Y_z[:,1:2]
    v_z = Y_z[:,2:3]
    w_z = Y_z[:,3:4]
    p_z = Y_z[:,4:5]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    w_xx = Y_xx[:,3:4]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    w_yy = Y_yy[:,3:4]
       
    c_zz = Y_zz[:,0:1]
    u_zz = Y_zz[:,1:2]
    v_zz = Y_zz[:,2:3]
    w_zz = Y_zz[:,3:4]

    outputs["e1"] = c_t + (u * c_x + v * c_y + w * c_z) - (1e0 / extra_variables['l1']) * (c_xx + c_yy + c_zz)
    outputs["e2"] = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1e0 / extra_variables['l2']) * (u_xx + u_yy + u_zz)
    outputs["e3"] = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1e0 / extra_variables['l2']) * (v_xx + v_yy + v_zz)
    outputs["e4"] = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1e0 / extra_variables['l2']) * (w_xx + w_yy + w_zz)
    #outputs["e2"] = p_x - (1e0 / extra_variables['l2']) * (u_xx + u_yy + u_zz)
    #outputs["e3"] = p_y - (1e0 / extra_variables['l2']) * (v_xx + v_yy + v_zz)
    #outputs["e4"] = p_z - (1e0 / extra_variables['l2']) * (w_xx + w_yy + w_zz)
    outputs["e5"] = u_x + v_y + w_z

    return outputs

def pde_fn_cp(outputs: Dict[str, tf.Tensor],
           x: tf.Tensor,
           y: tf.Tensor,
           z: tf.Tensor,
           t: tf.Tensor,
           extra_variables: Dict[str, tf.Tensor]):   
    """Define the partial differential equations (PDEs).

    :param outputs: Dictionary containing the network outputs for different variables.
    :param x: Spatial coordinate x.
    :param y: Spatial coordinate y.
    :param z: Spatial coordinate z.
    :param t: Temporal coordinate t.
    :param extra_variables: Additional variables if available (optional).
    :return: Dictionary of computed PDE terms for each variable.
    """

    Y = tf.stack([outputs["c"], outputs["u"], outputs["v"], outputs["w"], outputs["p"]], axis=1)    
    shape = tf.shape(Y)
    Y = tf.reshape(Y, [shape[0], -1])
    
    Y_x = pinnstf2.utils.fwd_gradient(Y, x)
    Y_y = pinnstf2.utils.fwd_gradient(Y, y)
    Y_z = pinnstf2.utils.fwd_gradient(Y, z)
    Y_t = pinnstf2.utils.fwd_gradient(Y, t)

    Y_xx = pinnstf2.utils.fwd_gradient(Y_x, x)
    Y_yy = pinnstf2.utils.fwd_gradient(Y_y, y)
    Y_zz = pinnstf2.utils.fwd_gradient(Y_z, z)

    Pec = 298
    Rey = rho * L * U / mu
    #print(Rey)

    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    w = Y[:,3:4]
    p = Y[:,4:5]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    w_t = Y_t[:,3:4]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    w_x = Y_x[:,3:4]
    p_x = Y_x[:,4:5]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    w_y = Y_y[:,3:4]
    p_y = Y_y[:,4:5]
       
    c_z = Y_z[:,0:1]
    u_z = Y_z[:,1:2]
    v_z = Y_z[:,2:3]
    w_z = Y_z[:,3:4]
    p_z = Y_z[:,4:5]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    w_xx = Y_xx[:,3:4]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    w_yy = Y_yy[:,3:4]
       
    c_zz = Y_zz[:,0:1]
    u_zz = Y_zz[:,1:2]
    v_zz = Y_zz[:,2:3]
    w_zz = Y_zz[:,3:4]

    outputs["e1"] = c_t + (u * c_x + v * c_y + w * c_z) - (1.0 / Pec) * (c_xx + c_yy + c_zz)
    outputs["e2"] = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / Rey) * (u_xx + u_yy + u_zz)
    outputs["e3"] = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / Rey) * (v_xx + v_yy + v_zz)
    outputs["e4"] = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / Rey) * (w_xx + w_yy + w_zz)
    outputs["e5"] = (1.0 / L * L * L ) * (u_x + v_y + w_z) - extra_variables['l1']
    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    pinnstf2.utils.extras(cfg)

    # train the model
    metric_dict, _ = pinnstf2.train(
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, pde_fn_cp=pde_fn_cp,output_fn=None
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstf2.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
