from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
import numpy as np
import tensorflow as tf
import pandas as pd
import pinnstf2

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from omegaconf import DictConfig


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data in the form of a PointCloudData object.
    """

    print(root_path)
    time_file = root_path + 'time.csv'
    coordinate_file = root_path + 'coordinate.csv'

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

    for i in range (0, 40):
        filename = root_path + 'velocity_data/' + 'velocity_' + str(i) + '.csv'
        tmp_df = pd.read_csv(filename)
        tmp_u = tmp_df['Velocity:0'].to_numpy()
        tmp_v = tmp_df['Velocity:1'].to_numpy()
        tmp_w = tmp_df['Velocity:2'].to_numpy()
        tmp_p = tmp_df['Pressure'].to_numpy()
        tmp_c = tmp_df['PassiveScalar'].to_numpy()
        u = np.append(u, tmp_u)
        v = np.append(v, tmp_v)
        w = np.append(w, tmp_w)
        p = np.append(p, tmp_p)
        c = np.append(c, tmp_c)

    u = u.reshape(-1,len(x))
    v = v.reshape(-1,len(x))
    w = w.reshape(-1,len(x))
    p = p.reshape(-1,len(x))
    c = c.reshape(-1,len(x))

    u = u.T
    v = v.T
    w = w.T
    p = p.T
    c = c.T

    t_star = time * (0.1 / 3e-4)
    x_star = x / 3e-4
    y_star = y / 3e-4
    z_star = z / 3e-4
    U_star = u / 0.1
    V_star = v / 0.1
    W_star = w / 0.1
    P_star = p / (1e3 * 0.1 * 0.1)
    C_star = c

    print(t_star.shape)
    print(x_star.shape)
    print(y_star.shape)
    print(z_star.shape)
    print(U_star.shape)
    print(V_star.shape)
    print(W_star.shape)
    print(P_star.shape)
    print(C_star.shape)

    print(t_star)
    print(U_star)
    print(V_star)
    print(W_star)

    #t_star = data["t_star"]  # T x 1
    #x_star = data["x_star"]  # N x 1
    #y_star = data["y_star"]  # N x 1
    #z_star = data["z_star"]  # N x 1
#
    #U_star = data["U_star"]  # N x T
    #V_star = data["V_star"]  # N x T
    #W_star = data["W_star"]  # N x T
    #P_star = data["P_star"]  # N x T
    #C_star = data["C_star"]  # N x T
#
    return pinnstf2.data.PointCloudData(
        spatial=[x_star, y_star, z_star],
        time=[t_star],
        solution={"u": U_star, "v": V_star, "w": W_star, "p": P_star, "c": C_star},
    )

def pde_fn(outputs: Dict[str, tf.Tensor],
           x: tf.Tensor,
           y: tf.Tensor,
           z: tf.Tensor,
           t: tf.Tensor):   
    """Define the partial differential equations (PDEs).

    :param outputs: Dictionary containing the network outputs for different variables.
    :param x: Spatial coordinate x.
    :param y: Spatial coordinate y.
    :param z: Spatial coordinate z.
    :param t: Temporal coordinate t.
    :param extra_variables: Additional variables if available (optional).
    :return: Dictionary of computed PDE terms for each variable.
    """

    Pec = 3.0 / 0.101822
    Rey = 3.0 / 0.101822

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
    outputs["e5"] = u_x + v_y + w_z

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
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstf2.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
