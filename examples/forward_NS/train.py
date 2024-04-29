from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
import pandas as pd
import pinnstf2

rho = 1e3
mu = 1e-3
L = 14e-3
U = 1.0e-2
Re = rho * L * U / mu

def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data in the form of a PointCloudData object.
    """

    #data = pinnstf2.utils.load_data(root_path, "cylinder_nektar_wake.mat")
    coordinate_file = '/mnt/d/forward_problem/node.csv'
    time_file = '/mnt/d/forward_problem/Time.csv'
    coordinate = pd.read_csv(coordinate_file)
    x = coordinate['Points_0'].to_numpy() * 1e-3 / L
    y = coordinate['Points_1'].to_numpy() * 1e-3 / L
    time = pd.read_csv(time_file) 
    time = time['Time'].to_numpy() * (U / L)
    time = time.reshape(-1,1)
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    u = np.array([])
    v = np.array([])
    p = np.array([])
    for i in range (0, 5):
        tmp_x = coordinate['Points_0'].to_numpy()
        tmp_u = np.zeros(tmp_x.shape)
        tmp_v = np.zeros(tmp_x.shape)
        tmp_p = np.zeros(tmp_x.shape)
        u = np.append(u, tmp_u)
        v = np.append(v, tmp_v)
        p = np.append(p, tmp_p)
    
    u = u.reshape(-1,len(x))
    v = v.reshape(-1,len(x))
    p = p.reshape(-1,len(x))

    u = u.T
    v = v.T
    p = p.T

    print(u)
    print(v)
    print(p)

    return pinnstf2.data.PointCloudData(
        spatial=[x, y], time=[time], solution={"u": u, "v": v, "p": p}
    )

def output_fn(outputs: Dict[str, tf.Tensor],
              x: tf.Tensor,
              y: tf.Tensor,
              t: tf.Tensor):
    """Define `output_fn` function that will be applied to outputs of net."""

    outputs["v"] = -pinnstf2.utils.gradient(outputs["psi"], x)
    outputs["u"] = pinnstf2.utils.gradient(outputs["psi"], y)

    return outputs

def pde_fn(outputs: Dict[str, tf.Tensor],
           x: tf.Tensor,
           y: tf.Tensor,
           t: tf.Tensor):   
    """Define the partial differential equations (PDEs)."""

    u_x, u_y, u_t = pinnstf2.utils.gradient(outputs["u"], [x, y, t])
    u_xx = pinnstf2.utils.gradient(u_x, x)
    u_yy = pinnstf2.utils.gradient(u_y, y)

    v_x, v_y, v_t = pinnstf2.utils.gradient(outputs["v"], [x, y, t])
    v_xx = pinnstf2.utils.gradient(v_x, x)
    v_yy = pinnstf2.utils.gradient(v_y, y)

    p_x, p_y = pinnstf2.utils.gradient(outputs["p"], [x, y])

    outputs["f_u"] = u_t + (outputs["u"] * u_x + outputs["v"] * u_y) + p_x - (1e0 / Re) * (u_xx + u_yy)
    outputs["f_v"] = v_t + (outputs["u"] * v_x + outputs["v"] * v_y) + p_y - (1e0 / Re) * (v_xx + v_yy)

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
        cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, pde_fn_cp = pde_fn_cp, output_fn=output_fn
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = pinnstf2.utils.get_metric_value(
        metric_dict=metric_dict, metric_names=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
