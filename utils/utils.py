import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

from omegaconf import OmegaConf
from pathlib import Path

import utils.io_dict_to_hdf5 as ioh5
import jax.numpy as jnp

##### Custom Resolver #####
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if_multi', lambda pred, a, b: a if pred.name=='MULTIRUN' else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


def add_colorbar(mappable,linewidth=2,location='right',**kwargs):
    ''' modified from https://supy.readthedocs.io/en/2021.3.30/_modules/supy/util/_plot.html'''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, drawedges=False,**kwargs)
    cbar.outline.set_linewidth(linewidth)
    plt.sca(last_axes)
    return cbar

def map_discrete_cbar(cmap,N):
    cmap = plt.get_cmap(cmap,N+1)
    bounds = np.arange(-.5,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def load_cfg(cfg_path, default_model_config=None):
    """ Load configuration file and merge with default model configuration

    Args:
        cfg_path (string): path to configuration file
        default_model_config (string): path to default model configuration file to merge with

    Returns:
        cfg: returns the merged configuration data
    """
    cfg = OmegaConf.load(cfg_path)
    for k in cfg.paths.keys():
        if k != 'user':
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
    params_curr = cfg.dataset
    if default_model_config is not None:
        params = OmegaConf.load(default_model_config)
        cfg.dataset = OmegaConf.merge(params, params_curr)
    return cfg


def load_feco_data( fpath, fnum=-1 ):
    """ Load rollout data from a .pkl file
    """
    pkl_files = list(Path(fpath).glob('rollout*.pkl'))
    pfile = pkl_files[fnum]
    data = pickle.load(open(pfile, 'rb'))

    leg_qpos_idx = np.arange(7,43)
    leg_qvel_idx = np.arange(6,42)
    all_poses = []
    all_vel = []
    for jj in leg_qpos_idx:
        jp = []
        for key in data.keys():
            if 'qposes' in data[key].keys():
                jp.append(data[key]['qposes'][:,jj])
        all_poses.append(np.concatenate(jp,axis=0))

    for jj in leg_qvel_idx:
        jp2 = []
        for key in data.keys():
            if 'qvels' in data[key].keys():
                jp2.append(data[key]['qvels'][:,jj])
        all_vel.append(np.concatenate(jp2,axis=0))

    all_poses = jnp.array(all_poses).T
    all_vel = jnp.array(all_vel).T
    dic = {'poses': all_poses, 'vels': all_vel, 
           'ameans':jnp.mean(all_poses,axis=0), 'astd':jnp.std(all_poses,axis=0), 
           'vmeans':jnp.mean(all_vel,axis=0), 'vstd':jnp.std(all_vel,axis=0) 
           }
    return dic
