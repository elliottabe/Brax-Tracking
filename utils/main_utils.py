import os
import subprocess as sp

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

import functools
from utils.utils import *
from custom_brax import custom_ppo_networks
from custom_brax import network_masks as masks
from custom_brax import custom_networks as networks
import flax.linen as nn

#### setup network constructors ####
def setup_network_factory( cfg, rollout_dir='rollout_data/' ):
    # load feco data
    feco_data = load_feco_data(cfg.paths.data_dir / rollout_dir )

    # setup network types
    network_factory = None # observation_size = env_state.obs.shape[-1], task_obs_size=task_obs_size, action_size=env.action_size, preprocess_observations_fn=normalize
    checkpoint_network_factory = None        
    network_type = custom_ppo_networks.make_intention_ppo_networks # default network type

    if  ('network_type' in cfg.train) and (cfg.train['network_type'] is not None):
        if ('encoderdecoder' in cfg.train['network_type']):
            network_type = custom_ppo_networks.make_encoderdecoder_ppo_networks
        elif ('sensoryintention' in cfg.train['network_type']):
            print('--------------- Using sensory intention network ---------------')
            network_type = custom_ppo_networks.make_sensory_intention_ppo_networks
            network_factory=functools.partial(
                network_type,
                encoder_hidden_layer_sizes=cfg.train['encoder_hidden_layer_sizes'],
                decoder_hidden_layer_sizes=cfg.train['decoder_hidden_layer_sizes'],
                value_hidden_layer_sizes=cfg.train['value_hidden_layer_sizes'],
                sensory_hidden_layer_sizes=cfg.train['sensory_hidden_layer_sizes'], 
                )
            checkpoint_network_factory=functools.partial(
                    network_type,
                    intention_latent_size=60,
                    encoder_hidden_layer_sizes=cfg.train.ckpt_net['encoder_hidden_layer_sizes'],
                    decoder_hidden_layer_sizes=cfg.train.ckpt_net['decoder_hidden_layer_sizes'],
                    value_hidden_layer_sizes=cfg.train.ckpt_net['value_hidden_layer_sizes'],
                    sensory_hidden_layer_sizes=cfg.train['sensory_hidden_layer_sizes'],
                    )

        elif ('fecointention' in cfg.train['network_type']):
            print('--------------- Using feco intention network ---------------')
            network_type = custom_ppo_networks.make_sensory_feco_intention_ppo_networks
            network_factory=functools.partial(
                network_type,
                encoder_hidden_layer_sizes=cfg.train['encoder_hidden_layer_sizes'],
                decoder_hidden_layer_sizes=cfg.train['decoder_hidden_layer_sizes'],
                value_hidden_layer_sizes=cfg.train['value_hidden_layer_sizes'],
                sensory_neurons=cfg.train['sensory_neurons'],
                joints = feco_data['ameans'].shape[0],
                angle_means = feco_data['ameans'],
                angle_std = feco_data['astd'],
                vel_means = feco_data['vmeans'],
                vel_std = feco_data['vstd'],
                random_bias = cfg.train['random_feco_bias'],
                )
            checkpoint_network_factory=functools.partial(
                    network_type,
                    intention_latent_size=60,
                    encoder_hidden_layer_sizes=cfg.train.ckpt_net['encoder_hidden_layer_sizes'],
                    decoder_hidden_layer_sizes=cfg.train.ckpt_net['decoder_hidden_layer_sizes'],
                    value_hidden_layer_sizes=cfg.train.ckpt_net['value_hidden_layer_sizes'],
                    sensory_neurons=cfg.train['sensory_neurons'],
                    joints = feco_data['ameans'].shape[0],
                    angle_means = feco_data['ameans'],
                    angle_std = feco_data['astd'],
                    vel_means = feco_data['vmeans'],
                    vel_std = feco_data['vstd'],
                    random_bias = cfg.train['random_feco_bias'],
                    )
            
        
        elif ('separatefeco' in cfg.train['network_type']):
            print('--------------- Using separate sensory feco and intention networks ---------------')
            act_keys = ['activation', 'activation_hook', 'activation_claw']
            afn = dict()
            for aa in act_keys:
                if aa in cfg.train:
                    if cfg.train[aa] == 'gaussian':
                        print('Using gaussian activation for ', aa)
                        afn[aa] = networks.gaussian_activation
                    elif cfg.train[aa] == 'relu':
                        afn[aa] = nn.relu
                    elif cfg.train[aa] == 'tanh':
                        afn[aa] = nn.tanh
                else:
                    afn[aa] = nn.relu
            iclaw = cfg.train['invert_claw'] if 'invert_claw' in cfg.train else False
            
            network_type = custom_ppo_networks.make_separate_sensory_and_intention_networks
            network_factory=functools.partial(
                network_type,
                intention_latent_size=cfg.train['intention_latent_size'],
                encoder_hidden_layer_sizes=cfg.train['encoder_hidden_layer_sizes'],
                decoder_hidden_layer_sizes=cfg.train['decoder_hidden_layer_sizes'],
                value_hidden_layer_sizes=cfg.train['value_hidden_layer_sizes'],
                joints = feco_data['ameans'].shape[0],
                nneurons=cfg.train['sensory_neurons'],
                angle_means = feco_data['ameans'],
                angle_std = feco_data['astd'],
                vel_means = feco_data['vmeans'],
                vel_std = feco_data['vstd'],
                std_scale = cfg.train['std_scale'],
                random_bias = cfg.train['random_feco_bias'],
                invert_claw=iclaw,
                **afn,
                )
            checkpoint_network_factory=functools.partial(
                    network_type,
                    intention_latent_size=cfg.train['intention_latent_size'],
                    encoder_hidden_layer_sizes=cfg.train.ckpt_net['encoder_hidden_layer_sizes'],
                    decoder_hidden_layer_sizes=cfg.train.ckpt_net['decoder_hidden_layer_sizes'],
                    value_hidden_layer_sizes=cfg.train.ckpt_net['value_hidden_layer_sizes'],
                    joints = feco_data['ameans'].shape[0],
                    nneurons=cfg.train['sensory_neurons'],
                    angle_means = feco_data['ameans'],
                    angle_std = feco_data['astd'],
                    vel_means = feco_data['vmeans'],
                    vel_std = feco_data['vstd'],
                    std_scale = cfg.train['std_scale'],
                    random_bias = cfg.train['random_feco_bias'],
                    invert_claw=iclaw,
                    **afn,
                    )
            
    
    if network_factory is None:
        print('--------------- Using default intention network ---------------')
        network_factory=functools.partial(
            network_type,
            encoder_hidden_layer_sizes=cfg.train['encoder_hidden_layer_sizes'],
            decoder_hidden_layer_sizes=cfg.train['decoder_hidden_layer_sizes'],
            value_hidden_layer_sizes=cfg.train['value_hidden_layer_sizes'],
            )
        checkpoint_network_factory=functools.partial(
                network_type,
                intention_latent_size=60,
                encoder_hidden_layer_sizes=cfg.train.ckpt_net['encoder_hidden_layer_sizes'],
                decoder_hidden_layer_sizes=cfg.train.ckpt_net['decoder_hidden_layer_sizes'],
                value_hidden_layer_sizes=cfg.train.ckpt_net['value_hidden_layer_sizes'],
                )
        
    return network_factory, checkpoint_network_factory


#### setup masks for gradient freezing ####
def setup_masks(cfg):
    freeze_mask_fn = None
    if  ('freeze_sensory' in cfg.train) and (cfg.train['freeze_sensory'] == True):
        if (cfg.train['freeze_decoder'] == True):
            print('--------------- freezing sensory and decoder network ---------------')
            freeze_mask_fn = masks.create_multiple_masks
        else:
            print('--------------- freezing sensory network ---------------')
            freeze_mask_fn = masks.create_sensory_mask
    elif (cfg.train['freeze_decoder'] == True):
        print('--------------- freezing decoder network ---------------')
        freeze_mask_fn = masks.create_decoder_mask

    return freeze_mask_fn