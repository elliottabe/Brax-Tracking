import os
import subprocess as sp

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
import functools
import jax
# jax.config.update("jax_enable_x64", True)
n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")
from typing import Dict
import wandb
from brax import envs
import signal
import sys
import pickle
import warnings
import hydra
from brax.io import model
from omegaconf import DictConfig, OmegaConf
from brax.training.agents.ppo import networks as ppo_networks
from custom_brax import custom_ppo as ppo
from custom_brax import custom_wrappers
from custom_brax import custom_ppo_networks
from custom_brax import network_masks as masks
from orbax import checkpoint as ocp
from flax.training import orbax_utils
# from envs.rodent import RodentSingleClip
from preprocessing.mjx_preprocess import process_clip_to_train
from envs.Fly_Env_Brax2 import FlyTracking, FlyMultiClipTracking, FlyRunSim
from utils.utils import *
from utils.fly_logging import log_eval_rollout

warnings.filterwarnings("ignore", category=DeprecationWarning)

from absl import app
from absl import flags

FLAGS = flags.FLAGS

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)
envs.register_environment("fly_freejnt_clip", FlyTracking)
envs.register_environment("fly_freejnt_multiclip", FlyMultiClipTracking)
envs.register_environment("fly_run_policy", FlyRunSim)


# Global Boolean variable that indicates that a signal has been received
interrupted = False

# Global Boolean variable that indicates then natural end of the computations
converged = False

# Definition of the signal handler. All it does is flip the 'interrupted' variable
def signal_handler(signum, frame):
    global interrupted
    interrupted = True

def get_gpu_memory():
    """Get total GPU memory with nvidia-smi

    Returns:
        list: total memory in MB for each GPU
    """
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    return [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

def closest_power_of_two(x):
    # Start with the largest power of 2 less than or equal to x
    power = 1
    while power * 2 <= x:
        power *= 2
    return power


# Register the signal handler
signal.signal(signal.SIGTERM, signal_handler)
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ##### Scale number of envs based on total memory per gpu #####
    tot_mem = get_gpu_memory()[0]
    num_envs = min([cfg.train.num_envs,int(closest_power_of_two(tot_mem/21.4))]) #21.4
    cfg.train.num_envs = cfg.num_gpus*num_envs
    if n_gpus != cfg.num_gpus:
        cfg.num_gpus = n_gpus
        cfg.train.num_envs = n_gpus*num_envs
    print(f'GPUs:{n_gpus}, tot_mem:{tot_mem}Mb, num_envs:{num_envs}, total_envs:{cfg.train.num_envs}')
    
    print('run_id:', cfg.run_id)
    if ('load_jobid' in cfg) and (cfg['load_jobid'] is not None) and (cfg['load_jobid'] !=''):
        run_id = cfg.load_jobid
        load_cfg_path = Path(cfg.paths.base_dir) / f'run_id={run_id}/logs/run_config.yaml'
        cfg = OmegaConf.load(load_cfg_path)
        continue_training = True
    else:
        run_id = cfg.run_id
        continue_training = False
    # Create paths if they don't exist and Path objects
    for k in cfg.paths.keys():
        if k != "user":
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
    env_cfg = cfg.dataset
    env_args = cfg.dataset.env_args
    # reference_path = cfg.paths.data_dir / f"clips/{env_cfg['clip_idx']}.p" all_clips_batch_interp.p
    reference_path = cfg.paths.data_dir / f"clips/{env_cfg['clip_idx']}"
    reference_path.parent.mkdir(parents=True, exist_ok=True)

    #### TODO: Need to handle this better
    with open(reference_path, "rb") as file:
        # Use pickle.load() to load the data from the file
        reference_clip = pickle.load(file)
    
    nclips=reference_clip.position.shape[0]
    # inds = jax.random.randint(jax.random.PRNGKey(0), (nclips,), 0, len(reference_clip.position))
    # reference_clip =reference_clip.replace(
    #     position=reference_clip.position[inds],
    #     quaternion=reference_clip.quaternion[inds],
    #     joints=reference_clip.joints[inds],
    #     body_positions=reference_clip.body_positions[inds],
    #     velocity=reference_clip.velocity[inds],
    #     joints_velocity=reference_clip.joints_velocity[inds],
    #     angular_velocity=reference_clip.angular_velocity[inds],
    #     body_quaternions=reference_clip.body_quaternions[inds],
    # )
    all_ref_clip = {}
    all_ref_clip['position'] = reference_clip.position
    all_ref_clip['quaternion'] = reference_clip.quaternion
    all_ref_clip['joints'] = reference_clip.joints
    all_ref_clip['body_positions'] =reference_clip.body_positions
    all_ref_clip['velocity'] = reference_clip.velocity
    all_ref_clip['joints_velocity'] = reference_clip.joints_velocity
    all_ref_clip['angular_velocity'] = reference_clip.angular_velocity
    all_ref_clip['body_quaternions'] = reference_clip.body_quaternions
    reference_clip = all_ref_clip
    global EVAL_STEPS
    EVAL_STEPS = 0
    ########## Handling requeuing ##########
    if ('restore_checkpoint' in cfg) and (cfg['jobid'] is not None) and (cfg['restore_checkpoint'] !=''):
        restore_checkpoint = cfg.restore_checkpoint
        print('Loading from ckpt:', restore_checkpoint)
    else: 
        try: #
            # Try to recover a state file with the relevant variables stored
            # from previous stop if any
            model_path = cfg.paths.ckpt_dir
            if any(model_path.iterdir()):
                from natsort import natsorted
                ##### Get all the checkpoint files #####
                ckpt_files = natsorted([Path(f.path) for f in os.scandir(model_path) if f.is_dir()])
                ##### Get the latest checkpoint #####
                max_ckpt = ckpt_files[-1]
                EVAL_STEPS = int(max_ckpt.stem)
                if EVAL_STEPS > 0 :
                    restore_checkpoint = max_ckpt
                    cfg = OmegaConf.load(cfg.paths.log_dir / "run_config.yaml")
                    cfg.dataset = cfg.dataset
                    cfg.dataset.env_args = cfg.dataset.env_args
                    env_cfg = cfg.dataset
                    env_args = cfg.dataset.env_args
                    continue_training = True
                    print(f'Loading: {max_ckpt}')
                else:
                    raise ValueError('Model path does not exist. Starting from scratch.')
            else:
                raise ValueError('Model path does not exist. Starting from scratch.')
        except (ValueError):
            # Otherwise bootstrap (start from scratch)
            print('Model path does not exist. Starting from scratch.')
            restore_checkpoint = None
            
    if  ('network_type' in cfg.train) and (cfg.train['network_type'] is not None) and ('encoderdecoder' in cfg.train['network_type']):
        network_type = custom_ppo_networks.make_encoderdecoder_ppo_networks
    else: 
        network_type = custom_ppo_networks.make_intention_ppo_networks

    while not interrupted and not converged:
        # Init env
        env = envs.get_environment(
            cfg.train.env_name,
            reference_clip=reference_clip,
            **env_args,
        )

        episode_length = (env_args.clip_length - 50 - env_args.ref_len) * env._steps_for_cur_frame
        print(f"episode_length {episode_length}")

        options = ocp.CheckpointManagerOptions(save_interval_steps=1)
        ckpt_mgr = ocp.CheckpointManager(
            cfg.paths.ckpt_dir,
            item_names=("normalizer_params", "params", "env_steps"),
            options=options,
        )
        train_fn = functools.partial(
            ppo.train,
            num_envs=cfg.train["num_envs"],
            num_timesteps=100_000, #cfg.train["num_timesteps"],
            num_evals=1, #int(cfg.train["num_timesteps"] / cfg.train["eval_every"]),
            num_resets_per_eval=cfg.train['num_resets_per_eval'],
            reward_scaling=cfg.train['reward_scaling'],
            episode_length=episode_length,
            normalize_observations=True,
            action_repeat=cfg.train['action_repeat'],
            clipping_epsilon=cfg.train["clipping_epsilon"],
            unroll_length=cfg.train['unroll_length'],
            num_minibatches=cfg.train["num_minibatches"],
            num_updates_per_batch=cfg.train["num_updates_per_batch"],
            discounting=cfg.train['discounting'],
            learning_rate=cfg.train["learning_rate"],
            kl_weight=cfg.train["kl_weight"],
            entropy_cost=cfg.train['entropy_cost'],
            batch_size=cfg.train["batch_size"],
            seed=cfg.train['seed'],
            network_factory=functools.partial(
                network_type,
                encoder_hidden_layer_sizes=cfg.train['encoder_hidden_layer_sizes'],
                decoder_hidden_layer_sizes=cfg.train['decoder_hidden_layer_sizes'],
                value_hidden_layer_sizes=cfg.train['value_hidden_layer_sizes'],
            ),
            checkpoint_network_factory=functools.partial(
                    custom_ppo_networks.make_intention_ppo_networks,
                    intention_latent_size=60,
                    encoder_hidden_layer_sizes=cfg.train.ckpt_net['encoder_hidden_layer_sizes'],
                    decoder_hidden_layer_sizes=cfg.train.ckpt_net['decoder_hidden_layer_sizes'],
                    value_hidden_layer_sizes=cfg.train.ckpt_net['value_hidden_layer_sizes'],
                ),
            checkpoint_path=restore_checkpoint,
            freeze_mask_fn=None if (cfg.train['freeze_decoder'] == False) else masks.create_decoder_mask,
            continue_training=continue_training,
            custom_wrap=True,  # custom wrappers to handle infos
            kl_loss=cfg.train['kl_loss'],
        )


        # run = wandb.init(
        #     dir=cfg.paths.log_dir,
        #     project=cfg.train.wandb_project,
        #     config=OmegaConf.to_container(cfg),
        #     notes=cfg.train.note,
        #     id=f'{run_id}',
        #     resume="allow",
        # )

        # wandb.run.name = (
        #     f"{env_cfg['name']}_{cfg.train['task_name']}_{cfg.train['algo_name']}_{run_id}"
        # )


        # def wandb_progress(num_steps, metrics):
        #     num_steps=int(num_steps)
        #     metrics["num_steps"] = num_steps
        #     wandb.log(metrics, commit=False)

        # Wrap the env in the brax autoreset and episode wrappers
        # rollout_env = custom_wrappers.RenderRolloutWrapperTracking(env)
        # define the jit reset/step functions
        # jit_reset = jax.jit(rollout_env.reset)
        # jit_step = jax.jit(rollout_env.step)

        def policy_params_fn(num_steps, make_policy, params, policy_params_fn_key, model_path=model_path):
            def block_until_ready_all(params):
                """Creates boolean mask were any leaves under decoder are set to False."""
                return params.block_until_ready()
            jax.tree.map(block_until_ready_all,params[1].policy)
            jax.profiler.save_device_memory_profile(f"memory_fullpipeline_{nclips}2.prof")
            print(f'num_steps: {num_steps}')
        #     global EVAL_STEPS
        #     EVAL_STEPS = EVAL_STEPS + 1
        #     print(f'Eval Step: {EVAL_STEPS}, num_steps: {num_steps}')
        #     # ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        #     # save_args = orbax_utils.save_args_from_target(params)
        #     # path = model_path / f'{EVAL_STEPS:03d}'
        #     # os.makedirs(path, exist_ok=True)
        #     # ckptr.save(path, params, force=True, save_args=save_args)
        #     policy_params = (params[0],params[1].policy)
        #     Env_steps = params[2]
        #     jit_inference_fn = jax.jit(make_policy(policy_params, deterministic=True))
        #     reset_rng, act_rng = jax.random.split(policy_params_fn_key)

        #     state = jit_reset(reset_rng)

        #     rollout = [state]
        #     # rollout_len = env_args["clip_length"]*int(rollout_env._steps_for_cur_frame)
        #     rollout_len = 500
        #     for i in range(rollout_len):
        #         _, act_rng = jax.random.split(act_rng)
        #         obs = state.obs
        #         ctrl, extras = jit_inference_fn(obs, act_rng)
        #         state = jit_step(state, ctrl)
        #         rollout.append(state)
        #     ##### Log the rollout to wandb #####
        #     log_eval_rollout(cfg,rollout,state,env,reference_clip,model_path,EVAL_STEPS)

        if not (cfg.paths.log_dir / "run_config.yaml").exists():
            OmegaConf.save(cfg, cfg.paths.log_dir / "run_config.yaml")
        print(OmegaConf.to_yaml(cfg))
        
        make_inference_fn, params, _ = train_fn(
            environment=env, 
            # progress_fn=wandb_progress, 
            policy_params_fn=policy_params_fn,
            checkpoint_manager=ckpt_mgr,
        )

        def block_until_ready_all(params):
            """Creates boolean mask were any leaves under decoder are set to False."""
            return params.block_until_ready()
        jax.tree.map(block_until_ready_all,params[1].policy)
        jax.profiler.save_device_memory_profile(f"memory_fullpipeline_{nclips}2.prof")
        # final_save_path = Path(f"{model_path}")/f'brax_ppo_{cfg.dataset.name}_run_finished'
        # model.save_params(final_save_path, params)
        # print(f'Run finished. Model saved to {final_save_path}')
        # ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        # save_args = orbax_utils.save_args_from_target(params)
        # path = model_path / f'{EVAL_STEPS:03d}'
        # os.makedirs(path, exist_ok=True)
        # ckptr.save(path, params, force=True, save_args=save_args)
        break
    # Save current state 
    # if interrupted:
    #     model.save_params(f"{model_path}/{num_steps}", params)
    #     sys.exit(99)
    # sys.exit(0)
    print('Finished')

if __name__ == "__main__":
    main()
