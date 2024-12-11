import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution
from brax.training.networks import MLP
import brax.training.agents.ppo.networks as ppo_networks
from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn

class TuningActivation(nn.Module):
    mu: float = 0.0
    sigma: float = 1.0
    @nn.compact
    def __call__(self, x):
        return jnp.exp(-((x - self.mu) ** 2) / ( self.sigma ** 2))
    
def gaussian_activation(x, mu=0.0, sigma=1.0):
    return jnp.exp(-((x - mu) ** 2) / ( sigma ** 2))

    
class FecoBiasInitializer(jax.nn.initializers.Initializer):
    def __init__(self, thresholds, nneu=2, scale=1.1, random_bias=False ):
        self.thresholds = thresholds
        self.nneu = nneu
        self.scale = scale
        self.random_bias = random_bias

    def __call__(self, key, shape, dtype=jnp.float32):
        na = len(self.thresholds)
        signs = jnp.tile(jnp.array([[-1,1]]).astype(dtype), (na, self.nneu))[:, :self.nneu]
        biases = jnp.tile( 1/ self.scale * self.thresholds.reshape(-1, 1).astype(dtype), self.nneu) * signs  # alternating signs
        if self.random_bias:
            biases = biases * jax.random.uniform(key, biases.shape, minval=-1, maxval=2).astype(dtype) # = -100% to +100%
        return jnp.broadcast_to(biases.flatten(), shape)

class FecoWeightInitializer(jax.nn.initializers.Initializer):
    def __init__(self, angle_std, nneu=2, scale=1.1):
        self.angle_std = angle_std
        self.nneu = nneu
        self.scale = scale

    def __call__(self, key, shape, dtype=jnp.float32):
        k = 1.0 / (self.scale * self.angle_std )
        weights = jnp.zeros(shape, dtype=dtype)
        for i in range(shape[0]):
            for j in range(self.nneu):
                weights = weights.at[i, i*self.nneu+j].set(k[i]*((-1)**j)) # alternating signs
        return weights

    
class CustomFeco(nn.Module):
    '''
    Custom Dense layer with Feco initialization and activation
    '''
    nangles: int
    angle_means: jnp.ndarray
    angle_std: jnp.ndarray
    nneurons: int = 2               # how many neurons per angle
    activation: networks.ActivationFn = nn.relu
    activate_layer: bool = True
    random_bias: bool = False
    scale: float = 1.1

    @nn.compact
    def __call__(self, x, eps=1e-3):
        output_features = self.nneurons * self.nangles
        assert self.angle_means.shape == (self.nangles,)
        assert self.angle_std.shape == (self.nangles,)
        std = self.angle_std + eps * jax.random.uniform(jax.random.PRNGKey(0), shape=self.angle_std.shape) # if there are zeros in angle_std
        weight_init = FecoWeightInitializer(std, nneu=self.nneurons, scale=self.scale )
        bias_init = FecoBiasInitializer(self.angle_means / std, nneu=self.nneurons, random_bias=self.random_bias, scale=self.scale )
        x = nn.Dense(features=output_features, kernel_init=weight_init, bias_init= bias_init, name="feco")(x)
        if self.activate_layer:
            x = self.activation(x)
        return x

class CustomFeco_inv(nn.Module):
    '''
    Custom Dense layer with Feco initialization and activation
    '''
    nangles: int
    angle_means: jnp.ndarray
    angle_std: jnp.ndarray
    nneurons: int = 2               # how many neurons per angle
    activation: networks.ActivationFn = nn.relu
    activate_layer: bool = True
    random_bias: bool = False
    scale: float = 1.1

    @nn.compact
    def __call__(self, x, eps=1e-3):
        output_features = self.nneurons * self.nangles
        assert self.angle_means.shape == (self.nangles,)
        assert self.angle_std.shape == (self.nangles,)
        std = self.angle_std + eps * jax.random.uniform(jax.random.PRNGKey(0), shape=self.angle_std.shape) # if there are zeros in angle_std
        weight_init = FecoWeightInitializer(std, nneu=self.nneurons, scale=self.scale )
        bias_init = FecoBiasInitializer(self.angle_means / std, nneu=self.nneurons, random_bias=self.random_bias, scale=self.scale )
        x = nn.Dense(features=output_features, kernel_init=weight_init, bias_init= bias_init, name="feco")(x)
        if self.activate_layer:
            x = 1 - self.activation(x)
        return x

class VariationalLayer(nn.Module):
    latent_size: int

    @nn.compact
    def __call__(self, x):
        mean_x = nn.Dense(self.latent_size, name="mean")(x)
        logvar_x = nn.Dense(self.latent_size, name="logvar")(x)

        return mean_x, logvar_x


class MLP(nn.Module):
    """MLP with Layer Norm"""

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                x = nn.LayerNorm()(x)
        return x


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


## Intention Networks (including those with integrated sensing)
class IntentionNetwork(nn.Module):
    """Full VAE model, encode -> decode with sampled actions"""

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    task_obs_size: int
    latents: int = 60

    def setup(self):
        self.encoder = MLP(layer_sizes=self.encoder_layers, activate_final=True)
        self.latent = VariationalLayer(latent_size=self.latents)
        self.decoder = MLP(layer_sizes=self.decoder_layers)

    def __call__(self, obs, key):
        _, encoder_rng = jax.random.split(key)
        traj = obs[..., : self.task_obs_size]
        latent_mean, latent_logvar = self.latent(self.encoder(traj))
        z = reparameterize(encoder_rng, latent_mean, latent_logvar)
        action = self.decoder(
            jnp.concatenate([z, obs[..., self.task_obs_size :]], axis=-1)
        )

        return action, {"latent_mean": latent_mean, "latent_logvar": latent_logvar, "z": z}
    

class SensoryIntentionNetwork(nn.Module):
    """Full VAE model, encode -> decode with sampled actions and encoded sensory observations"""

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    sensory_layers: Sequence[int]
    task_obs_size: int
    latents: int = 60

    def setup(self):
        self.encoder = MLP(layer_sizes=self.encoder_layers, activate_final=True)
        self.latent = VariationalLayer(latent_size=self.latents)
        self.decoder = MLP(layer_sizes=self.decoder_layers)
        self.sensory_encoder = MLP(layer_sizes=self.sensory_layers, 
                                   kernel_init = jax.nn.initializers.glorot_normal() )  # no final activation

    def __call__(self, obs, key):
        _, encoder_rng = jax.random.split(key)
        # encode reference trajectory
        traj = obs[..., : self.task_obs_size]
        latent_mean, latent_logvar = self.latent(self.encoder(traj))
        z = reparameterize(encoder_rng, latent_mean, latent_logvar)
        # encode sensory state
        sensez = self.sensory_encoder(obs[..., self.task_obs_size:]) # encoded sensory state
        # decode action
        action = self.decoder(
            jnp.concatenate([z, sensez], axis=-1)
        )

        return action, {"latent_mean": latent_mean, "latent_logvar": latent_logvar, "sense_z": sensez, "z":z }


class SensoryFecoIntentionNetwork(nn.Module):
    """Full VAE model, encode -> decode with sampled actions and encoded sensory observations using Feco Model"""

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    sensory_neurons: int
    task_obs_size: int
    angle_means: jnp.ndarray
    angle_std: jnp.ndarray
    vel_means: jnp.ndarray
    vel_std: jnp.ndarray
    latents: int = 60
    joints: int = 36
    sensory_activation: networks.ActivationFn = nn.relu
    random_bias: bool = False
    body_pos: int = 7
    body_vel: int = 6

    def setup(self):
        assert self.angle_std.shape[0] == self.joints
        self.encoder = MLP(layer_sizes=self.encoder_layers, activate_final=True)
        self.latent = VariationalLayer(latent_size=self.latents)
        self.decoder = MLP(layer_sizes=self.decoder_layers)
        self.sensory_hook = CustomFeco(nangles=self.joints, angle_means=self.vel_means, angle_std=self.vel_std, 
                                          nneurons=self.sensory_neurons, activation=self.sensory_activation, random_bias=self.random_bias)
        self.sensory_claw = CustomFeco(nangles=self.joints, angle_means=self.angle_means, angle_std=self.angle_std, 
                                          nneurons=self.sensory_neurons, activation=self.sensory_activation, random_bias=self.random_bias)
        self.npos = self.joints + self.body_pos
        self.nvel = self.joints + self.body_vel

    def __call__(self, obs, key):
        _, encoder_rng = jax.random.split(key)
        # encode reference trajectory
        traj = obs[..., : self.task_obs_size]
        latent_mean, latent_logvar = self.latent(self.encoder(traj))
        z = reparameterize(encoder_rng, latent_mean, latent_logvar)
        # encode sensory state
        qpos = jax.lax.dynamic_slice_in_dim( obs, self.task_obs_size, self.npos, axis=-1 ) # should be 7 + njoints
        qvel = jax.lax.dynamic_slice_in_dim( obs, self.task_obs_size + self.npos, self.nvel, axis=-1 ) # should be 6 + njoints
        z1 = self.sensory_claw(jax.lax.dynamic_slice_in_dim(qpos, -self.joints, self.joints, axis=-1))
        z2 = self.sensory_hook(jax.lax.dynamic_slice_in_dim(qvel, -self.joints, self.joints, axis=-1))
        sensez = jnp.concatenate([
            jax.lax.dynamic_slice_in_dim(qpos, 0, self.body_pos, axis=-1),
            jax.lax.dynamic_slice_in_dim(qvel, 0,  self.body_vel, axis=-1),
            z1, z2], axis=-1)
        # decode action
        action = self.decoder(
            jnp.concatenate([z, sensez], axis=-1)
        )

        return action, {"latent_mean": latent_mean, "latent_logvar": latent_logvar, "sense_z": sensez, "z":z }



class NoisySensoryIntentionNetwork(nn.Module):
    """Full VAE model, encode -> decode with sampled actions and sensory observations (encoded and sampled) """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    sensory_layers: Sequence[int]
    task_obs_size: int
    latents: int = 60
    slatents: int = 60


    def setup(self):
        self.encoder = MLP(layer_sizes=self.encoder_layers, activate_final=True)
        self.latent = VariationalLayer(latent_size=self.latents)
        self.decoder = MLP(layer_sizes=self.decoder_layers)
        self.sensory_encoder = MLP(layer_sizes=self.sensory_layers, activate_final=True )
        if self.slatents == 0:
            self.slatents = self.latents
        self.sensory_latent = VariationalLayer(latent_size=self.slatents)

    def __call__(self, obs, key):
        sensory_rng, encoder_rng = jax.random.split(key)
        # encode reference trajectory
        traj = obs[..., : self.task_obs_size]
        latent_mean, latent_logvar = self.latent(self.encoder(traj))
        z = reparameterize(encoder_rng, latent_mean, latent_logvar)
        # encode sensory state
        slatent_mean, slatent_logvar = self.sensory_latent( self.sensory_encoder(obs[..., self.task_obs_size:]) ) # encoded sensory state
        sensez = reparameterize(sensory_rng, slatent_mean, slatent_logvar)  # noisy sensory observation
        # decode action
        action = self.decoder(
            jnp.concatenate([z, sensez], axis=-1)
        )

        return action, {"latent_mean": latent_mean, "latent_logvar": latent_logvar, "slatent_mean": slatent_mean, "slatent_logvar": slatent_logvar, "sense_z": sensez, "z":z }


class EncoderDecoderNetwork(nn.Module):
    """encoder and decoder, no sampling in bottleneck. Task obs always first"""

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    task_obs_size: int
    latents: int = 60

    def setup(self):
        self.encoder = MLP(layer_sizes=self.encoder_layers, activate_final=True)
        self.decoder = MLP(layer_sizes=self.decoder_layers)
        self.bottleneck = nn.Dense(self.latents, name="bottleneck")

    def __call__(self, obs, key):
        traj = obs[..., : self.task_obs_size]
        z = self.bottleneck(self.encoder(traj))
        action = self.decoder(
            jnp.concatenate([z, obs[..., self.task_obs_size :]], axis=-1)
        )

        return action, {"z":z}
    

class SensoryEncoderDecoderNetwork(nn.Module):
    """reference encoder, sensory encoder and action decoder, no sampling in bottleneck. Task obs always first"""

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    sensory_layers: Sequence[int]
    task_obs_size: int
    latents: int = 60

    def setup(self):
        self.encoder = MLP(layer_sizes=self.encoder_layers, activate_final=True)
        self.decoder = MLP(layer_sizes=self.decoder_layers)
        self.bottleneck = nn.Dense(self.latents, name="bottleneck")
        self.sensory_encoder = MLP(layer_sizes=self.sensory_layers )  # no final activation

    def __call__(self, obs, key):
        traj = obs[..., : self.task_obs_size]
        z = self.bottleneck(self.encoder(traj))
        # encode sensory state
        sensez = self.sensory_encoder(obs[..., self.task_obs_size:]) # encoded sensory state
        # decode action
        action = self.decoder(
            jnp.concatenate([z, sensez], axis=-1)
        )

        return action, {"sense_z": sensez, "z":z }


class RandomIntentionNetwork(nn.Module):
    """Decoder only, throw out the reference obs and give random intention"""

    decoder_layers: Sequence[int]
    task_obs_size: int
    latents: int = 60

    def setup(self):
        self.decoder = MLP(layer_sizes=self.decoder_layers)

    def __call__(self, obs, key):
        _, intention_rng = jax.random.split(key)

        # Hack to get the right shape
        z = jax.random.normal(intention_rng, obs.shape)
        action = self.decoder(
            jnp.concatenate(
                [z[: self.latents], obs[..., self.task_obs_size :]], axis=-1
            )
        )

        return action, {"z":z}
    

def make_intention_policy(
    param_size: int,
    latent_size: int,
    total_obs_size: int,
    task_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> IntentionNetwork:
    """Creates an intention policy network."""

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size],
        task_obs_size=task_obs_size,
        latents=latent_size,
    )

    def get_action(processor_params, policy_params, obs, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, key=key)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_key),
        apply=get_action,
    )



def make_sensory_intention_policy(
    param_size: int,
    latent_size: int,
    total_obs_size: int,
    task_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    sensory_hidden_layer_sizes: Sequence[int] = (1024, 60),
) -> IntentionNetwork:
    """Creates a sensory+ intention policy network."""

    policy_module = SensoryIntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size],
        sensory_layers=list(sensory_hidden_layer_sizes),
        task_obs_size=task_obs_size,
        latents=latent_size,
    )

    def get_action(processor_params, policy_params, obs, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, key=key)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_z_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_z_key),
        apply=get_action,
    )

def make_sensory_feco_intention_policy(
    param_size: int,
    latent_size: int,
    total_obs_size: int,
    task_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    sensory_hidden_layer_sizes: Sequence[int] = (0),
    sensory_neurons: int = 2,
    joints: int = 36,
    angle_means: jnp.ndarray = jnp.zeros(36),
    angle_std: jnp.ndarray = jnp.ones(36),
    vel_means: jnp.ndarray = jnp.zeros(36),
    vel_std: jnp.ndarray = jnp.ones(36),
    random_bias: bool = False,
    body_pos: int = 7,
    body_vel: int = 6,
    sensory_activation: networks.ActivationFn = nn.relu,
    **kwargs: Any,
) -> IntentionNetwork:
    """Creates a sensory+ intention policy network."""

    policy_module = SensoryFecoIntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size],
        task_obs_size=task_obs_size,
        latents=latent_size,
        sensory_neurons=sensory_neurons,
        angle_means=angle_means,
        angle_std=angle_std,
        vel_means=vel_means,
        vel_std=vel_std,
        joints=joints,
        sensory_activation=sensory_activation,
        random_bias=random_bias,
        body_pos=body_pos,
        body_vel=body_vel,
    )
    assert total_obs_size == task_obs_size + 2 * joints + body_pos + body_vel
    def get_action(processor_params, policy_params, obs, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, key=key)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_z_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_z_key),
        apply=get_action,
    )


def make_noisy_sensory_intention_policy(
    param_size: int,
    latent_size: int,
    total_obs_size: int,
    task_obs_size: int,
    slatents_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    sensory_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> IntentionNetwork:
    """Creates a noisy sensory+ intention policy network."""

    policy_module = NoisySensoryIntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size],
        sensory_layers=list(sensory_hidden_layer_sizes),
        task_obs_size=task_obs_size,
        latents=latent_size,
        slatents=slatents_size,
    )

    def get_action(processor_params, policy_params, obs, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, key=key)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_z_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_z_key),
        apply=get_action,
    )


def make_encoderdecoder_policy(
    param_size: int,
    latent_size: int,
    total_obs_size: int,
    task_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> IntentionNetwork:
    """Creates an encoder-decoder policy network."""

    policy_module = EncoderDecoderNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size],
        task_obs_size=task_obs_size,
        latents=latent_size,
    )

    def apply(processor_params, policy_params, obs, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, key=key)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_key),
        apply=apply,
    )


def make_sensory_encoderdecoder_policy(
    param_size: int,
    latent_size: int,
    total_obs_size: int,
    task_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    sensory_hidden_layer_sizes: Sequence[int] = (1024, 60),
) -> IntentionNetwork:
    """Creates an encoder-decoder policy network with sensory encoder."""

    policy_module = SensoryEncoderDecoderNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size],
        sensory_layers=list(sensory_hidden_layer_sizes),
        task_obs_size=task_obs_size,
        latents=latent_size,
    )

    def get_action(processor_params, policy_params, obs, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, key=key)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_z_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_z_key),
        apply=get_action,
    )


def make_random_intention_policy(
    param_size: int,
    latent_size: int,
    total_obs_size: int,
    task_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> RandomIntentionNetwork:
    """Creates an intention policy network."""

    policy_module = RandomIntentionNetwork(
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size],
        task_obs_size=task_obs_size,
        latents=latent_size,
    )

    def apply(processor_params, policy_params, obs, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, key=key)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_key),
        apply=apply,
    )



############# NEW NETWORKS ####################
### Updated by HG on 2024-11-08

# Actor Networks with Sensory Latents
class IntentionNetwork_wSL(nn.Module):
    """Full VAE model, encode -> decode with sampled actions
    Pre-encoded sensory state is also passed in
    """

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    task_obs_size: int
    latents: int = 60

    def setup(self):
        self.encoder = MLP(layer_sizes=self.encoder_layers, activate_final=True)
        self.latent = VariationalLayer(latent_size=self.latents)
        self.decoder = MLP(layer_sizes=self.decoder_layers)

    def __call__(self, obs, sense_z, key):
        _, encoder_rng = jax.random.split(key)
        traj = obs[..., : self.task_obs_size]
        latent_mean, latent_logvar = self.latent(self.encoder(traj))
        z = reparameterize(encoder_rng, latent_mean, latent_logvar)
        action = self.decoder(
            jnp.concatenate([z, sense_z], axis=-1)
        )

        return action, {"latent_mean": latent_mean, "latent_logvar": latent_logvar, 'sense_z':sense_z}
    

### Sensory Networks Only

class SensoryEncodingNetwork(nn.Module):
    """ turn sensory observations into latents based on the Feco model
    """
    task_obs_size: int
    joints: int
    angle_means: jnp.ndarray 
    angle_std: jnp.ndarray 
    vel_means: jnp.ndarray 
    vel_std: jnp.ndarray 
    nneurons: int = 2
    activation: networks.ActivationFn = nn.relu
    body_pos: int = 7
    body_vel: int = 6
    std_scale: float = 1.1
    #npos: int = -1      # specify how many qpos are there, default is 7 + joints
    #nvel: int = -1      # specify how many qvel are there, default is 6 + joints
    joint_idx: Sequence[int] = dataclasses.field(default_factory=lambda: [0]) # not used yet -> since it can be non-contiguous?
    

    def setup(self):
        self.sensory_hook = CustomFeco(nangles=self.joints, angle_means=self.vel_means, angle_std=self.vel_std, 
                                          nneurons=self.nneurons, activation=self.activation, scale=self.std_scale, )
        self.sensory_claw = CustomFeco(nangles=self.joints, angle_means=self.angle_means, angle_std=self.angle_std, 
                                          nneurons=self.nneurons, activation=self.activation, scale=self.std_scale, )
        #if self.npos < 0:
        self.npos = self.joints + self.body_pos
        #if self.nvel < 0:
        self.nvel = self.joints + self.body_vel

    def __call__(self, obs, key):
        #_, encoder_rng = jax.random.split(key)
        # encode sensory state
        qpos = jax.lax.dynamic_slice_in_dim( obs, self.task_obs_size, self.npos, axis=-1 ) # should at least be 7 + njoints
        qvel = jax.lax.dynamic_slice_in_dim( obs, self.task_obs_size + self.npos, self.nvel, axis=-1 ) # should at least be 6 + njoints
        z1 = self.sensory_claw(jax.lax.dynamic_slice_in_dim(qpos, -self.joints, self.joints, axis=-1))
        z2 = self.sensory_hook(jax.lax.dynamic_slice_in_dim(qvel, -self.joints, self.joints, axis=-1))
        sensez = jnp.concatenate([
            jax.lax.dynamic_slice_in_dim(qpos, 0, self.body_pos, axis=-1),
            jax.lax.dynamic_slice_in_dim(qvel, 0,  self.body_vel, axis=-1),
            z1, z2], axis=-1)

        return sensez


class SensoryEncodingNetwork_v2(nn.Module):
    """ turn sensory observations into latents based on the Feco model
    """
    task_obs_size: int
    joints: int
    angle_means: jnp.ndarray 
    angle_std: jnp.ndarray 
    vel_means: jnp.ndarray 
    vel_std: jnp.ndarray 
    nneurons: int = 2
    activation: networks.ActivationFn = nn.relu  # deprecated
    activation_hook: networks.ActivationFn = nn.tanh
    activation_claw: networks.ActivationFn = nn.relu
    body_pos: int = 7
    body_vel: int = 6
    std_scale: float = 1.1
    #npos: int = -1      # specify how many qpos are there, default is 7 + joints
    #nvel: int = -1      # specify how many qvel are there, default is 6 + joints
    joint_idx: Sequence[int] = dataclasses.field(default_factory=lambda: [0]) # not used yet -> since it can be non-contiguous?
    random_bias: bool = False

    def setup(self):
        self.sensory_hook = CustomFeco(nangles=self.joints, angle_means=self.vel_means, angle_std=self.vel_std, 
                                          nneurons=self.nneurons, activation=self.activation_hook, scale=self.std_scale, )
        self.sensory_claw = CustomFeco(nangles=self.joints, angle_means=self.angle_means, angle_std=self.angle_std, 
                                          nneurons=self.nneurons, activation=self.activation_claw, scale=self.std_scale, 
                                          random_bias=self.random_bias)
        #if self.npos < 0:
        self.npos = self.joints + self.body_pos
        #if self.nvel < 0:
        self.nvel = self.joints + self.body_vel

    def __call__(self, obs, key):
        #_, encoder_rng = jax.random.split(key)
        # encode sensory state
        qpos = jax.lax.dynamic_slice_in_dim( obs, self.task_obs_size, self.npos, axis=-1 ) # should at least be 7 + njoints
        qvel = jax.lax.dynamic_slice_in_dim( obs, self.task_obs_size + self.npos, self.nvel, axis=-1 ) # should at least be 6 + njoints
        z1 = self.sensory_claw(jax.lax.dynamic_slice_in_dim(qpos, -self.joints, self.joints, axis=-1))
        z2 = self.sensory_hook(jax.lax.dynamic_slice_in_dim(qvel, -self.joints, self.joints, axis=-1))
        sensez = jnp.concatenate([
            jax.lax.dynamic_slice_in_dim(qpos, 0, self.body_pos, axis=-1),
            jax.lax.dynamic_slice_in_dim(qvel, 0,  self.body_vel, axis=-1),
            z1, z2], axis=-1)

        return sensez
    


class SensoryEncodingNetwork_v3(nn.Module):
    """ turn sensory observations into latents based on the Feco model
    """
    task_obs_size: int
    joints: int
    angle_means: jnp.ndarray 
    angle_std: jnp.ndarray 
    vel_means: jnp.ndarray 
    vel_std: jnp.ndarray 
    nneurons: int = 2
    activation: networks.ActivationFn = nn.relu  # deprecated
    activation_hook: networks.ActivationFn = nn.tanh
    activation_claw: networks.ActivationFn = nn.relu
    body_pos: int = 7
    body_vel: int = 6
    std_scale: float = 1.1
    #npos: int = -1      # specify how many qpos are there, default is 7 + joints
    #nvel: int = -1      # specify how many qvel are there, default is 6 + joints
    joint_idx: Sequence[int] = dataclasses.field(default_factory=lambda: [0]) # not used yet -> since it can be non-contiguous?
    random_bias: bool = False
    invert_claw: bool = False  # not yet used

    def setup(self):
        self.sensory_hook = CustomFeco(nangles=self.joints, angle_means=self.vel_means, angle_std=self.vel_std, 
                                          nneurons=self.nneurons, activation=self.activation_hook, scale=self.std_scale, )
        n1 = self.nneurons//2
        n2 = self.nneurons - n1
        self.sensory_claw_1 = CustomFeco(nangles=self.joints, angle_means=self.angle_means, angle_std=self.angle_std, 
                                          nneurons=n1, activation=self.activation_claw, scale=self.std_scale, 
                                          random_bias=self.random_bias)
        self.sensory_claw_2 = CustomFeco_inv(nangles=self.joints, angle_means=self.angle_means, angle_std=self.angle_std, 
                                          nneurons=n2, activation=self.activation_claw, scale=self.std_scale, 
                                          random_bias=self.random_bias)
        print(f'Making two claw layers of size {n1}, {n2} ------------------- ')
        #if self.npos < 0:
        self.npos = self.joints + self.body_pos
        #if self.nvel < 0:
        self.nvel = self.joints + self.body_vel
        self.n_sensory_latents = self.joints * 2 * self.nneurons

    def __call__(self, obs, key):
        #_, encoder_rng = jax.random.split(key)
        # encode sensory state
        qpos = jax.lax.dynamic_slice_in_dim( obs, self.task_obs_size, self.npos, axis=-1 ) # should at least be 7 + njoints
        qvel = jax.lax.dynamic_slice_in_dim( obs, self.task_obs_size + self.npos, self.nvel, axis=-1 ) # should at least be 6 + njoints
        opto_inp = jax.lax.dynamic_slice_in_dim( obs, -self.n_sensory_latents, self.n_sensory_latents, axis=-1 ) # 1440!!!! -- too long
        z1_1 = self.sensory_claw_1(jax.lax.dynamic_slice_in_dim(qpos, -self.joints, self.joints, axis=-1))
        z1_2 = self.sensory_claw_2(jax.lax.dynamic_slice_in_dim(qpos, -self.joints, self.joints, axis=-1))
        z2 = self.sensory_hook(jax.lax.dynamic_slice_in_dim(qvel, -self.joints, self.joints, axis=-1))
        #opto_inp = jnp.nan_to_num(opto_inp, nan=0.0)
        encoded_neu = jnp.concatenate([z1_1, z1_2, z2], axis=-1) + opto_inp
        sensez = jnp.concatenate([
            jax.lax.dynamic_slice_in_dim(qpos, 0, self.body_pos, axis=-1),
            jax.lax.dynamic_slice_in_dim(qvel, 0,  self.body_vel, axis=-1),
            encoded_neu], axis=-1)

        return sensez
    
###### NOT USED YET
'''
class SensoryEncodingNetwork_v2(nn.Module):
    """ turn sensory observations into latents based on the Feco model
        pre-filtered observations
    """
    task_obs_size: int
    joints: int
    angle_means: jnp.ndarray 
    angle_std: jnp.ndarray 
    vel_means: jnp.ndarray 
    vel_std: jnp.ndarray 
    nneurons: int = 2
    activation: networks.ActivationFn = nn.relu
    body_pos: int = 7
    body_vel: int = 6
    npos: int = -1      # specify how many qpos are there, default is 7 + joints
    nvel: int = -1      # specify how many qvel are there, default is 6 + joints
    joint_idx: Sequence[int] = [0] # not used yet -> since it can be non-contiguous?

    def setup(self):
        self.sensory_hook = CustomFeco(nangles=self.joints, angle_means=self.vel_means, angle_std=self.vel_std, 
                                          nneurons=self.nneurons, activation=self.activation,)
        self.sensory_claw = CustomFeco(nangles=self.joints, angle_means=self.angle_means, angle_std=self.angle_std, 
                                          nneurons=self.nneurons, activation=self.activation)
        if self.npos < 0:
            self.npos = self.joints + self.body_pos
        if self.nvel < 0:
            self.nvel = self.joints + self.body_vel

    def __call__(self, obs_filtered, key):
        #_, encoder_rng = jax.random.split(key)
        # encode sensory state
        # obs = (body_pos, joints_pos, body_vel, joints_vel)
        bpos, bvel, jpos, jvel = obs_filtered
        sensez = jnp.concatenate([
            bpos, bvel, 
            self.sensory_claw(jpos), self.sensory_hook(jvel)],
            axis=-1)

        return sensez

'''


###### the apply function needs different arguments = obs, sense_z, key
def make_intention_w_sl_policy(
param_size: int,        # dimensionality of the action space
    latent_size: int,       # dimensionality of the intention latent space
    total_obs_size: int,    # dimensionality of the observation space
    task_obs_size: int,     # dimensionality of the task observation space (first part of the observation - e.g. reference trajectory)
    sense_latent_size: int = 85,     # needs to be 7 + 6 + 2 * joints
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> IntentionNetwork:
    """Creates an intention policy network that also uses sensory latents."""

    policy_module = IntentionNetwork_wSL(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size],
        task_obs_size=task_obs_size,
        latents=latent_size,
    )

    def get_action(processor_params, policy_params, obs, sense_z, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, sense_z=sense_z, key=key)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_sense_z = jnp.zeros((1, sense_latent_size))
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_sense_z, dummy_key),
        apply=get_action,
    )



def make_sensory_encoding_network(
    total_obs_size: int,
    task_obs_size: int,
    joints: int = 36,
    joint_idx: Sequence[int] = range(36),  # not used yet
    nneurons: int = 2,          # how many neurons per angle
    angle_means: jnp.ndarray = jnp.zeros(36)*.5,
    angle_std: jnp.ndarray = jnp.ones(36)*.5,
    vel_means: jnp.ndarray = jnp.zeros(36),
    vel_std: jnp.ndarray = jnp.ones(36),
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    std_scale = 1.1,
    invert_claw: bool = False,
    **kwargs,
) -> networks.FeedForwardNetwork:
    """
    Minimal function to create a sensory encoding network that takes in observations and returns sensory latents.
    Based on Feco Hook and Claw neurons. Defaults:Activation is ReLU, Free joint has 7 qpos, 6 qvel.
    """
    if invert_claw:
        sensory_module = SensoryEncodingNetwork_v3(  # changed to v2
            task_obs_size=task_obs_size,
            joints=joints,
            joint_idx=joint_idx,
            angle_means=angle_means,
            angle_std=angle_std,
            vel_means=vel_means,
            vel_std=vel_std,
            nneurons=nneurons,
            std_scale=std_scale,
            **kwargs
        )
    else:
        sensory_module = SensoryEncodingNetwork_v2(  # changed to v2
            task_obs_size=task_obs_size,
            joints=joints,
            joint_idx=joint_idx,
            angle_means=angle_means,
            angle_std=angle_std,
            vel_means=vel_means,
            vel_std=vel_std,
            nneurons=nneurons,
            std_scale=std_scale,
            **kwargs
        )

    def encode_senses(processor_params, sensory_params, obs, key):
        obs = preprocess_observations_fn(obs, processor_params)
        return sensory_module.apply(sensory_params, obs=obs, key=key)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: sensory_module.init(key, dummy_total_obs, dummy_key),
        apply=encode_senses,
    )


def make_intention_and_sensory_networks(
    param_size: int,        # dimensionality of the action space
    latent_size: int,       # dimensionality of the intention latent space
    total_obs_size: int,    # dimensionality of the observation space
    task_obs_size: int,     # dimensionality of the task observation space (first part of the observation - e.g. reference trajectory)
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    joints: int = 36,
    joint_idx: Sequence[int] = range(36),   # not used yet (currently, last k joints are encoded)
    nneurons: int = 2,      # how many neurons per angle
    angle_means: jnp.ndarray = jnp.zeros(36)*.5,
    angle_std: jnp.ndarray = jnp.ones(36)*.5,
    vel_means: jnp.ndarray = jnp.zeros(36),
    vel_std: jnp.ndarray = jnp.ones(36),
    std_scale = 1.1,
    **kwargs,
    ) -> Tuple[IntentionNetwork, networks.FeedForwardNetwork]:
    """Creates an intention policy network and a sensory encoding network."""
    sense_latent_size = 7 + 6 + 2 * joints * nneurons

    intention_net = make_intention_w_sl_policy( param_size=param_size, latent_size=latent_size, total_obs_size=total_obs_size, task_obs_size=task_obs_size,
                                               sense_latent_size=sense_latent_size, encoder_hidden_layer_sizes=encoder_hidden_layer_sizes, decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
                                               preprocess_observations_fn=preprocess_observations_fn)
    sensory_net = make_sensory_encoding_network(total_obs_size=total_obs_size, task_obs_size=task_obs_size, 
                                                joints=joints, joint_idx=joint_idx, nneurons=nneurons, std_scale=std_scale,
                                               angle_means=angle_means, angle_std=angle_std, vel_means=vel_means, vel_std=vel_std, 
                                               preprocess_observations_fn=preprocess_observations_fn,
                                               **kwargs)
    return (intention_net, sensory_net)