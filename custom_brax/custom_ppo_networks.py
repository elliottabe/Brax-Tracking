"""
Custom network definitions.
This is needed because we need to route the observations 
to proper places in the network in the case of the VAE (CoMic, Hasenclever 2020)
"""

import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution

from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn

from custom_brax import custom_networks


@flax.struct.dataclass
class PPOImitationNetworks:
    policy_network: custom_networks.IntentionNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPOImitationNetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            logits, _ = policy_network.apply(*params, observations, key_network)

            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}

            # Sample action based on logits (mean and logvar)
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )

            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)

            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "logits": logits,
            }

        return policy

    return make_policy


# intention policy
def make_intention_ppo_networks(
    observation_size: int,
    task_obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPOImitationNetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = custom_networks.make_intention_policy(
        parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        total_obs_size=observation_size,
        task_obs_size=task_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_encoderdecoder_ppo_networks(
    observation_size: int,
    task_obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPOImitationNetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = custom_networks.make_encoderdecoder_policy(
        parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        total_obs_size=observation_size,
        task_obs_size=task_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )




# random_decoder policy
def make_randomdecoder_ppo_networks(
    observation_size: int,
    task_obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPOImitationNetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = custom_networks.make_random_intention_policy(
        parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        total_obs_size=observation_size,
        task_obs_size=task_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
