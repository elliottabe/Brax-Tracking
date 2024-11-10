import jax
from custom_brax.custom_losses import PPONetworkParams, PPOSensingNetworkParams
import copy


def create_decoder_mask(params, decoder_name="decoder"):
    """Creates mask where the depth of nodes that contains 'decoder' becomes leaves, and decoder is set to frozen, and the rest to learned."""

    param_mask = copy.deepcopy(params)
    for key in param_mask.policy["params"]:
        if key == decoder_name:
            param_mask.policy["params"][key] = "frozen"
        else:
            param_mask.policy["params"][key] = "learned"

    for key in param_mask.value:
        param_mask.value[key] = "learned"

    return param_mask


def create_sensory_mask(params, layer_name="sensory"):
    """Creates mask where the depth of nodes that contains 'sensory' becomes leaves, and sensory encoder is set to frozen, and the rest to learned."""

    param_mask = copy.deepcopy(params)
    try:
        for key in param_mask.sensory["params"]:
                param_mask.sensory["params"][key] = "frozen" # freeze all sensory params
    except:
        pass #no sensory parmas

    
    for key in param_mask.policy["params"]:
        if layer_name in key:
            param_mask.policy["params"][key] = "frozen"
        else:
            param_mask.policy["params"][key] = "learned"

    for key in param_mask.value:
        param_mask.value[key] = "learned"

    return param_mask

def create_multiple_masks(params, layer_name=["sensory","decoder"]):
    """Creates mask where the depth of nodes that contains 'sensory' or 'decoder' becomes leaves, 
    and sensory encoder is set to frozen, and the rest to learned."""

    param_mask = copy.deepcopy(params)
    if 'sensory' in layer_name:
        try:
            for key in param_mask.sensory["params"]:
                param_mask.sensory["params"][key] = "frozen"
        except:
            pass #no sensory params

    for key in param_mask.policy["params"]:
        if any(lname in key for lname in layer_name):
            param_mask.policy["params"][key] = "frozen"
        else:
            param_mask.policy["params"][key] = "learned"

    for key in param_mask.value:
        param_mask.value[key] = "learned"

    return param_mask


def create_bias_mask(params):
    """Creates boolean mask were any leaves under decoder are set to False."""

    def _mask_fn(path, _):
        def f(key):
            try:
                return key.key
            except:
                return key.name

        # Check if any part of the path contains 'decoder'
        return "frozen" if "bias" in [str(f(part)) for part in path] else "learned"

    # Create mask using tree_map_with_path
    return jax.tree_util.tree_map_with_path(lambda path, _: _mask_fn(path, _), params)