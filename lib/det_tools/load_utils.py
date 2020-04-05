from collections import OrderedDict
import pickle

import torch

def _init_backbone(backbone, model_path):
    state_dict = _load_c2_pickled_weights(model_path)
    state_dict = _rename_weights_for_resnet50(state_dict)
    del state_dict['stem.conv1.bias']
    backbone.load_state_dict(state_dict, strict=False)
    own_keys = set(backbone.state_dict().keys())
    del state_dict
    return 0

def _rename_basic_resnet_weights(layer_keys):
    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [k.replace(".w", ".weight") for k in layer_keys]
    layer_keys = [k.replace(".bn", "_bn") for k in layer_keys]
    layer_keys = [k.replace(".b", ".bias") for k in layer_keys]
    layer_keys = [k.replace(".biasranch", ".branch") for k in layer_keys]
    layer_keys = [k.replace(".biaseta", ".beta") for k in layer_keys]
    layer_keys = [k.replace("res", "res_blocks") for k in layer_keys]
    layer_keys = [k.replace("conv1.bias", "stem.conv1.bias") for k in layer_keys]
    layer_keys = [k.replace("conv1.weight", "stem.conv1.weight") for k in layer_keys]
    # Affine-Channel -> BatchNorm enaming
    layer_keys = [k.replace("running.mean", "running_mean") for k in layer_keys]
    layer_keys = [k.replace("running.var", "running_var") for k in layer_keys]
    layer_keys = [k.replace(".beta", ".bias") for k in layer_keys]
    layer_keys = [k.replace(".gamma", ".weight") for k in layer_keys]
    layer_keys = [k.replace("_bn.scale", "_bn.weight") for k in layer_keys]
    layer_keys = [k.replace("res_blocks.conv1_bn.", "stem.bn1.") for k in layer_keys]
    # Make torchvision-compatible
    layer_keys = [k.replace("res_blocks2.", "layer0.") for k in layer_keys]
    layer_keys = [k.replace("res_blocks3.", "layer1.") for k in layer_keys]
    layer_keys = [k.replace("res_blocks4.", "layer2.") for k in layer_keys]
    layer_keys = [k.replace("res_blocks5.", "layer3.") for k in layer_keys]

    layer_keys = [k.replace(".branch2a.", ".conv1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2a_bn.", ".bn1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b.", ".conv2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b_bn.", ".bn2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c.", ".conv3.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c_bn.", ".bn3.") for k in layer_keys]
    layer_keys = [k.replace(".branch1.", ".downsample.0.") for k in layer_keys]
    layer_keys = [k.replace(".branch1_bn.", ".downsample.1.") for k in layer_keys]
    return layer_keys

def _rename_weights_for_resnet50(weights):
    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())
    layer_keys = _rename_basic_resnet_weights(layer_keys)
    key_map = {k: v for k, v in zip(original_keys, layer_keys)}
    new_weights = OrderedDict()
    for k in original_keys:
        v = weights[k]
        if "_momentum" in k:
            continue
        if "fc1000" in k:
            continue
        w = torch.from_numpy(v)
        new_weights[key_map[k]] = w
    return new_weights

def _load_c2_pickled_weights(file_path):
    with open(file_path, "rb") as f:
        if torch._six.PY3:
            data = pickle.load(f, encoding="latin1")
        else:
            data = pickle.load(f)
    if "blobs" in data:
        weights = data["blobs"]
    else:
        weights = data
    return weights
