from collections import OrderedDict
import pickle

import torch

def get_padded_tensor(tensor, multiple_number, pad_value=0):
    t_height, t_width = tensor.shape[-2], tensor.shape[-1]
    padded_height = (t_height + multiple_number - 1) // \
                    multiple_number * multiple_number
    padded_width = (t_width + multiple_number - 1) // \
                   multiple_number * multiple_number
    ndim = tensor.ndim
    if ndim == 4:
        padded_tensor = torch.ones([tensor.shape[0], tensor.shape[1], padded_height, padded_width]) * pad_value
        padded_tensor = padded_tensor.type_as(tensor)
        padded_tensor[:, :, :t_height, :t_width] = tensor
    elif ndim == 3:
        padded_tensor = torch.ones([tensor.shape[0], padded_height, padded_width]) * pad_value
        padded_tensor = padded_tensor.type_as(tensor)
        padded_tensor[:, :t_height, :t_width] = tensor
    else:
        raise Exception('Not supported tensor dim: {}'.format(ndim))
    return padded_tensor

def _init_backbone(backbone, model_path, strict):
    state_dict = _load_c2_pickled_weights(model_path)
    state_dict = _rename_weights_for_resnet50(state_dict)
    backbone.load_state_dict(state_dict, strict=strict)
    del state_dict

def _rename_basic_resnet_weights(layer_keys):
    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [k.replace(".w", ".weight") for k in layer_keys]
    layer_keys = [k.replace(".bn", "_bn") for k in layer_keys]
    layer_keys = [k.replace(".b", ".bias") for k in layer_keys]
    layer_keys = [k.replace(".biasranch", ".branch") for k in layer_keys]
    layer_keys = [k.replace(".biaseta", ".beta") for k in layer_keys]
    # Affine-Channel -> BatchNorm enaming
    layer_keys = [k.replace("running.mean", "running_mean") for k in layer_keys]
    layer_keys = [k.replace("running.var", "running_var") for k in layer_keys]
    layer_keys = [k.replace(".beta", ".bias") for k in layer_keys]
    layer_keys = [k.replace(".gamma", ".weight") for k in layer_keys]
    layer_keys = [k.replace("res.conv1_bn", "bn1") for k in layer_keys]
    ## Make torchvision-compatible
    layer_keys = [k.replace("res2.", "layer1.") for k in layer_keys]
    layer_keys = [k.replace("res3.", "layer2.") for k in layer_keys]
    layer_keys = [k.replace("res4.", "layer3.") for k in layer_keys]
    layer_keys = [k.replace("res5.", "layer4.") for k in layer_keys]

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
