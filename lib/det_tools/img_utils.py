import torch

def pad_tensor_to_multiple_number(tensor, multiple_number, pad_value=0):
    t_height, t_width = tensor.shape[-2], tensor.shape[-1]
    padded_height = (t_height + multiple_number - 1) // \
                    multiple_number * multiple_number
    padded_width = (t_width + multiple_number - 1) // \
                   multiple_number * multiple_number
    ndim = tensor.ndim
    if ndim == 4:
        padded_tensor = torch.ones([tensor.shape[0], tensor.shape[1], padded_height, padded_width]) * pad_value
        padded_tensor = padded_tensor.cuda(tensor.device)
        padded_tensor[:, :, :t_height, :t_width] = tensor
    elif ndim == 3:
        padded_tensor = torch.ones([tensor.shape[0], padded_height, padded_width]) * pad_value
        padded_tensor = padded_tensor.cuda(tensor.device)
        padded_tensor[:, :t_height, :t_width] = tensor
    else:
        raise Exception('Not supported tensor dim: {}'.format(ndim))
    return padded_tensor
