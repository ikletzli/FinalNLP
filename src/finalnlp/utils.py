import math

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()
            total_params += params
    return total_params

def count_bit_linear_layer_params(model, BitLinear):
    total_params = 0
    for name, module in model.named_children():
        if isinstance(module, BitLinear):
            total_params += count_parameters(module)
        else:
            # Recursively apply to child modules
            total_params += count_bit_linear_layer_params(module, BitLinear)
    
    return total_params

def count_bytes(model, BitLinear):
    all_params = count_parameters(model)
    bit_linear_params = count_bit_linear_layer_params(model, BitLinear)
    non_bit_params = all_params - bit_linear_params
    bits = bit_linear_params * 2 + non_bit_params * 32
    return math.ceil(bits / 8)