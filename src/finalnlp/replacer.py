from torch import nn

def replace_linears_in_hf(
    model: nn.Module,
    BitLinear: nn.Linear,
):
    """
    Replaces all instances of nn.Linear in the given model with BitLinear15b.

    Args:
        model (nn.Module): The model to modify.

    Returns:
        None
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace the nn.Linear with BitLinear matching in features and and out_features, and add it to the model
            setattr(
                model,
                name,
                BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                ),
            )
        else:
            # Recursively apply to child modules
            replace_linears_in_hf(module, BitLinear)


def replace_linears_in_pytorch_model(
    model: nn.Module,
    BitLinear: nn.Linear,
):
    """
    Replaces all instances of nn.Linear in the given model with BitLinear15b.

    Args:
        model (nn.Module): The model to modify.

    Returns:
        None
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace the nn.Linear with BitLinear matching in features and and out_features, and add it to the model
            setattr(
                model,
                name,
                BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                ),
            )
        else:
            # Recursively apply to child modules
            replace_linears_in_pytorch_model(module, BitLinear)