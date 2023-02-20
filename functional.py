from __future__ import annotations
from copy import deepcopy

from torch import nn, Tensor


def add_temporal_dimension(tensor: Tensor, time_step: int):
    txb = tensor.size(0)
    t = time_step
    b = txb//t
    shape = [t, b]
    shape.extend(tensor.shape[1:])
    tensor = tensor.view(shape)
    return tensor


def set_spiking_mode(model: nn.Module, spiking_neurons: type[nn.Module] | tuple[type[nn.Module]], time_step: int, input_type='static', reduction='none'):
    def spiking_forward_pre_hook(m: nn.Module, input: tuple[Tensor]):
        x = add_temporal_dimension(input[0], time_step)
        return x

    def spiking_forward_hook(m: nn.Module, input: tuple[Tensor], output: Tensor):
        output = output.flatten(0, 1)
        return output

    def register_hook(m: nn.Module):
        if isinstance(m, spiking_neurons):
            m.register_forward_pre_hook(spiking_forward_pre_hook)
            m.register_forward_hook(spiking_forward_hook)

    model.apply(register_hook)

    def repeat_input_hook(m: nn.Module, input: tuple[Tensor]):
        x = input[0]
        x = x.repeat(time_step, *[1]*(x.dim()-1))
        return x

    def reduce_hook(m: nn.Module, input: tuple[Tensor], output: Tensor):
        output = add_temporal_dimension(output, time_step)
        if reduction == 'mean':
            output = output.mean(0)
        elif reduction == 'sum':
            output = output.sum(0)
        elif reduction == 'none':
            pass
        else:
            raise ValueError(
                f'`reduction` must be one of `mean`, `sum` and `none`, but got {reduction}')
        return output
    if input_type == 'static':
        model.register_forward_pre_hook(repeat_input_hook)
    elif input_type == 'dynamic':
        pass
    model.register_forward_hook(reduce_hook)


def convert_spiking_neuron(module: nn.Module, activation: type[nn.Module] | tuple[type[nn.Module]], spiking_neuron: nn.Module):
    module_output = module
    if isinstance(module, activation):
        module_output = deepcopy(spiking_neuron)
    else:
        for name, child in module.named_children():
            module_output.add_module(
                name, convert_spiking_neuron(child, activation, spiking_neuron)
            )
    del module
    return module_output


if __name__ == "__main__":
    from torchvision.models.resnet import resnet18
    from spikingjelly.activation_based.neuron import LIFNode
    import torch

    # instance an ann model.
    model = resnet18()
    # convert all activations to a specific spiking neuron.
    convert_spiking_neuron(model, nn.ReLU, LIFNode(step_mode='m'))
    # add hook functions which modify inputs or outputs of spiking neurons
    # and also repeat input images `time_step` times and average the output membrane potential.
    set_spiking_mode(model, LIFNode, 4)
    # now, this model is a snn model.
    print(model)
    x = torch.randn(32, 3, 224, 224)
    # we can just feed inputs into model with no difference from ann.
    y = model(x)
    # and also the output of model has the same shape as ann output.
    print(y.shape)  # torch.Size([32, 1000])
