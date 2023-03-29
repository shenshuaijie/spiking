# spiking

## Spiking Neural Networks: functional.py
A simple method that converts ANN models into SNN models.

### 1. Theory

Although there is a temporal dimension in Spiking Neural Networks, most of the modules used in ANNs don't have to deal with the temporal dimension. For instance, modules such as `Linear`, `BatchNorm`, and `Convolution` perform computations independently across different time steps. Therefore, we can concatenate features of different time steps along the batch dimension. For example, for feature maps with shape `(N,C,H,W)` in a training with T time steps, we can reshape feature maps from shape `(T,N,C,H,W)` to shape `(T*N,C,H,W)`, and feed them into time-independent modules. We can then restore the feature maps before feeding them into time-dependent modules.

Considering that most modules are time-independent, we only need to handle the inputs and outputs of time-dependent modules. To achieve this, we can add a `forward_pre_hook` function and a `forward_hook` function to time-dependent modules, such as the LIF module. The forward_pre_hook function reshapes the inputs from shape `(T*N,C,H,W)` to shape `(T,N,C,H,W)` before calling the module's forward function. The forward_hook function then recovers the outputs from shape `(T,N,C,H,W)` to shape `(T*N,C,H,W)` after calling the module's forward function.


### 2. usage

There is an example provided in `functional.py`.

Here are the steps you can follow:

1. Instantiate an ANN model. 
2. Call `convert_spiking_neurons` to convert all activations of the ANN, such as `ReLU`, into spiking neurons. If you want to implement a multiply-free interface (MFI), you need to carefully design your model and ensure that all modules that perform multiplication are directly connected to spiking neurons.
3. Call `set_spiking_mode`, which adds hook functions to `spiking_neurons` and the entire model to handle different input formats (`static` or `dynamic`) and convert the output/prediction into the appropriate format.
4. Train the model like a normal SNN model!

## Event Representation: dvs2frame.py
There are some event representation methods.

Given a set of $N$ input events $\{(x_i,y_i,t_i,p_i)\}_{i=0,\dots,N-1}$, where $(x_i, y_i)$ represent coordinates of event `i`, $t_i$ represent the timestamp when event `i` occurs, and $p_i$ represent the polarity of event `i`, `-1` or `+1`.
### 1. Stack Based on Time
SBN stacks all event in a certain time window, `ΔT`, and generate a frame with shape `(1, H, W)`.

// TODO