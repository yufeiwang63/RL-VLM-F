import torch
from torch import nn as nn


import numpy as np

def identity(x):
    return x

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

class CNN(nn.Module):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            batch_norm_conv=False,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=nn.Tanh(),
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_conv = batch_norm_conv
        self.batch_norm_fc = batch_norm_fc
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

        # find output dim of conv_layers by trial and add normalization conv layers
        test_mat = torch.zeros(1, self.input_channels, self.input_width,
                               self.input_height)  # initially the model is on CPU (caller should then move it to GPU if
        for conv_layer in self.conv_layers:
            test_mat = conv_layer(test_mat)
            # import pdb; pdb.set_trace()
            self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        fc_input_size = int(np.prod(test_mat.shape))
        # import pdb; pdb.set_trace()
        # used only for injecting input directly into fc layers
        fc_input_size += added_fc_input_size

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            norm_layer = nn.BatchNorm1d(hidden_size)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            self.fc_norm_layers.append(norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(fc_input_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input): # input is a batch of flattened images.
        h = input
        h = self.apply_forward(h, self.conv_layers, self.conv_norm_layers,
                               use_batch_norm=self.batch_norm_conv)
        # flatten channels for fc layers
        h = h.reshape(h.size(0), -1)
        h = self.apply_forward(h, self.fc_layers, self.fc_norm_layers,
                               use_batch_norm=self.batch_norm_fc)

        output = self.output_activation(self.last_fc(h))
        return output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      use_batch_norm=False):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h
    

if __name__ == "__main__":
    conv_args=dict( # conv layers
        kernel_sizes=[5, 3, 3, 3], 
        n_channels=[16, 32, 64, 128],
        strides=[3, 2, 2, 2],
        output_size=1,
    )
    conv_kwargs=dict(
        hidden_sizes=[], # linear layers after conv
        batch_norm_conv=False,
        batch_norm_fc=False,
    )

    net = CNN(
        **conv_args,
        paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        input_height=256,
        input_width=256,
        input_channels=3,
        # output_size=conv_output_size,
        init_w=1e-3,
        hidden_init=fanin_init,
        **conv_kwargs
    )

    random_input = torch.randn(1, 3, 256, 256)
    out = net(random_input)
    print(out.shape)
    print(out)

