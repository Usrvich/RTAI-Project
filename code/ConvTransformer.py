import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import toeplitz

from general_functions import compute_bounds
class ConvTransformer(nn.Module):
    def __init__(self, prev_layer, cur_layer, prev_transformer=None):
        super(ConvTransformer, self).__init__()

        self.prev_layer = prev_layer
        self.cur_layer = cur_layer

        self.kernel = cur_layer.weight
        self.padding = cur_layer.padding
        self.stride = cur_layer.stride
        in_channels = torch.tensor([self.cur_layer.in_channels])[0]
        out_channels = torch.tensor([self.cur_layer.out_channels])[0]
        # self.shape_in = prev_layer.bias.shape[1]



        # input_size = self.shape_in[2:]
        # kernel_size = self.kernel.size()[2:]
        # output_size = torch.tensor([(input_size[i] - kernel_size[i] + 2 * self.padding[i]) // self.stride[i] + 1 for i in [0,1]])
        # self.shape_out = torch.cat([torch.tensor([1, out_channels]), output_size])
        # # convert the conv layer to a fully connected layer
        # input_prox = torch.eye(in_channels*input_size[0]*input_size[1])
        # input_prox = input_prox.reshape(1, in_channels, input_size[0], input_size[1], in_channels*input_size[0]*input_size[1])
        # self.weights = torch.empty(1, out_channels, output_size[0], output_size[1], in_channels*input_size[0]*input_size[1])
        # for i in range(in_channels*input_size[0]*input_size[1]):
        #     self.weights[:,:,:,:,i] = torch.nn.functional.conv2d(input_prox[:,:,:,:,i], self.kernel, stride=self.stride, padding=self.padding)
        #
        # self.bias = cur_layer.bias.reshape(1,-1,1).repeat(1,1,output_size[0]*output_size[1])
        # self.bias = self.bias.reshape(1, out_channels*output_size[0]*output_size[1], 1) # dim: 1 x output_flat x 1
        self.bias_lower = None
        self.bias_upper = None

        self.weights_upper = None
        self.weights_lower = None

        # self.bias_upper = self.bias.detach()
        # self.bias_lower = self.bias.detach()
        self.prev_transformer = prev_transformer

        self.shape_in = None
        self.shape_out = None

        self.lower_bound = None
        self.upper_bound = None






    def forward(self, lower_bound_prev, upper_bound_prev):

        self.shape_in = self.prev_transformer.shape_out.detach()

        self.kernel = self.cur_layer.weight

        self.padding = self.cur_layer.padding
        self.stride = self.cur_layer.stride

        in_channels = torch.tensor([self.cur_layer.in_channels])[0]
        out_channels = torch.tensor([self.cur_layer.out_channels])[0]
        input_size = self.shape_in[2:]
        kernel_size = self.kernel.size()[2:]
        output_size = torch.tensor([(input_size[i] - kernel_size[i] + 2 * self.padding[i]) // self.stride[i] + 1 for i in [0,1]])

        # convert the conv layer to a fully connected layer
        # input_prox = torch.eye(in_channels*input_size[0]*input_size[1])
        # input_prox = input_prox.reshape(1, in_channels, input_size[0], input_size[1], in_channels*input_size[0]*input_size[1])
        weights = torch.empty(1, out_channels, output_size[0], output_size[1],
                              in_channels * input_size[0] * input_size[1])
        for i in range(in_channels*input_size[0]*input_size[1]):
            unit_vector = torch.zeros(in_channels * input_size[0] * input_size[1])
            unit_vector[i] = 1
            unit_vector = unit_vector.reshape(1, in_channels, input_size[0], input_size[1])
            weights[0, :, :, :, i] = torch.nn.functional.conv2d(unit_vector, self.kernel, stride=self.stride, padding=self.padding)
            # weights[:,:,:,:,i] = torch.nn.functional.conv2d(input_prox[:,:,:,:,i], self.kernel, stride=self.stride, padding=self.padding)
        weights = weights.reshape(1, out_channels*output_size[0]*output_size[1], in_channels*input_size[0]*input_size[1]) # dim: 1 x output_flat x input_flat

        bias = self.cur_layer.bias.reshape(1,-1,1).repeat(1,1,output_size[0]*output_size[1])
        bias = bias.reshape(1, out_channels*output_size[0]*output_size[1], 1) # dim: 1 x output_flat x 1

        lower_bound, upper_bound = compute_bounds(weights, weights, bias, bias, lower_bound_prev, upper_bound_prev)


        self.weights_lower = weights
        self.weights_upper = weights
        self.bias_lower = bias
        self.bias_upper = bias
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.shape_out = torch.cat([torch.tensor([1, out_channels]), output_size])

        # print("lower bound grad for conv: ", lower_bound.grad)
        # print("upper bound grad for conv: ", upper_bound.grad)

        return lower_bound, upper_bound

    #
    # # use the circulant matrix and matrix multiplication to represent the convolution transformation
    # def forward(self, lower_bound_prev, upper_bound_prev):
    #     # use doubly block circulant matrix to compute the bounds
    #     self.shape_in = self.prev_transformer.shape_out
    #     self.kernel = self.cur_layer.weight
    #     self.padding = self.cur_layer.padding
    #     self.stride = self.cur_layer.stride
    #
    #     input_batch_size = self.shape_in[0]
    #     input_channels = self.shape_in[1]
    #     input_row_num = self.shape_in[2]
    #     input_col_num = self.shape_in[3]
    #
    #
    #     filter_row_num = self.kernel.shape[2]
    #     filter_col_num = self.kernel.shape[3]
    #
    #     output_row_num = input_row_num + filter_row_num - 1
    #     output_col_num = input_col_num + filter_col_num - 1
    #
    #     filter_padded = np.pad(self.kernel.detach().numpy(),
    #                            ((0,0),(0,0),(0,output_row_num-filter_row_num),(0,output_col_num-filter_col_num)),
    #                            'constant', constant_values=0)
    #
    #     toeplitz_list = []
    #
    #     for i in range(input_channels):
    #         for j in range(filter_padded.shape[2]-1, -1, -1):
    #             c = filter_padded[0,i,j,:]
    #             r = np.r_[c[0], np.zeros(filter_padded.shape[3]-1, dtype=int)]
    #             toeplitz_m = toeplitz(c,r)
    #             toeplitz_list.append(toeplitz_m)
    #
    #     toeplitz_list = np.array(toeplitz_list)
    #
    #     c = range(1, filter_padded.shape[2]+1)
    #     r = np.r_[c[0], np.zeros(filter_padded.shape[3]-1, dtype=int)]
    #
    #     doubly_indices = toeplitz(c,r)
    #
    #     toeplitz_shape = toeplitz_list[0].shape
    #
    #     h = toeplitz_shape[0] * doubly_indices.shape[0]
    #     w = toeplitz_shape[1] * doubly_indices.shape[1]
    #
    #     doubly_blocked_shape = [h, w]
    #
    #     doubly_blocked_matrix = np.zeros(doubly_blocked_shape)
    #
    #     doubly_blocked_matrix_list = []
    #
    #     for i in range(input_channels):











