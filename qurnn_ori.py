import torch.nn as nn
import torch
import torch
from PIL import Image
import numpy as np
class QURNNCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, cell_dim, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        

        """
        super(QURNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim
        self.bias = bias

        self.W_time = nn.Linear(input_dim + hidden_dim, cell_dim, bias)
        self.W_absorb = nn.Linear(input_dim + hidden_dim, cell_dim, bias)
        # self.up_h1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias)
        # self.up_h2 = nn.Linear(input_dim + cell_dim + hidden_dim, hidden_dim, bias)

        self.energy_h = nn.Linear(hidden_dim + input_dim, hidden_dim, bias)
        self.h_weight = nn.Linear(hidden_dim + input_dim, hidden_dim, bias)
        # self.state_linear = nn.Linear(cell_dim ,hidden_dim, bias)
        self.state_linear = nn.Linear(cell_dim + input_dim, hidden_dim,bias=False)
        # self.cell_t = nn.Parameter(torch.rand((cell_dim)), requires_grad=True)
        # self.cell_h = nn.Parameter(torch.rand((cell_dim)), requires_grad=True)
        self.time_ratio = nn.Linear(input_dim, hidden_dim, bias)
        self.energy_init = nn.Linear(input_dim, hidden_dim, bias)
        self.time = nn.Linear (input_dim,hidden_dim,bias)
        # self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, input_tensor, cur_state):

        h_cur, excited_cell, time_cell = cur_state
        
        # initial
        # time_cell = time_cell * 0 
        # excited_cell = excited_cell * 0
        # from IPython import embed
        # embed()

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        # energy_init = self.sigmoid(self.energy_init(input_tensor))
        # ratio = self.sigmoid(self.time_ratio(input_tensor))
        # # Count the leap neuron
        
        up_time = self.tanh(self.W_time(combined))
        # time_cell = ratio * (time_cell+ up_time) - 1
        time_cell = self.relu(time_cell+ up_time) - 1
        leap = time_cell <= 0

        time_cell = (1-leap.type(torch.float32)) * time_cell
        

        # update excited state (release)
        energy = leap.type(torch.float32) * excited_cell
        # excited_cell = excited_cell - energy +  energy_init * (1-leap.type(torch.float32))
        excited_cell = excited_cell - energy

        # update excited state (absorb)
        absorb_energy = self.sigmoid(self.W_absorb(combined))
        excited_cell = self.relu(excited_cell + absorb_energy)      
        
        h_up = h_cur * energy

        h_state = torch.tanh(self.energy_h(torch.cat([input_tensor, h_up], dim=1)))

        h_weight = self.sigmoid(self.h_weight(combined))
        h_next = self.tanh((1 - h_weight) * h_cur + h_weight * h_state)

        return h_next, excited_cell, time_cell, 1-leap.type(torch.float32)

    def init_hidden(self, batch_size, cell_dim, hidden_dim):
        return (torch.rand(batch_size, hidden_dim, device=self.W_time.weight.device),
                torch.rand(batch_size, cell_dim, device=self.W_time.weight.device),
                torch.rand(batch_size, cell_dim, device=self.W_time.weight.device))
        
        """
        super(QURNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim

        # self.kernel_size = kernel_size
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)

        self.W_time = nn.Linear(input_dim + hidden_dim, cell_dim, bias)
        self.W_absorb = nn.Linear(input_dim + hidden_dim, cell_dim, bias)
        # self.up_h1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias)
        # self.up_h2 = nn.Linear(input_dim + cell_dim + hidden_dim, hidden_dim, bias)

        self.energy_h = nn.Linear(hidden_dim + input_dim, hidden_dim, bias)
        self.h_weight = nn.Linear(hidden_dim + input_dim, hidden_dim, bias)
        # self.state_linear = nn.Linear(cell_dim ,hidden_dim, bias)
        self.state_linear = nn.Linear(cell_dim , hidden_dim,bias=False)
        # self.cell_t = nn.Parameter(torch.rand((cell_dim)), requires_grad=True)
        # self.cell_h = nn.Parameter(torch.rand((cell_dim)), requires_grad=True)
        self.time_ratio = nn.Linear(input_dim, hidden_dim, bias)
        self.energy_init = nn.Linear(input_dim, hidden_dim, bias)
        self.time = nn.Linear (input_dim,hidden_dim,bias)
        # self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.exp = torch.exp()
        # self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    # def qu(self, x):
    #     x = torch.floor(x)

    #     emery_pre = 1/(x*x)
    #     emery_sub = 1/(self.b_ih)
    #     return emery_pre - emery_sub

    # def 

    def forward(self, input_tensor, cur_state):

        h_cur, excited_cell, time_cell = cur_state
        
        # initial
        # time_cell = time_cell * 0 
        # excited_cell = excited_cell * 0
        # from IPython import embed
        # embed()

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        energy_init = self.sigmoid(self.energy_init(input_tensor))
        ratio = self.sigmoid(self.time_ratio(input_tensor))
        # # Count the leap neuron
        
        up_time = self.tanh(self.W_time(combined))
        time_cell = ratio * self.relu(time_cell+ up_time) - 1
        # time_cell = time_cell+ up_time- 1
        # time_cell = time_cell - 1
        leap = time_cell <= 0

        time_cell = (1-leap.type(torch.float32)) * time_cell
        

        # update excited state (release)
        energy = leap.type(torch.float32) * excited_cell
        excited_cell = excited_cell - energy +  energy_init * (1-leap.type(torch.float32))
        # excited_cell = excited_cell - energy
        

        # update excited state (absorb)
        absorb_energy = self.sigmoid(self.W_absorb(combined))
        excited_cell = excited_cell + absorb_energy      
        
        h_up = h_cur * energy

        
        
        h_state = self.sigmoid(self.energy_h(torch.cat([input_tensor, h_up], dim=1)))
        # h_state = torch.exp(self.energy_h(torch.cat([input_tensor, h_up], dim=1)))
        # h_state = self.sigmoid(self.energy_h(torch.cat([input_tensor,h_cur],dim=1)))

        # h_state = self.relu(self.state_linear(excited_cell+time_cell))
        # h_state = self.sigmoid(self.state_linear(torch.cat([excited_cell+time_cell,input_tensor],dim=1)))
        # h_weight = self.sigmoid(self.h_weight(combined))
        # update hidden state
        h_next = self.tanh(h_cur) * h_state
        # h_next = self.tanh((1 - h_weight) * h_cur + (h_weight) * h_state)
        # h_next = self.tanh(torch.square(1 - h_weight) * h_cur + torch.square(h_weight) * h_state)
        # h_next = self.tanh(h_weight * h_state + h_cur)

        return h_next, excited_cell, time_cell

    def init_hidden(self, batch_size, cell_dim, hidden_dim):
        return (torch.rand(batch_size, hidden_dim, device=self.W_time.weight.device), 
                torch.rand(batch_size, cell_dim, device=self.W_time.weight.device), 
                torch.rand(batch_size, cell_dim, device=self.W_time.weight.device)) 
"""
class QURNN(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, cell_dim, output_dim, num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False):
        super(QURNN, self).__init__()

        # self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        cell_dim = self._extend_for_multilayer(cell_dim, num_layers)
        
        if not len(cell_dim) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(QURNNCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          cell_dim=self.cell_dim[i],
                                          bias=self.bias))
            # cell_list.append(nn.LSTM(cur_input_dim, self.hidden_dim[i], batch_first=True))

        self.cell_list = nn.ModuleList(cell_list)

        self.output_weight = nn.Sequential(nn.Linear(hidden_dim[-1], hidden_dim[-1]), 
                                            nn.BatchNorm1d(hidden_dim[-1]),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim[-1], output_dim))

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        input_tensor = input_tensor.squeeze()

        # input_tensor = input_tensor.unsqueeze(dim=0)
        
        if not self.batch_first:
            # (t, b, c) -> (b, t, c)
            input_tensor = input_tensor.permute(1, 0, 2)
        
        # from IPython import embed
        # embed()
        b, seq_len, c = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             cell_dim=self.cell_dim,
                                             hidden_dim=self.hidden_dim)

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor
        # y1, y2 = self.cell_list[0](cur_layer_input)
        # # from IPython import embed
        # # embed()
        # output = self.output_weight(y2[0][0])
        for layer_idx in range(self.num_layers):

            h, c_e, c_t = hidden_state[layer_idx]
            output_inner = []
            delta_time = []
            for t in range(seq_len):
                h, c_e, c_t, time = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :,],
                                                 cur_state=[h, c_e, c_t])
                output_inner.append(h)
                delta_time.append(time)

            layer_output = torch.stack(output_inner, dim=1)
            # increasement = torch.stack(delta_time,dim=1).squeeze()
            # increasement = delta_time[0].repeat(300,1)
            # print(delta_time[0].shape)
            # increasement = increasement[:20,:].repeat_interleave(10,dim=0)
            # print(increasement[:20,:].shape)
            
            # increasement = time
            # 64 300 300
            # print(increasement.shape)
            cur_layer_input = layer_output
            # increasement = increasement.to(torch.int)
            # print(torch.sum(increasement))
            # tensor_normalized = (increasement - increasement.min()) / (increasement.max() - increasement.min())  # 归一化到 0 到 1
            # tensor_normalized = (tensor_normalized * 255).byte()  # 转换为 0 到 255 的整数


# 将 Tensor 转换为 NumPy 数组
            # print(tensor_normalized.shape)
            
            # numpy_array =  tensor_normalized.cpu().numpy()

# 将 NumPy 数组转换为 PIL 图像对象
            # image = Image.fromarray(numpy_array)

# 保存图像
            # image.save("leap_mask_finally.png")
            layer_output_list.append(layer_output)
            last_state_list.append([h, c_e, c_t])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # from IPython import embed
        # embed()

        output = self.output_weight(last_state_list[-1][0])
        return layer_output_list, last_state_list, output
    
    def _init_hidden(self, batch_size, cell_dim, hidden_dim):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, cell_dim[i], hidden_dim[i]))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param