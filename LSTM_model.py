import torch.nn as nn
import torch

class LSTMCell(nn.Module):
    '''
    input_size: 输入 X 的特征数量
    hidden_size: 隐含层的特征数量
    cell_size: 与hidde_size相同
    隐含状态h 跟 c 合在一起构成了LSTM的隐含状态

    '''

    def __init__(self, input_size, hidden_size, cell_size, output_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.gate = nn.Linear(input_size + hidden_size,cell_size)
        self.output = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        f_gate = self.sigmoid(self.gate(combined))
        i_gate = self.sigmoid(self.gate(combined))
        o_gate = self.sigmoid(self.gate(combined))
        z_state = self.tanh(self.gate(combined))
        cell = torch.add(torch.mul(cell, f_gate), torch.mul(z_state, i_gate))
        hidden =torch.mul(self.tanh(cell),o_gate)
        output = self.output(hidden)
        output = self.softmax(output)

        return output , hidden, cell

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def intCell(self):
        return torch.zeros(1, self.cell_size)