from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ExplicitGRUCell(nn.Module):
    """用基础算子显式实现 GRU 门，保证高阶 MAML 梯度在 CPU/GPU 上可移植。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        # 三组门按 reset、update、candidate 的顺序存放。
        input_gates = F.linear(x, self.weight_ih, self.bias_ih)
        hidden_gates = F.linear(previous, self.weight_hh, self.bias_hh)
        input_reset, input_update, input_candidate = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_candidate = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_candidate + reset * hidden_candidate)
        # 论文 Eq. (11)：h_t = (1-z_t) h_{t-1} + z_t h_tilde。
        return (1.0 - update) * previous + update * candidate


class GRURegressor(nn.Module):
    """可堆叠的显式 GRU 回归器，输出窗口末端的健康度/RUL 百分比。"""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_activation: str = "linear",
    ) -> None:
        super().__init__()
        if output_activation not in {"linear", "sigmoid"}:
            raise ValueError("output_activation must be 'linear' or 'sigmoid'")
        self.hidden_size = hidden_size
        self.dropout = dropout if num_layers > 1 else 0.0
        self.output_activation = output_activation
        self.cells = nn.ModuleList(
            [ExplicitGRUCell(input_size if index == 0 else hidden_size, hidden_size) for index in range(num_layers)]
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状：[batch, time, feature]；每层维护独立隐藏状态。
        hidden = [x.new_zeros((x.shape[0], self.hidden_size)) for _ in self.cells]
        for time_index in range(x.shape[1]):
            layer_input = x[:, time_index]
            for layer_index, cell in enumerate(self.cells):
                hidden[layer_index] = cell(layer_input, hidden[layer_index])
                layer_input = hidden[layer_index]
                if layer_index + 1 < len(self.cells):
                    layer_input = F.dropout(layer_input, p=self.dropout, training=self.training)
        output = self.head(hidden[-1])
        # 归一化 RUL 使用 sigmoid 可确保预测落在 [0, 1]；linear 用于消融实验。
        return torch.sigmoid(output) if self.output_activation == "sigmoid" else output
