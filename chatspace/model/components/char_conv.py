"""
Copyright 2019 Pingpong AI Research, ScatterLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn


class CharConvolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=config["embedding_dim"],
            out_channels=config["cnn_features"],
            kernel_size=config["cnn_filter"],
        )
        self.conv2 = nn.Conv1d(
            in_channels=config["cnn_features"],
            out_channels=config["cnn_features"],
            kernel_size=config["cnn_filter"] * 2 + 1,
        )
        self.padding_1 = nn.ConstantPad1d(1, 0)
        self.padding_2 = nn.ConstantPad1d(3, 0)

    def forward(self, embed_input):
        embed_input = torch.transpose(embed_input, dim0=1, dim1=2)
        conv1_output = self.conv1(embed_input)
        conv1_paded = self.padding_1(conv1_output)
        conv2_output = self.conv2(conv1_paded)
        conv2_paded = self.padding_2(conv2_output)
        return torch.cat((conv1_paded, conv2_paded), dim=1)
