import torch
import torch.nn as nn
import numpy as np
import os
import sys
script_dir = os.path.dirname(__file__)
other_dir = os.path.join(script_dir, "..", "Other")
sys.path.append(other_dir)
from utility_functions import make_coord_grid

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, 
            bias=bias)
        
        self.init_weights()
    
    def init_weights(self):   # 初始化权重，根据是否是第一层采用不同的初始化方法
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        # 前向传播过程，即将输入数据经过线性变换后再经过正弦函数变换
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input):
        # 可视化激活分布，返回正弦函数变换前后的结果
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class SIREN(nn.Module):
    # 定义了一个SIREN类，用于组合多个SineLayer层构建整个SIREN模型
    def __init__(self, opt):
        # 在初始化函数__init__中，根据传入的参数opt，构建了SIREN模型的结构
        super().__init__()
        
        self.opt = opt
        self.net = []

        self.net.append(SineLayer(opt['n_dims'], 
            opt['nodes_per_layer'], 
            is_first=True, omega_0=30))

        i = 0
        while i < opt['n_layers']:            
            self.net.append(SineLayer(opt['nodes_per_layer'], 
                opt['nodes_per_layer'], 
                is_first=False, omega_0=30))
            i += 1

        final_linear = nn.Linear(opt['nodes_per_layer'], opt['n_outputs'])
            
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / opt['nodes_per_layer']) / 30, 
                                            np.sqrt(6 / opt['nodes_per_layer']) / 30)
            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):     
        output = self.net(coords)
        return output
