# 导入必要模块
import torch
import torch.nn as nn
import numpy as np
import os
import sys
script_dir = os.path.dirname(__file__)
other_dir = os.path.join(script_dir, "..", "Other")
sys.path.append(other_dir)
"""这段代码的作用是将当前脚本文件所在目录的上一级目录中的"Other"目录添加到Python模块搜索路径中:
os.path.dirname(__file__)用于获取当前脚本文件的所在目录。
os.path.join(script_dir, "..", "Other")将当前脚本文件所在目录的上一级目录与"Other"目录拼接，得到"Other"目录的绝对路径。
sys.path.append(other_dir)将"Other"目录的绝对路径添加到Python模块搜索路径中，这样在后续的代码中就可以导入该目录下的模块或文件了。
"""
# 导入自定义模块
from utility_functions import make_coord_grid, PositionalEncoding
# 这里引用爆红，可能是python版本的问题。前述代码已经将不同文件夹的内容进行了引用
from siren import SineLayer


# 定义带有余弦激活函数的残差层
class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
  
    # 初始化权重
    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
    # 前向传播
    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)

    # 改变层的节点数
    def change_nodes_per_layer(self, num_nodes):
        l1_weights = self.linear_1.weight.detach().clone()
        l1_bias = self.linear_1.bias.detach().clone()
        l2_weights = self.linear_2.weight.detach().clone()
        l2_bias = self.linear_2.bias.detach().clone()
        
        print(l1_weights.shape)
        self.features = num_nodes
        
        self.linear_1 = nn.Linear(num_nodes, num_nodes, bias=True)
        self.linear_2 = nn.Linear(num_nodes, num_nodes, bias=True)

        self.init_weights()
        
        print(self.linear_1.weight.shape)

# 定义全连接神经网络
class fSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.net = []
        #self.net.append(
        #    PositionalEncoding(opt)
        #)
                #opt['num_positional_encoding_terms']*opt['n_dims']*2, 
        self.net.append(
            SineLayer(
                opt['n_dims'],
                opt['nodes_per_layer'], 
                is_first=True, omega_0=opt['omega']
                )
            )
        # 添加残差层
        i = 0
        while i < opt['n_layers']:
            self.net.append(ResidualSineLayer(opt['nodes_per_layer'], 
                ave_first=i>0,
                ave_second=(i==opt['n_layers']-1),
                omega_0=opt['omega']))                 
            i += 1
        # 添加最终线性层并初始化权重
        final_linear = nn.Linear(opt['nodes_per_layer'], 
                                 opt['n_outputs'], bias=True)
            
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / opt['nodes_per_layer']) / 30, 
                                            np.sqrt(6 / opt['nodes_per_layer']) / 30)
            
        self.net.append(final_linear)
        #self.net.append(nn.BatchNorm1d(opt['n_outputs'], affine=False))
        
        self.net = nn.Sequential(*self.net)

    # 改变层的节点数
    
    def change_nodes_per_layer(self, num_nodes):
        for layer in self.net:
            if(layer.__class__ == ResidualSineLayer):
                print("Residual sine")
                layer.change_nodes_per_layer(num_nodes)
            elif(layer.__class__ == SineLayer):
                print("Sine layer")
            elif(layer.__class__ == nn.Linear):
                print("Linear")
            else:
                print(layer.__class__)
    
    def forward(self, coords):     
        output = self.net(coords)
        return output

        