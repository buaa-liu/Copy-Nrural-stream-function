from __future__ import absolute_import, division, print_function
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
script_dir = os.path.dirname(__file__)
other_dir = os.path.join(script_dir, "..", "Other")
sys.path.append(other_dir)
sys.path.append(script_dir)
from siren import SIREN
from fSRN import fSRN
from fVSRN import fVSRN
from grid import Grid
from options import save_options
from utility_functions import make_coord_grid, create_folder

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def save_model(model,opt):   # 保存模型和选项
    folder = create_folder(save_folder, opt["save_name"])
    path_to_save = os.path.join(save_folder, folder)
    
    torch.save({'state_dict': model.state_dict()}, 
        os.path.join(path_to_save, "model.ckpt.tar"),
        pickle_protocol=4
    )
    save_options(opt, path_to_save)

def load_model(opt, device):  # 加载模型，从指定位置加载模型的状态字典，并将其加载到指定的设备
    path_to_load = os.path.join(save_folder, opt["save_name"])
    model = create_model(opt)

    ckpt = torch.load(os.path.join(path_to_load, 'model.ckpt.tar'), 
        map_location = device)
    
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(opt['device'])
    return model

def create_model(opt):   # 创建模型，根据选项中指定的模型类型，创建相应的模型实例
    if(opt['model'] == "siren"):
        return SIREN(opt)
    elif(opt['model'] == 'fSRN'):
        return fSRN(opt)
    elif(opt['model'] == 'fVSRN'):
        return fVSRN(opt)
    elif(opt['model'] == 'grid'):
        return Grid(opt)
    else:
        print(f"Model {opt['model']} does not exist.")
        quit()

def sample_grid(model, grid, max_points = 100000):  # 对模型在网格上进行采样，使用模型对给定网格上的点进行前向传播，并返回采样值
    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    vals = forward_maxpoints(model, coord_grid, max_points = max_points)
    coord_grid_shape[-1] = model.opt['n_outputs']
    vals = vals.reshape(coord_grid_shape)
    return vals

def sample_grad_grid(model, grid,  
    output_dim = 0, max_points=1000):   # 对模型在网络上进行梯度采样
    
    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])
    
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)       

    output_shape = list(coord_grid.shape)
    output_shape[-1] = model.opt['n_dims']
    print("Output shape")
    print(output_shape)
    output = torch.empty(output_shape, 
        dtype=torch.float32, device=model.opt['device'], 
        requires_grad=False)

    for start in range(0, coord_grid.shape[0], max_points):
        vals = model(
            coord_grid[start:min(start+max_points, coord_grid.shape[0])])
        grad = torch.autograd.grad(vals[:,output_dim], 
            coord_grid, 
            grad_outputs=torch.ones_like(vals[:,output_dim])
            )[0][start:min(start+max_points, coord_grid.shape[0])]
        
        output[start:min(start+max_points, coord_grid.shape[0])] = grad

    output = output.reshape(coord_grid_shape)
    
    return output

def sample_grid_for_image(model, grid, 
    boundary_scaling = 1.0):       # 对模型在图像网格上进行采样，返回值通常用于生成图像
    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])
    if(len(coord_grid.shape) == 4):
        coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
    
    coord_grid *= boundary_scaling

    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    vals = forward_maxpoints(model, coord_grid)
    coord_grid_shape[-1] = model.opt['n_outputs']
    vals = vals.reshape(coord_grid_shape)
    if(model.opt['loss'] == "l1occupancy"):
        vals = vals[..., 0:-1]
    return vals

def sample_occupancy_grid_for_image(model, grid, opt, boundary_scaling = 1.0):  # 
    coord_grid = make_coord_grid(grid, 
        opt['device'], flatten=False,
        align_corners=opt['align_corners'])
    if(len(coord_grid.shape) == 4):
        coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
    
    coord_grid *= boundary_scaling

    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    vals = forward_maxpoints(model, coord_grid)
    coord_grid_shape[-1] = model.opt['n_outputs']
    vals = vals.reshape(coord_grid_shape)
    if(model.opt['loss'] == "l1occupancy"):
        vals = vals[...,-1]
    return vals

def sample_grad_grid_for_image(model, grid, boundary_scaling = 1.0, 
    input_dim = 0, output_dim = 0):  # 对模型在图像网格上进行梯度采样，返回的是输出维度相对于输入网格的梯度

    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])      
    if(len(coord_grid.shape) == 4):
        coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
    coord_grid *= boundary_scaling
    
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)
    vals = forward_maxpoints(model, coord_grid)    


    grad = torch.autograd.grad(vals[:,output_dim], 
        coord_grid,#[:,input_dim], 
        grad_outputs=torch.ones_like(vals[:, output_dim]),
        allow_unused=True)
    

    grad = grad[0][:,input_dim]
    coord_grid_shape[-1] = 1
    grad = grad.reshape(coord_grid_shape)
    
    return grad

def sample_rect(model, starts, widths, samples):  # 在矩形网格上对模型进行采样。
    # 在指定的起始点、宽度和采样数量下对模型及逆行前向传播， 并返回采样值
    positions = []
    for i in range(len(starts)):
        positions.append(
            torch.arange(starts[i], starts[i] + widths[i], widths[i] / samples[i], 
                dtype=torch.float32, device=model.opt['device'])
        )
    grid_to_sample = torch.stack(torch.meshgrid(*positions), dim=-1)
    vals = model.forward(grid_to_sample)
    return vals

def sample_grad_rect(model, starts, widths, samples, input_dim, output_dim):  # 在矩形网格上对模型进行梯度采样
    positions = []
    for i in range(len(starts)):
        positions.append(
            torch.arange(starts[i], starts[i] + widths[i], widths[i] / samples[i], 
                dtype=torch.float32, device=model.opt['device'])
        )
    grid_to_sample = torch.stack(torch.meshgrid(*positions), dim=-1).requires_grad_(True)
    vals = model.forward(grid_to_sample)
    
    grad = torch.autograd.grad(vals[:,output_dim], 
        grid_to_sample,#[:,input_dim], 
        grad_outputs=torch.ones_like(vals[:, output_dim]),
        allow_unused=True)
    

    grad = grad[0][:,input_dim]
    grid_to_sample[-1] = 1
    grad = grad.reshape(grid_to_sample)  #                    

    return vals

def forward_w_grad(model, coords):  # 在给定的坐标处对模型进行前向传播，并返回输出及其相对于输入的梯度
    coords = coords.requires_grad_(True)
    output = model(coords)
    return output, coords

def forward_maxpoints(model, coords, max_points=100000):  # 在坐标点处对模型进行前向出纳比，但是将输入坐标划分为多个小批次以节省内存
    #print(coords.shape)
    output_shape = list(coords.shape)
    output_shape[-1] = 1
    output = torch.empty(output_shape, 
        dtype=torch.float32, device=coords.device)
    for start in range(0, coords.shape[0], max_points):
        #print("%i:%i" % (start, min(start+max_points, coords.shape[0])))
        output[start:min(start+max_points, coords.shape[0])] = \
            model(coords[start:min(start+max_points, coords.shape[0])])
    return output

class LReLULayer(nn.Module):  # 自定义的带有LeakyRelu激活函数的先行曾
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, 
            bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(
                -torch.nn.init.calculate_gain("leaky_relu", 0.2),
                torch.nn.init.calculate_gain("leaky_relu", 0.2)
            )

    def forward(self, input):
        return F.leaky_relu(self.linear(input), 0.2)
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.linear(input)
        return F.leaky_relu(intermediate, 0.2), intermediate

