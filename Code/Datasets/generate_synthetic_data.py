import os
import numpy as np
import torch
import h5py
from netCDF4 import Dataset
from math import pi, sin, atan, cos, tan
import skimage  # 图像处理库，基于Scikit-image
import sys
from torch import tensor
script_dir = os.path.dirname(__file__)
utility_fn_dir = os.path.join(script_dir, "..", "Other")
sys.path.append(utility_fn_dir)
from utility_functions import tensor_to_cdf, tensor_to_h5, jacobian, normal, binormal, spatial_gradient



project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..", "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def vortex_x(x, y, z, x0, y0, z0, A=720):
    # 定义名为vortex_x的函数，接受6个参数：x, y, z坐标表示要查询速度的点的位置；
    # x0, y0, z0坐标表示涡旋中心的位置；A是一个标量系数，默认值为720，用于调整涡旋强度。
    sym = 1 if z - z0 <= 0 else -1
    # 计算符号变量sym，决定涡旋影响的符号。如果查询点的z坐标小于等于涡旋中心的z坐标（即查询点在涡旋中心下方或同一高度），
    # sym为1；否则为-1。这反映了涡旋在不同侧可能产生的速度方向差异。
    dist = (((x-x0)**2) + ((y-y0)**2) + ((z-z0)**2))  # 计算查询点到涡旋中心的距离平方dist，用于后续计算涡旋影响的衰减
    num = sym *  -A * (z - z0) * (x - x0)
    # 计算分子部分。它包括涡旋强度系数A、符号sym、以及查询点和涡旋中心在z轴和x轴上的相对位置差(z - z0)和(x - x0)的乘积。
    # 这个表达式体现了涡旋在x方向上诱导速度的大小和方向
    denom = (2*pi) * dist * ((((x-x0)**2) + ((y-y0)**2))**0.5)
    # 计算分母部分，包含了2*pi（一个常数项）、距离平方根dist，以及查询点在xy平面上距涡旋中心的径向距离（即(x-x0)^2 + (y-y0)^2的平方根）。
    # 分母代表了涡旋影响随距离增加而指数下降的物理规律
    return num / denom
    # 最后，返回分子除以分母的结果，即查询点在x方向上的涡旋诱导速度分量。
    # 这个值描述了涡旋在该点造成的流体或空气在x轴方向的速度变化。

def vortex_y(x, y, z, x0, y0, z0, A=720):
    sym = 1 if z - z0 <= 0 else -1
    dist = (((x-x0)**2) + ((y-y0)**2) + ((z-z0)**2))
    num = sym *  -A * (z - z0) * (y - y0)
    denom = (2*pi) * dist * ((((x-x0)**2) + ((y-y0)**2))**0.5)
    return num / denom

def vortex_z(x, y, z, x0, y0, z0, A=720):
    sym = 1 if z - z0 <= 0 else -1
    dist = (((x-x0)**2) + ((y-y0)**2) + ((z-z0)**2))
    num = sym *  A * ((((x-x0)**2)+((y-y0)**2))**0.5)
    denom = (2*pi) * dist
    return num / denom


"""这段代码定义了一个名为generate_vortices_data的函数，
其主要功能是生成一个包含两个涡旋流场数据的三维数组，并将其归一化处理后保存为NetCDF格式的文件"""
def generate_vortices_data(resolution = 128):  # 默认值128，用于决定生成数据的空间分辨率

    # [channels, u, v, w]
    a = np.zeros([3, resolution, resolution, resolution], dtype=np.float32)
    # 初始化一个四维数组a，形状为(3, resolution, resolution, resolution)，
    # 用于存储u、v、w三个通道（代表三维空间中的三个速度分量）的数据，数据类型为32位浮点
    # max vf mag = 6.78233, divide all components by that
    i = 0
    start = 1 
    end = 10   # 设置循环起始值start为1，结束值end为10，用于构建三维网格，
    for x in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
        # 使用嵌套循环遍历从1到10区间内按分辨率分割的每个网格点，分别对应x、y、z坐标
        j = 0
        for y in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)): 
            k = 0
            for z in np.arange(start, end + (end-start) / resolution, (end-start) / (resolution-1)):
                # 计算每个网格点上的速度分量u、v、w，通过调用之前定义的vortex_x、vortex_y、vortex_z函数，
                # 取两组涡旋中心(-5.5, -5.5, -5.5)和(15.0, 15.0, 15.0)的平均效果
                u = 0.5 * (vortex_x(x, y, z, -5.5, -5.5, -5.5) + \
                    vortex_x(x, y, z, 15.0, 15.0, 15.0))
                v = 0.5 * (vortex_y(x, y, z, -5.5, -5.5, -5.5) + \
                    vortex_y(x, y, z, 15.0, 15.0, 15.0))
                w = 0.5 * (vortex_z(x, y, z, -5.5, -5.5, -5.5) + \
                    vortex_z(x, y, z, 15.0, 15.0, 15.0))
                a[:,k,j,i] = np.array([u, v, w], dtype=np.float32)  # 将计算出的速度分量(u, v, w)存储到数组a对应的位置上
                #print("%0.02f %0.02f %0.02f" % (x, y, z))
                #print("%i %i %i" % (i, j, k))
                k += 1
            j += 1
        i += 1
    # 打印数组的最大值、最小值、均值及范数最大值，然后将数组a中的所有元素除以其最大范数进行归一化
    print(a.max())
    print(a.min())
    print(a.mean())
    print(np.linalg.norm(a, axis=0).max())
    a /= np.linalg.norm(a, axis=0).max()

    # 定义通道名称为'u', 'v', 'w'，然后将归一化后的numpy数组a转换为PyTorch张量，增加一个批次维度，
    # 转为32位浮点数类型，并通过tensor_to_cdf函数保存为名为vortices.nc的NetCDF格式文件，其中包含u、v、w三个通道数据
    channel_names = ['u', 'v', 'w']
    #tensor_to_h5(torch.tensor(a).unsqueeze(0).type(torch.float32), 
    #    "vortices.h5", channel_names) 
    tensor_to_cdf(torch.tensor(a).unsqueeze(0).type(torch.float32), 
        "vortices.nc", channel_names) 

def generate_flow_past_cylinder(resolution = 128, a=2):
    start = - 5
    end = 5
    # 创建一个三维网格，其中每个维度分别是 x、y、z 方向的坐标
    zyx = torch.meshgrid(
        [torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution)],
        indexing='ij'
    )
    zyx = torch.stack(zyx).type(torch.float32)  # 将三维网格堆叠成一个张量，并转换为 float32 类型
    x = zyx[2].clone()  # 获取 x 方向上的坐标
    y = zyx[1].clone()  # 获取 y 方向上的坐标
    z = zyx[0].clone()  # 获取 z 方向上的坐标
    #r = (x**2 + y**2)**0.5
    #mask = r < a
    #theta = torch.atan(y/x)

    # 计算流场的速度分量 u、v、w
    u = ((a**2)*(x**2 - y**2))/((x**2 + y**2)**2) - 1
    v = ((a**3)*x*y) / ((x**2 + y**2)**2)
    
    #u = torch.cos(2*theta) / r**2 - 1
    #v = torch.sin(2*theta) / r**2
    w = torch.zeros_like(u)   # w 分量初始化为零向量
    
    vf = torch.stack([u, v, w], dim=0)    # 将速度分量堆叠成一个张量，维度顺序为 (u, v, w)
    #vf = vf * ~mask
    # 计算速度分量对坐标的偏导数，构建雅可比矩阵 J
    dudx = -(a**3)*x*(x**2-3*y**2) / ((x**2 + y**2)**3)
    dudy = -(a**3)*y*(y**2-3*x**2) / ((x**2 + y**2)**3)
    dudz = torch.zeros_like(dudx)
    dvdx = (a**3)*y*(y**2-3*x**2) / ((x**2 + y**2)**3)
    dvdy = (a**3)*x*(x**2-3*y**2) / ((x**2 + y**2)**3)
    dvdz = torch.zeros_like(dudx)
    dwdx = torch.zeros_like(dudx)
    dwdy = torch.zeros_like(dudx)
    dwdz = torch.zeros_like(dudx)
    # 构建雅可比矩阵 J
    J = torch.stack([
        torch.stack([dudx, dudy, dudz]),
        torch.stack([dvdx, dvdy, dvdz]),
        torch.stack([dwdx, dwdy, dwdz])
    ])
    print('J.shape',J.shape)
    J = J.flatten(2).permute(2,0,1)    # 对 J 进行形状变换，将其展平并转置
    print('J.shape', J.shape)
    print('vf.shape', vf.shape)
    vf = vf.flatten(1).permute(1,0).unsqueeze(2)   # 对 vf 进行形状变换，将其展平并添加一个维度
    print('vf.shape', vf.shape)
    Jv = torch.bmm(J, vf)    # 计算 J 和 vf 的矩阵乘积，得到 Jv
    print('Jv.shape',Jv.shape)
    b = torch.cross(Jv, vf)    # 计算 b 和 n，这里使用了叉乘运
    print('b.shape',b.shape)
    n = torch.cross(b, vf)
    print('n.shape', n.shape)
    # 将 b 和 n 转换成特定的格式并保存为文件
    tensor_to_cdf(b.squeeze().permute(1,0).reshape(1,3,128,128,128), "binormal.nc")
    tensor_to_cdf(n.squeeze().permute(1,0).reshape(1,3,128,128,128), "normal.nc")
    #tensor_to_cdf(vf.unsqueeze(0).type(torch.float32), 
    #    "cylinder.nc")    

def generate_ABC_flow(resolution = 128, 
                      A=np.sqrt(3), B=np.sqrt(2), C=1):
    
    start = 0
    end = 2*np.pi
    
    zyx = torch.meshgrid(
        [torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution)],
        indexing='ij'
    )
    zyx = torch.stack(zyx).type(torch.float32)
    x = zyx[2].clone()
    y = zyx[1].clone()
    z = zyx[0].clone()
    
    u = A*torch.sin(z) + C*torch.cos(y)
    v = B*torch.sin(x) + A*torch.cos(z)
    w = C*torch.sin(y) + B*torch.cos(x)
    
    abc = torch.stack([u,v,w], dim=0).unsqueeze(0)
    print(abc.shape)
    print(abc.max())
    print(abc.min())
    print(abc.mean())
    print(abc.norm(dim=1).max())
    abc /= abc.norm(dim=1).max()
    
    channel_names = ['u', 'v', 'w']
    tensor_to_cdf(abc.type(torch.float32), "ABC_flow.h5", channel_names)
    tensor_to_cdf(abc.type(torch.float32), "ABC_flow.nc", channel_names)


# 生成一个 3D Hill Vortex（希尔涡旋），并将其保存为 NetCDF 文件格式
def generate_hill_vortex(resolution=128,A=np.sqrt(3), B=np.sqrt(2), C=1):  # 默认生成一个 128x128x128 的 3D 网格

    # 生成的网格范围从 -2 到 2
    start = -2
    end = 2
    # 生成一个 3D 网格，其尺寸为 resolution x resolution x resolution。网格点在 -2 到 2 之间均匀分布
    zyx = torch.meshgrid(
        [torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution),
        torch.linspace(start, end, steps=resolution)],
        indexing='ij'
    )
    zyx = torch.stack(zyx).type(torch.float32)

    # 提取网格的 x、y、z 坐标，并计算每个网格点到原点的距离 r
    x = zyx[2].clone()
    y = zyx[1].clone()
    z = zyx[0].clone()
    r = (x**2 + y**2 + z**2)**0.5
    # 使用分段函数来计算速度分量 u、v 和 w。在距离 r 大于 1 的区域和小于等于 1 的区域，速度分量的计算公式不同
    u = (r > 1).type(torch.LongTensor) * (3*y*x)/(2*(r**5)) + \
        (r <= 1).type(torch.LongTensor) * (1.5*y*x)
    v = (r > 1).type(torch.LongTensor) * ((3*y**2) - r**2)/(2*(r**5)) + \
        (r <= 1).type(torch.LongTensor) * (1.5*(1 - 2*(r**2)))
    w = (r > 1).type(torch.LongTensor) * (3*y*z)/(r*(r**5)) + \
        (r <= 1).type(torch.LongTensor) * (1.5*y*z)
    # 将 u、v 和 w 堆叠成一个张量，并在第 0 维添加一个批次维度
    hill = torch.stack([u,v,w], dim=0).unsqueeze(0)
    print('hill.shape', hill.shape)
    print(hill.max())
    print(hill.min())
    print(hill.mean())
    print(hill.norm(dim=1).max())

    # 将速度场归一化，使其范数的最大值为 1
    hill /= hill.norm(dim=1).max()
    
    channel_names = ['u', 'v', 'w']
    tensor_to_cdf(hill.type(torch.float32), 
        "hill.nc", channel_names)


# 这段代码的作用是从二进制文件中读取三维流场数据（U、V、W分量），对数据进行预处理（包括类型转换、特定值替换、重塑形状以及转换为Tensor），
# 然后将这些数据合并并保存为两种不同的科学数据格式文件（HDF5和NetCDF）。下面是逐行的详细解释
def isabel_from_bin():
    u = np.fromfile('U.bin', dtype='>f')
    u = u.astype(np.float32)
    u[np.argwhere(u == 1e35)] = 0
    u = u.reshape([100, 500, 500])
    u = torch.tensor(u).unsqueeze(0).unsqueeze(0)

    v = np.fromfile('V.bin', dtype='>f')
    v = v.astype(np.float32)
    v[np.argwhere(v == 1e35)] = 0
    v = v.reshape([100, 500, 500])
    v = torch.tensor(v).unsqueeze(0).unsqueeze(0)

    w = np.fromfile('W.bin', dtype='>f')
    w = w.astype(np.float32)
    w[np.argwhere(w == 1e35)] = 0
    w = w.reshape([100, 500, 500])
    w = torch.tensor(w).unsqueeze(0).unsqueeze(0)

    uvw = torch.cat([u,v,w], dim=1)

    tensor_to_h5(uvw, "isabel.h5")
    tensor_to_cdf(uvw, "isabel.nc")
    # tensor_to_nc(uvw, "isabel.nc")

def plume_data_reading():
    # 使用numpy从指定路径读取代表流动u分量的数据文件，数据类型设置为32位浮点型
    u = np.fromfile('F:/Visualization Data/Plume/15plume3d435.ru',
                    dtype=np.float32)
    v = np.fromfile('F:/Visualization Data/Plume/15plume3d435.rv',
                    dtype=np.float32)
    w = np.fromfile('F:/Visualization Data/Plume/15plume3d435.rw',
                    dtype=np.float32)
    # 将读取的u分量数据转换为PyTorch张量，并重塑其形状为(1024, 252, 252)，假设这代表了时间步数x高度x宽度的三维数据结构
    u = torch.tensor(u).reshape(1024, 252, 252)
    v = torch.tensor(v).reshape(1024, 252, 252)
    w = torch.tensor(w).reshape(1024, 252, 252)
    # 将u、v、w三个张量沿新的维度堆叠起来，形成一个形状为(1, 1024, 252, 252, 3)的五维张量，其中新增的第一维可能用于批处理
    uvw = torch.stack([u, v, w]).unsqueeze(0)
    # 定义一个函数tensor_to_h5（未在此代码段中给出）来将PyTorch张量uvw保存为HDF5格式的文件，便于高效存储和后续访问
    tensor_to_h5(uvw, "plume.h5")

'''
Comments from Professor Crawfis's original code hosted at
http://web.cse.ohio-state.edu/~crawfis.3/Data/Tornado/tornadoSrc.c

Gen_Tornado creates a vector field of dimension [xs,ys,zs,3] from
a proceedural function. By passing in different time arguements,
a slightly different and rotating field is created.

The magnitude of the vector field is highest at some funnel shape
and values range from 0.0 to around 0.4 (I think).

I just wrote these comments, 8 years after I wrote the function.

Developed by Roger A. Crawfis, The Ohio State University
'''
def generate_crawfis_tornado(x_res, y_res, z_res, time=0):

    tornado = np.zeros([z_res, y_res, x_res, 3])
    r2 = 8
    SMALL = 0.00000000001
    xdelta = 1.0 / (x_res-1)
    ydelta = 1.0 / (y_res-1)
    zdelta = 1.0 / (z_res-1)

    z_ind = 0
    for z in np.arange(0.0, 1.0, zdelta):
        xc = 0.5 + 0.1*sin(0.04*time+10.0*z);           #For each z-slice, determine the spiral circle.
        yc = 0.5 + 0.1*cos(0.03*time+3.0*z);            #(xc,yc) determine the center of the circle.
        r = 0.1 + 0.4 * z*z + 0.1 * z * sin(8.0*z);     #The radius also changes at each z-slice.
        r2 = 0.2 + 0.1*z;                               #r is the center radius, r2 is for damping
        y_ind = 0               
        for y in np.arange(0.0, 1.0, ydelta):
            x_ind = 0
            for x in np.arange(0.0, 1.0, xdelta):
                temp = ( (y-yc)*(y-yc) + (x-xc)*(x-xc) ) ** 0.5
                scale = abs( r - temp )
                '''
                I do not like this next line. It produces a discontinuity 
                in the magnitude. Fix it later.
                '''
                if ( scale > r2 ):
                    scale = 0.8 - scale
                else:
                    scale = 1.0

                z0 = 0.1 * (0.1 - temp*z )
                if ( z0 < 0.0 ):
                    z0 = 0.0

                temp = ( temp*temp + z0*z0 )**0.5
                scale = (r + r2 - temp) * scale / (temp + SMALL)
                scale = scale / (1+z)

                # In u,v,w order 
                tornado[z_ind, y_ind, x_ind, 0] = scale * (y-yc) + 0.1*(x-xc)
                tornado[z_ind, y_ind, x_ind, 1] = scale * -(x-xc) + 0.1*(y-yc)
                tornado[z_ind, y_ind, x_ind, 2] = scale * z0

                x_ind = x_ind + 1
            y_ind = y_ind + 1
        z_ind = z_ind + 1
    
    return tornado

def generate_lorenz_attractor(x_res, y_res, z_res, sigma=10, beta=8/3, rho=28):

    vf = np.zeros([z_res, y_res, x_res, 3])
    
    start = -50
    end = 50
    xdelta = (end-start) / (x_res-1)
    ydelta = (end-start) / (y_res-1)
    zdelta = (end-start) / (z_res-1)

    z_ind = 0
    for z in np.arange(start, end, zdelta): 
        
        y_ind = 0               
        for y in np.arange(start, end, ydelta):

            x_ind = 0
            for x in np.arange(start, end, xdelta):
                
                vf[z_ind, y_ind, x_ind, 0] = sigma * (y-x)
                vf[z_ind, y_ind, x_ind, 1] = x * (rho - z) - y
                vf[z_ind, y_ind, x_ind, 2] = x*y - beta*z

                x_ind = x_ind + 1
            y_ind = y_ind + 1
        z_ind = z_ind + 1
    
    return vf/np.linalg.norm(vf, axis=3).max()


"""生成一个包含100个三维空间中随机点的CSV文件，这些点被设计用于作为某些模拟或可视化（如流体动力学中的涡旋模拟）中的种子点"""
def generate_vortices_seed_points():
    # 使用PyTorch库生成一个形状为[100, 3]的张量，其中的元素是0到1之间的随机数
    seeds = torch.rand([100, 3])*2-1
    # 将这些随机点缩放，使其范围变为-1到1乘以64，即-64到64。
    seeds *= 64
    # 接着将所有点平移，使它们的范围变为0到128
    seeds += 64
    # 导入csv模块，以便写入CSV文件
    import csv
    # 使用with语句打开一个名为'vortices_seeds.csv'的文件，以写入模式('w')，并确保换行处理得当
    with open('vortices_seeds.csv', 'w', newline='') as csvfile:
        # 创建一个csv.writer对象，指定逗号作为分隔符，使用竖线作为引用字符，并设置最少的引用规则。
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # 遍历之前生成的随机点张量的每一行
        for i in range(seeds.shape[0]):
            # 将当前行转换为NumPy数组（以便于写入CSV文件），然后通过writerow方法写入文件
            w.writerow(seeds[i].numpy())
        
def generate_cylinder_seed_points():    
    seeds = torch.rand([100, 3])*2-1
    seeds *= 64
    seeds += 64
    import csv
    with open('cylinder_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())

def generate_ABC_flow_seed_points():    
    seeds = torch.rand([100, 3])*2-1
    seeds *= 16
    seeds[:,0] += 32
    seeds[:,1] += 64
    seeds[:,2] += 64
    import csv
    with open('ABC_flow_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())
            
def generate_tornado_seed_points():    
    seeds = torch.rand([100, 3])*2-1
    seeds[:,0] *= 16
    seeds[:,0] += 32
    seeds[:,1] *= 16
    seeds[:,1] += 32
    seeds[:,2] *= 64
    seeds[:,2] += 64
    import csv
    with open('tornado_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())
   
def generate_isabel_seed_points():    
    seeds = torch.rand([200, 3])*2-1
    seeds[:100,0] *= 50
    seeds[:100,0] += 350
    seeds[:100,1] *= 50
    seeds[:100,1] += 350
    seeds[:100,2] *= 50
    seeds[:100,2] += 50
    
    seeds[100:,0] *= 80
    seeds[100:,0] += 100
    seeds[100:,1] *= 80
    seeds[100:,1] += 100
    seeds[100:,2] *= 50
    seeds[100:,2] += 50

    import csv
    with open('isabel_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())
    
def generate_plume_seed_points():    
    seeds = torch.rand([200, 3])*2-1
    seeds[:50,0] *= 20
    seeds[:50,0] += 125
    seeds[:50,1] *= 20
    seeds[:50,1] += 125
    seeds[:50,2] *= 20
    seeds[:50,2] += 511
    
    seeds[50:100,0] *= 128
    seeds[50:100,0] += 128
    seeds[50:100,1] *= 128
    seeds[50:100,1] += 128
    seeds[50:100,2] *= 10
    seeds[50:100,2] += 790
    
    seeds[100:,0] *= 128
    seeds[100:,0] += 128
    seeds[100:,1] *= 128
    seeds[100:,1] += 128
    seeds[100:,2] *= 50
    seeds[100:,2] += 236

    import csv
    with open('plume_seeds.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(seeds.shape[0]):
            w.writerow(seeds[i].numpy())
      
def generate_seed_files():
    generate_vortices_seed_points()
    generate_cylinder_seed_points()
    generate_ABC_flow_seed_points()
    generate_tornado_seed_points()
    generate_isabel_seed_points()
    generate_plume_seed_points()      


"""综上所述，该脚本主要目的是生成一系列与流体动力学模拟相关的数据集（包括绕流圆柱、涡旋数据、以及模拟的龙卷风数据），
并以高精度格式保存其中一个数据集到NetCDF文件，便于后续的分析或模拟使用"""
if __name__ == '__main__':
    # 确保如果此脚本被直接运行（而非作为模块导入），则执行以下代码
    torch.manual_seed(0)
    # 设置PyTorch的随机数生成器种子，以便结果可复现

    generate_seed_files()  # 调用函数来生成种子文件，可能是用于后续模拟的初始条件
    # generate_flow_past_cylinder(resolution=10, a=2)
    # 调用函数生成绕流圆柱的流场数据，设置分辨率为10，圆柱半径或其他参数为2

    generate_vortices_data(resolution=10)   # 调用另一个函数生成涡旋相关的数据，分辨率为10
    generate_flow_past_cylinder(resolution=128)  # 再次调用生成绕流圆柱的函数，但这次使用更高的分辨率128，可能为了获得更精细的模拟结果

    generate_ABC_flow()  # 调用函数生成模拟的ABC流数据
    generate_hill_vortex()

    t = generate_crawfis_tornado(128, 128, 128, 0)
    # 生成一个代表“Crawfis龙卷风”（可能是某种特殊类型的涡旋或模拟场景）的数据，尺寸为128x128x128，附加参数0。
    # 结果存储在变量t中。

    t = torch.tensor(t).permute(3, 0, 1, 2).unsqueeze(0).type(torch.float32)
    # 将t转换为PyTorch张量，然后调整维度顺序（假设原始数据是CHW或类似格式，转换为适合深度学习的NCHW格式），
    # 并添加一个批次维度（unsqueeze(0)），最后确保数据类型为float32。

    # t = generate_lorenz_attractor(128, 128, 128)
    # t = torch.tensor(t).permute(3, 0, 1, 2).unsqueeze(0).type(torch.float32)
    tensor_to_cdf(t, "tornado_generated.nc")
    # 将张量t的数据保存到一个名为`tornado_generated.nc`的NetCDF文件中，NetCDF是一种常用的科学数据格式，适用于多维数组数据
    quit()