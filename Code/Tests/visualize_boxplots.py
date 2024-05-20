from __future__ import absolute_import, division, print_function
# 导入所需的模块：
import argparse
import os
import sys

# 设置脚本所需的路径：
script_dir = os.path.dirname(__file__)
other_dir = os.path.join(script_dir, "..", "Other")
models_dir = os.path.join(script_dir, "..", "Models")
datasets_dir = os.path.join(script_dir, "..", "Datasets")

# 将脚本所需的路径添加到系统路径中：
sys.path.append(other_dir)
sys.path.append(models_dir)
sys.path.append(datasets_dir)
sys.path.append(script_dir)
import numpy as np
import matplotlib.pyplot as plt

# 定义项目文件夹路径，并设置数据文件夹、输出文件夹和保存模型文件夹的路径：
project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")  # 数据文件夹
output_folder = os.path.join(project_folder_path, "Output")   # 输出文件夹
save_folder = os.path.join(project_folder_path, "SavedModels")  # 保存模型文件夹

if __name__ == '__main__':
    # 解析命令行参数：
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')
    parser.add_argument('--error_files',default=None,type=str,nargs='+',help=".npy files to load")
    parser.add_argument('--names',default=None,type=str,nargs='+',help=".npy files to load")
    parser.add_argument('--title',default=None,type=str,help="Title")
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    folder_with_npys = os.path.join(output_folder, "Error")

    # 遍历给定的.npy文件列表，加载数据并计算其中数据的中位数：
    arrays = []
    files = args['error_files']
    names = args['names']


    # 设置绘图样式和字体参数：
    for filename in files:
        full_path = os.path.join(folder_with_npys, filename)
        a = np.load(full_path)
        arrays.append(a)
        print(np.median(a))

    plt.style.use('ggplot')
    #plt.style.use('seaborn')
    #plt.style.use('seaborn-paper')

    font = {#'font.family' : 'normal',
        #'font.weight' : 'bold',
        'font.size'   : 20,
        'lines.linewidth' : 3}
    plt.rcParams.update(font)

    # 绘制箱线图：
    plt.boxplot(arrays, vert=False, showfliers=False, labels=names)
    plt.xlabel(args['title'])
    plt.show()
    
    
        
    
        



        

