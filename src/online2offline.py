import numpy as np
import matplotlib.pyplot as plt

import os
import glob

# 读取 .svc 文件
def read_svc_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # 跳过第一行（无意义的单个数字）
            # 分割每行数据#
            parts = line.strip().split()
            if len(parts)!= 7:
                continue
            x, y, timestamp, pen_status, azimuth, altitude, pressure = map(int, parts)
            data.append((x, y, timestamp, pen_status, azimuth, altitude, pressure))
    return np.array(data)


# 生成笔迹图像
def plot_handwriting(data, output_path):
    # 创建画布
    plt.figure(figsize=(10, 10))
    #plt.gca().invert_yaxis()  # 反转 y 轴，因为数位板的坐标原点通常在左上角

    # 初始化变量
    prev_x, prev_y = None, None
    current_line = []

    # 遍历数据点
    for x, y, _, pen_status, _, _, pressure in data:
        # 逆时针旋转 90 度：交换 x 和 y，并调整符号
        rotated_x = -y  # 新 x = -旧 y
        rotated_y = x    # 新 y = 旧 x

        if pen_status == 1:  # 笔在纸上
            if prev_x is not None and prev_y is not None:
                # 根据笔压调整线条粗细
                line_width = 0.1 + (pressure / 1023) * 5  # 笔压归一化并映射到线条粗细
                plt.plot([prev_x, rotated_x], [prev_y, rotated_y], color='black', linewidth=line_width, solid_capstyle='round')
            prev_x, prev_y = rotated_x, rotated_y
        else:  # 笔在空中
            prev_x, prev_y = None, None

    # 设置图像属性
    plt.axis('off')  # 关闭坐标轴
    plt.gca().set_aspect('equal', adjustable='box')  # 保持比例

    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 主函数
def main():
    input_dir = './Dataset/house'    # top-level input folder
    output_dir = './image/house'     # top-level output folder for PNGs
    os.makedirs(output_dir, exist_ok=True)
    
    # collect .svc files (case-insensitive)
    svc_files = glob.glob(os.path.join(input_dir, '*.svc'))
    svc_files += glob.glob(os.path.join(input_dir, '*.SVC'))
    svc_files.sort()
    
    if not svc_files:
        print(f'No .svc files found in {input_dir}')
        return
    
    for file_path in svc_files:        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, base_name + '.png')
        
        # ensure subfolder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)


        # 读取数据
        data = read_svc_file(file_path)

        # 生成笔迹图像
        plot_handwriting(data, output_path)
        print(f"saved: {output_path}")

if __name__ == '__main__':
    main()



