import csv
import os
import numpy as np
from osgeo import gdal

def scale_to_ubyte(array):
    # 将数组转换为浮点数，以进行计算
    array_float = array.astype(np.float32)
    # 找到数组的最小值和最大值
    min_val = np.min(array_float)
    max_val = np.max(array_float)
    # 进行线性缩放，将数值范围映射到 0 到 255
    scaled_array = ((array_float - min_val) / (max_val - min_val)) * 255
    # 将结果转换为整数
    scaled_array = np.floor(scaled_array).astype(np.uint8)
    # 确保数值在 0 到 255 的范围内
    scaled_array = np.clip(scaled_array, 0, 255)
    return scaled_array

def find_row_by_id(csv_file_path, target_id):
    # 打开CSV文件并读取内容
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        print(target_id)
        # 遍历CSV中的每一行
        for row in reader:
            # 如果'ID'列的值匹配
            if row['id'] == target_id:
                # 提取第2、3、4列的值，并合成一个列表
                result = [float(row['lat']), float(row['long']), float(row['diam'])]
                return result
    return [None, None, None]  # 如果没有找到对应的id

def tile_concat(dem_file_path):
    # 初始化存储拼接后数组的列表
    tile_list = []

    # 文件名与它们的位置信息对应
    tile_files = {
        'HMC_13E10_dt5.tif': 'top_left',  # 左上
        'HMC_13E20_dt5.tif': 'top_right',  # 右上
        'HMC_13E30_dt5.tif': 'bottom_left',  # 左下
        'HMC_13E40_dt5.tif': 'bottom_right'  # 右下
    }

    # 加载所有四个 tile 数据
    for tile_file, position in tile_files.items():
        tile_path = os.path.join(dem_file_path, tile_file)

        # 打开每个tile文件
        dataset = gdal.Open(tile_path)
        tile_array = dataset.ReadAsArray().astype(np.float16)
        tile_list.append((position, tile_array))

    # 按照田字型顺序拼接数组
    # 假设拼接的方向是上下拼接（竖向）和左右拼接（横向）

    # 1. 垂直拼接: 左上和左下拼接
    top_tile = tile_list[0][1]  # 左上
    bottom_tile = tile_list[2][1]  # 左下
    left_column = np.vstack((top_tile, bottom_tile))  # 垂直拼接

    # 2. 垂直拼接: 右上和右下拼接
    top_tile = tile_list[1][1]  # 右上
    bottom_tile = tile_list[3][1]  # 右下
    right_column = np.vstack((top_tile, bottom_tile))  # 垂直拼接

    # 3. 水平拼接: 将左列与右列拼接
    all_tiles_array = np.hstack((left_column, right_column))  # 水平拼接

    return all_tiles_array