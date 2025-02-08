import os
import cv2
import numpy as np
from scipy.optimize import minimize
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from osgeo import gdal
import csv
from utils import scale_to_ubyte, find_row_by_id, tile_concat
from contour_seg_file_de import enhance_dem_array, get_polys,  draw_best_polys, get_crater_contours
from reposition import reposition

def get_crater_contours_fix_h(img_array, optimal_h, edge_margin):
    h, w = img_array.shape
    enhanced_array = enhance_dem_array(img_array, optimal_h)
    contours = get_polys(enhanced_array, h, w, edge_margin)
    return enhanced_array, contours

if __name__ == "__main__":
    left_up_dem_tile_path = "./DEM/Isidis/HMC_13E10_dt5.tif"
    dem_file_path = "./DEM/Isidis"
    csv_file_path = "./catalog/Isidis_3km+.csv"
    output_folder = "./experiment/pos"
    pos_bia_num = 0

    # 处理tile文件
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 左上角的tile
    dataset = gdal.Open(left_up_dem_tile_path)
    tile_array = dataset.ReadAsArray()
    tile_width = dataset.RasterXSize
    tile_height = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    top_left_x = int(geotransform[0])
    top_left_y = int(geotransform[3])
    # 整个研究区域
    all_tiles_array = tile_concat(dem_file_path)
    bottom_left_x = int(top_left_x)
    bottom_left_y = int(top_left_y - int(tile_height * 2 * 50))

    # 边缘提取目标函数的权重参数
    w_irregularity = 1
    w_bound = 1
    w_volume = 1  # 添加体积的权重
    w_compactness = 1
    edge_margin = 1

    resolution = 50

    # 对一个撞击坑编目进行轮廓提取与位置修正
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # 从表格中获取撞击坑的坐标与直径
            crater_id = row['id']
            y_tile, x_tile, diam = find_row_by_id(csv_file_path, crater_id)
            diam = int(diam * 20)
            y_tile = (y_tile - bottom_left_y) // resolution
            y_tile = int(2 * tile_height - y_tile)
            x_tile = (x_tile - bottom_left_x) // resolution
            y_min = max(int(y_tile - 0.55 * diam), 0)
            y_max = min(int(y_tile + 0.55 * diam), 2 * tile_height)
            x_min = max(int(x_tile - 0.55 * diam), 0)
            x_max = min(int(x_tile + 0.55 * diam), 2 * tile_width)
            img_array = all_tiles_array[y_min:y_max, x_min:x_max]

            print("id:", crater_id)

            # 提取轮廓
            enhanced_array, contours, optimal_h = get_crater_contours(img_array, w_irregularity, w_bound, w_volume,
                                                                      w_compactness, edge_margin)

            if len(contours) > 0:
                # 保存初步结果
                output_img_path = output_folder + f'/{crater_id}.jpg'
                output_contour_path = output_folder + f'/{crater_id}_contour.jpg'
                img = Image.fromarray(scale_to_ubyte(img_array))
                img.save(output_img_path)
                draw_best_polys(enhanced_array, contours, output_contour_path)
                # 提取轮廓列表
                outside_contour = np.squeeze(contours[0], axis=1)
            else:
                # 错误撞击坑或错误图像
                img.save(output_folder + '/wrong' + f'/{crater_id}.jpg')
                continue

            # 修正撞击坑的位置与尺寸
            h, w = np.shape(enhanced_array)
            diam = (diam*1000)//resolution
            is_touched, reposition_array, reposition_enhanced_array, new_x_tile, new_y_tile, new_h, new_w, new_diam = reposition(
                all_tiles_array, tile_height,
                tile_width, crater_id, x_tile, y_tile, y_min, y_max, x_min, x_max, outside_contour, h, w,
                optimal_h, enhanced_array, max_iterations=5)

            # 处理位置修正结果
            if is_touched == 1 and reposition_array is not None and reposition_enhanced_array is not None:
                pos_bia_num += 1
                # 单独保存修正前的结果
                img.save(output_folder + '/pos_bia/' + f'/{crater_id}.jpg')
                # 单独保存修正后的结果
                output_reposition_img_path = os.path.join(output_folder + '/reposition/', f"processed_{crater_id}.jpg")
                output_enhanced_img_path = os.path.join(output_folder + '/reposition/', f"processed_{crater_id}_eh.jpg")
                # reposition_array = scale_to_ubyte(reposition_array)
                # print("reposition_array.shape:", reposition_array.shape)
                reposition_img = Image.fromarray(reposition_array)
                reposition_enhanced_img = Image.fromarray(reposition_enhanced_array)
                if reposition_img.size == (0, 0):
                    print("图像为空，无法保存")
                else:
                    reposition_img.save(output_reposition_img_path)
                    reposition_enhanced_img.save(output_enhanced_img_path)
                    print(f"Processed image saved to {output_img_path}")

        print("pos_bia_num:", pos_bia_num)





