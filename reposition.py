import numpy as np
from utils import scale_to_ubyte
from contour_seg_file_de import enhance_dem_array, get_polys,  draw_best_polys, get_crater_contours

def calculate_centroids(contour, h, w):
    if len(contour) > 0:
        contour_x = [point[0] for point in contour]
        contour_y = [point[1] for point in contour]
        centroid_x = sum(contour_x) / len(contour_x)
        centroid_y = sum(contour_y) / len(contour_y)
        centroids = [centroid_x, centroid_y]
    else:
        centroids = [w // 2, h // 2]
    return centroids

def touching_edge_nums(contour, image_width, image_height, threshold=0.2):
    touched_edges = []

    # 计算接触上边界的点数
    top_contact = sum(1 for x, y in contour if 0 <= y <= 2)
    top_ratio = top_contact / image_width if image_width > 0 else 0
    if top_ratio >= threshold:  # 判断接触比例是否超过阈值
        touched_edges.append("top")

    # 计算接触下边界的点数
    bottom_contact = sum(1 for x, y in contour if (image_height - 3) <= y <= (image_height - 1))
    bottom_ratio = bottom_contact / image_width if image_width > 0 else 0
    if bottom_ratio >= threshold:  # 判断接触比例是否超过阈值
        touched_edges.append("bottom")

    # 计算接触左边界的点数
    left_contact = sum(1 for x, y in contour if 0 <= x <= 2)
    left_ratio = left_contact / image_height if image_height > 0 else 0
    if left_ratio >= threshold:  # 判断接触比例是否超过阈值
        touched_edges.append("left")

    # 计算接触右边界的点数
    right_contact = sum(1 for x, y in contour if (image_width - 3) <= x <= (image_width - 1))
    right_ratio = right_contact / image_height if image_height > 0 else 0
    if right_ratio >= threshold:  # 判断接触比例是否超过阈值
        touched_edges.append("right")

    # 返回与边界相交的边和相交的边数
    return touched_edges, len(touched_edges)

def expand_edges(expanded_img_array, tile_array, tile_h, tile_w, x_tile, y_tile, h, w, centroid, touched_edges,
                 padding):
    img_center = (w // 2, h // 2)  # 假设图像中心在(width//2, height//2)
    x_left_bias = 0
    x_right_bias = 0
    y_top_bias = 0
    y_bottom_bias = 0

    # 左右边界触碰情况
    if "left" in touched_edges and "right" in touched_edges:
        x_right_bias = padding
        x_left_bias = padding
    else:
        if "left" in touched_edges:
            x_left_bias = img_center[0] - centroid[0]  # +
            if x_left_bias < 0:
                x_left_bias = 0

        elif "right" in touched_edges:
            x_right_bias = centroid[0] - img_center[0]
            if x_right_bias < 0:
                x_right_bias = 0

    # 上下边界触碰情况
    if "top" in touched_edges and "bottom" in touched_edges:
        y_top_bias = padding
        y_bottom_bias = padding
    else:
        if "top" in touched_edges:
            y_top_bias = img_center[1] - centroid[1]
            if y_top_bias < 0:
                y_top_bias = 0
        elif "bottom" in touched_edges:
            y_bottom_bias = centroid[1] - img_center[1]
            if y_bottom_bias < 0:
                y_bottom_bias = 0

    # 定义图框坐标
    # 原图框坐标
    lu_x_or = x_tile - w // 2
    lu_y_or = y_tile - h // 2
    rb_x_or = x_tile + w // 2
    rb_y_or = y_tile + h // 2

    # 新图框坐标
    lu_x = max(round(lu_x_or - x_left_bias), 0)
    lu_y = max(round(lu_y_or - y_top_bias), 0)
    rb_x = min(round(rb_x_or + x_right_bias), 2 * tile_w - 1)
    rb_y = min(round(rb_y_or + y_bottom_bias), 2 * tile_h - 1)

    # print("tile_array.shape:", tile_array.shape)
    expanded_img_array = tile_array[lu_y:rb_y, lu_x:rb_x]

    new_x_tile = (lu_x + rb_x) // 2
    new_y_tile = (lu_y + rb_y) // 2
    expand_w = rb_x - lu_x
    expand_h = rb_y - lu_y

    return new_x_tile, new_y_tile, expand_h, expand_w, expanded_img_array

def reposition_bound(expanded_img_array, expanded_enhanced_array, expand_h, expand_w, new_contour, lu_x, lu_y, rb_x,
                     rb_y):
    print("expanded_img_array.shape:", expanded_img_array.shape)
    # Step 1: Find the min and max x, y coordinates of the contour
    contour_x = [point[0] for point in new_contour]
    contour_y = [point[1] for point in new_contour]

    min_x = min(contour_x)
    max_x = max(contour_x)
    min_y = min(contour_y)
    max_y = max(contour_y)

    # Step 2: Add 10% extra space around the bounding box
    width_expansion = 0.1 * (max_x - min_x)
    height_expansion = 0.1 * (max_y - min_y)

    new_min_x = int(min_x - width_expansion)
    new_max_x = int(max_x + width_expansion)
    new_min_y = int(min_y - height_expansion)
    new_max_y = int(max_y + height_expansion)

    # Step 3: Ensure the new bounding box does not exceed the original image bounds
    new_min_x = max(new_min_x, 0)
    new_max_x = min(new_max_x, expand_w)
    new_min_y = max(new_min_y, 0)
    new_max_y = min(new_max_y, expand_h)

    # Step 4: Crop the image based on the new bounding box
    reposition_enhanced_array = expanded_enhanced_array[new_min_y:new_max_y, new_min_x:new_max_x]
    reposition_img_array = expanded_img_array[new_min_y:new_max_y, new_min_x:new_max_x]

    # Step 5: Calculate the centroid of the contour in the local image
    centroid_x = sum(contour_x) / len(contour_x)
    centroid_y = sum(contour_y) / len(contour_y)

    # Step 6: Calculate the centroid in the larger satellite image coordinates
    new_x_tile = lu_x + (centroid_x * (rb_x - lu_x) / expand_w)
    new_y_tile = lu_y + (centroid_y * (rb_y - lu_y) / expand_h)

    # Step 7: Calculate the new height and width of the cropped image
    new_h = new_max_y - new_min_y
    new_w = new_max_x - new_min_x

    return reposition_img_array, reposition_enhanced_array, new_x_tile, new_y_tile, new_h, new_w

def reposition(tile_array, tile_h, tile_w, crater_id, x_tile, y_tile, y_min, y_max, x_min, x_max, polygon, h, w,
               optimal_h, enhanced_array, max_iterations=10):
    padding = ((h + w) / 2) // 10  # 每次扩展直径的1/20
    edge_margin = 1
    is_touched = 0
    expanded_enhanced_array = None
    expanded_img_array = None

    # 通过循环，拓展撞击坑窗口
    for i in range(max_iterations):
        print("epoch:", i)

        # 质心
        centroid = calculate_centroids(polygon, h, w)

        # 计算图框接触
        touched_edges, num_touched = touching_edge_nums(polygon, w, h, threshold=0.1)
        if num_touched == 0:
            break
        else:
            is_touched = 1

        # 返回拓宽后的图像矩阵
        new_x_tile, new_y_tile, expand_h, expand_w, expanded_img_array = expand_edges(expanded_img_array, tile_array,
                                                                                      tile_h, tile_w, x_tile, y_tile, h,
                                                                                      w, centroid, touched_edges,
                                                                                      padding)
        # 归一化
        if np.shape(expanded_img_array)[0] > 0:
            expanded_img_array = scale_to_ubyte(expanded_img_array)
        else:
            break

        # 更新宽高和中心坐标
        x_tile = new_x_tile
        y_tile = new_y_tile
        h = expand_h
        w = expand_w

        # 更新轮廓
        # expanded_enhanced_array, contours = get_crater_contours_fix_h(expanded_img_array, optimal_h, edge_margin)
        expanded_enhanced_array, contours, optimal_h = get_crater_contours(expanded_img_array, w_irregularity=1,
                                                                           w_bound=1, w_volume=1, w_compactness=1,
                                                                           edge_margin=edge_margin)

        if len(contours) > 0:
            polygon = np.squeeze(contours[0], axis=1)
        else:
            continue

        print(
            f"Iteration {i + 1}: Touched edges {touched_edges}")

    if expanded_enhanced_array is not None and expanded_img_array is not None:
        # 拓宽后 enhanced_array polygon
        #   通过计算轮廓块的质心与轮廓的边界，重新生成紧致的窗口, 得到新的中心坐标new_x, new_y, x_bia, y_bia, new_h, new_w
        reposition_img_array, reposition_array, new_x_tile, new_y_tile, new_h, new_w = reposition_bound(
            expanded_img_array, expanded_enhanced_array, expand_h, expand_w, polygon, x_min, y_max, x_max, y_min)
        new_diam = (new_h + new_w) // 2

        return is_touched, reposition_img_array, reposition_array, new_x_tile, new_y_tile, new_h, new_w, new_diam
    else:
        return is_touched, expanded_img_array, expanded_enhanced_array, x_tile, y_tile, h, w, diam
