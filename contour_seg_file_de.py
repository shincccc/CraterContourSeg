import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

def enhance_dem_array(array, n):
    # Ensure input is 2D
    if len(array.shape) != 2:
        array = array[:, :, 0]
    # img = Image.fromarray(array)
    # img.show()

    # Calculate max, min, and _h
    max_val = np.max(array)
    min_val = np.min(array)
    _h = max_val - min_val
    a = _h / n

    # Apply modulus and normalize
    new_array = array % a
    norm_array = (new_array - np.min(new_array)) / (np.max(new_array) - np.min(new_array)) * 255
    norm_array = norm_array.astype(np.uint8)
    # img_e_array = Image.fromarray(norm_array)
    # img_e_array.show()
    return norm_array

def get_polys(enhanced_image, h, w, edge_margin):
    # Dynamic thresholding using Otsu's method
    _, thresh = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_OTSU)

    # plt.figure(figsize=(8, 6))
    # plt.imshow(thresh, cmap='gray')
    # plt.title(f"Binary Image (T={_})")
    # plt.axis('off')
    # plt.show()

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = [cnt for cnt in contours if not contains_edge_point(cnt, h, w, edge_margin)]
    contours = [cnt for cnt in contours if calculate_boundary_proportion(cnt, w, h, threshold=3)<=0.8]

    polygons = []
    for contour in contours:
        # Reduce epsilon to preserve more detail
        epsilon = 0.001 * cv2.arcLength(contour, True)  # Try smaller epsilon
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True)
        polygons.append(polygon)
    # Sort polygons by area
    polygons_sorted = sorted(polygons, key=cv2.contourArea, reverse=True)
    return polygons_sorted

def calculate_boundary_proportion(polygon, img_width, img_height, threshold=1):

    boundary_points = 0
    total_points = len(polygon)

    for point in polygon:
        x, y = point[0]
        # 判断该点是否接近图像边界
        if x < threshold or y < threshold or x > img_width - threshold or y > img_height - threshold:
            boundary_points += 1

    # 如果多边形的总点数为0，返回0避免除以0错误
    if total_points == 0:
        return 0

    # 计算边界点的比例
    proportion = boundary_points / total_points
    return proportion

def draw_best_polys(enhanced_image, polygons_sorted, output_img_path):
    # Visualize the polygons on the image
    img_with_polys = np.copy(enhanced_image)
    img_with_polys = cv2.cvtColor(img_with_polys, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels for color display
    #img_with_polys_copy = np.copy(img_with_polys)
    largest_polygon = [polygons_sorted[0]]

    # 所有的轮廓
    # cv2.drawContours(img_with_polys_copy, polygons_sorted, -1, (0, 255, 0), 2)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(img_with_polys_copy)
    # plt.title("All Contours")
    # plt.axis('off')
    # plt.show()

    #最优的轮廓
    cv2.drawContours(img_with_polys, largest_polygon, -1, (0, 255, 0), 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_with_polys)
    plt.title("Best Contour")
    plt.axis('off')
    #plt.show()

    cv2.imwrite(output_img_path, img_with_polys)

    # largest_polygon_1 = [polygons_sorted[1]]
    # cv2.drawContours(img_with_polys, largest_polygon_1, -1, (0, 255, 0), 3)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(img_with_polys)
    # plt.title("Polygons Detected")
    # plt.axis('off')
    # plt.show()

def objective_function(num, img_array, w_irregularity, w_bound, w_volume, w_compactness, edge_margin, w, h):

    norm_array = enhance_dem_array(img_array, num)
    polygons = get_polys(norm_array, h, w, edge_margin)

    if not polygons:
        return float('inf')  # 没有检测到多边形，返回很大的值

    # 计算多边形的最大面积
    max_area = cv2.contourArea(polygons[0])
    perimeter = cv2.arcLength(polygons[0], True)
    # 计算不规则性：总面积 / 凸包面积
    convex_hull = cv2.convexHull(polygons[0])
    hull_area = cv2.contourArea(convex_hull)
    if hull_area == 0:
        irregularity = float('inf')  # 防止除零错误
        SI = float('inf')
    else:
        irregularity = hull_area / max_area
        SI = pow(perimeter, 2) / (max_area * 4 * np.pi)

    # 计算边界点的比例作为惩罚项
    p = 5
    boundary_penalty = calculate_boundary_proportion(polygons[0], w, h, threshold=1)
    if boundary_penalty <= 0.2:
        boundary_penalty = 0
    else:
        boundary_penalty = p*boundary_penalty

    # 计算多边形轮廓的平均高度 h_rim
    mask = np.zeros_like(img_array, dtype=np.uint8)
    cv2.drawContours(mask, [polygons[0]], -1, 255, thickness=1)  # 绘制轮廓
    contour_pixels = img_array[mask == 255]
    if len(contour_pixels) > 0:
        h_rim = np.mean(contour_pixels)  # 计算轮廓处的平均高程
    else:
        h_rim = 0  # 如果轮廓处没有有效像素，设为0

    # 计算多边形内部的体积 v（差值的累加）
    mask_fill = np.zeros_like(img_array, dtype=np.uint8)
    cv2.drawContours(mask_fill, [polygons[0]], -1, 255, thickness=cv2.FILLED)  # 填充多边形
    poly_pixels = img_array[mask_fill == 255]
    # 计算体积：多边形内部的像素值与轮廓平均值的差值
    if len(poly_pixels) > 0:
        # 计算灌满所需的体积
        filled_volume = np.sum(np.maximum(0, h_rim - poly_pixels))  # 计算体积
        # 计算 poly_pixels 区域的像素个数
        pixel_count = len(poly_pixels)
        # 用像素个数乘以 h_rim 得到总值
        total_volume = pixel_count * h_rim
        # 计算最终的比例值
        fv =  total_volume / filled_volume
    else:
        filled_volume = 0  # 如果没有有效像素，体积为0
        fv = 0  # 如果没有有效像素，比例值为0

    # 总目标函数
    objective_value = fv * w_volume + SI * w_compactness + irregularity * w_irregularity + boundary_penalty
    return objective_value

def get_crater_contours(img_array, w_irregularity, w_bound, w_volume, w_compactness, edge_margin):
    h, w = img_array.shape
    # Perform optimization to find the optimal value of `num`
    result = differential_evolution(objective_function, bounds=[(2, 7)],
                                    args=(
                                    img_array, w_irregularity, w_bound, w_volume, w_compactness, edge_margin, w, h),
                                    strategy='best1bin', maxiter=15, popsize=3, recombination=0.5)
    optimal_h = result.x[0]  # Optimal value for `h`
    print(f"Optimal_h: {optimal_h}")
    #Enhance the DEM array using the optimal `h`
    enhanced_array = enhance_dem_array(img_array, optimal_h)
    # Get the polygons detected in the enhanced image
    contours = get_polys(enhanced_array, h, w, edge_margin)
    return enhanced_array, contours, optimal_h

if __name__ == "__main__":
    file_path = r"D:\crater dats\论文样本"
    output_folder = r"D:\crater dats\experiment\ablation\V+SI+I+R"  # 新的文件夹保存处理后的图片
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #边缘提取目标函数的权重参数
    w_irregularity = 1
    w_bound = 1
    w_volume = 1
    w_compactness = 1
    edge_margin = 1

    for filename in os.listdir(file_path):
        img_path = os.path.join(file_path, filename)
        img = Image.open(img_path)
        img_array = np.array(img)
        enhanced_array, contours, optimal_h = get_crater_contours(img_array, w_irregularity, w_bound, w_volume, w_compactness, edge_margin)
        #保存轮廓图
        output_img_path = os.path.join(output_folder, f"processed_{filename}")
        draw_best_polys(enhanced_array, contours, output_img_path)
        #修正撞击坑的位置与尺寸
        outside_contour = np.squeeze(contours[0], axis=1)

        print(f"Processed image saved to {output_img_path}")
