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

    # Calculate max, min, and _h
    max_val = np.max(array)
    min_val = np.min(array)
    _h = max_val - min_val
    a = _h / n

    # Apply modulus and normalize
    new_array = array % a
    norm_array = (new_array - np.min(new_array)) / (np.max(new_array) - np.min(new_array)) * 255
    norm_array = norm_array.astype(np.uint8)
    return norm_array

def get_polys(enhanced_image, h, w, edge_margin):
    # Dynamic thresholding using Otsu's method
    _, thresh = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if calculate_boundary_proportion(cnt, w, h, threshold=3) <= 0.8]

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
        # Check if the point is near the image boundary
        if x < threshold or y < threshold or x > img_width - threshold or y > img_height - threshold:
            boundary_points += 1

    # Avoid division by zero if total points is 0
    if total_points == 0:
        return 0

    # Calculate the proportion of boundary points
    proportion = boundary_points / total_points
    return proportion

def draw_best_polys(enhanced_image, polygons_sorted, output_img_path):
    # Visualize the polygons on the image
    img_with_polys = np.copy(enhanced_image)
    img_with_polys = cv2.cvtColor(img_with_polys, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels for color display
    largest_polygon = [polygons_sorted[0]]

    # Draw the best contour
    cv2.drawContours(img_with_polys, largest_polygon, -1, (0, 255, 0), 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_with_polys)
    plt.title("Best Contour")
    plt.axis('off')

    cv2.imwrite(output_img_path, img_with_polys)

def objective_function(num, img_array, w_irregularity, w_bound, w_volume, w_compactness, edge_margin, w, h):
    norm_array = enhance_dem_array(img_array, num)
    polygons = get_polys(norm_array, h, w, edge_margin)

    if not polygons:
        return float('inf')  # Return a large value if no polygons are detected

    # Calculate the maximum area of the polygon
    max_area = cv2.contourArea(polygons[0])
    perimeter = cv2.arcLength(polygons[0], True)
    # Calculate irregularity: total area / convex hull area
    convex_hull = cv2.convexHull(polygons[0])
    hull_area = cv2.contourArea(convex_hull)
    if hull_area == 0:
        irregularity = float('inf')  # Avoid division by zero
        SI = float('inf')
    else:
        irregularity = hull_area / max_area
        SI = pow(perimeter, 2) / (max_area * 4 * np.pi)

    # Calculate boundary point proportion as a penalty term
    boundary_penalty = calculate_boundary_proportion(polygons[0], w, h, threshold=1)

    # Calculate the average height at the polygon's boundary (h_rim)
    mask = np.zeros_like(img_array, dtype=np.uint8)
    cv2.drawContours(mask, [polygons[0]], -1, 255, thickness=1)  # Draw the contour
    contour_pixels = img_array[mask == 255]
    if len(contour_pixels) > 0:
        h_rim = np.mean(contour_pixels)  # Calculate average elevation at the contour
    else:
        h_rim = 0  # Set to 0 if no valid pixels are found

    # Calculate the volume inside the polygon (v)
    mask_fill = np.zeros_like(img_array, dtype=np.uint8)
    cv2.drawContours(mask_fill, [polygons[0]], -1, 255, thickness=cv2.FILLED)  # Fill the polygon
    poly_pixels = img_array[mask_fill == 255]
    # Calculate volume: difference between internal pixel values and average boundary height
    if len(poly_pixels) > 0:
        # Calculate the filled volume
        filled_volume = np.sum(np.maximum(0, h_rim - poly_pixels))  # Calculate volume
        # Calculate the number of pixels inside the polygon
        pixel_count = len(poly_pixels)
        # Multiply the number of pixels by h_rim to get the total volume
        total_volume = pixel_count * h_rim
        # Calculate the final ratio
        fv = total_volume / filled_volume
    else:
        filled_volume = 0  # Set to 0 if no valid pixels
        fv = 0  # Set the ratio to 0 if no valid pixels

    # Total objective function value
    objective_value = fv * w_volume + SI * w_compactness + irregularity * w_irregularity + boundary_penalty
    return objective_value

def get_crater_contours(img_array, w_irregularity, w_bound, w_volume, w_compactness, edge_margin):
    h, w = img_array.shape
    # Perform optimization to find the optimal value of `num`
    result = gp_minimize(
        objective_wrapper,
        dimensions=bounds,
        acq_func='EI',
        n_calls=15,
        n_initial_points=5,
        random_state=42,
        x0=[4.0]
    )
    optimal_h = result.x[0]  # Optimal value for `h`
    print(f"Optimal_h: {optimal_h}")
    # Enhance the DEM array using the optimal `h`
    enhanced_array = enhance_dem_array(img_array, optimal_h)
    # Get the polygons detected in the enhanced image
    contours = get_polys(enhanced_array, h, w, edge_margin)
    return enhanced_array, contours, optimal_h

if __name__ == "__main__":
    file_path = r"D:\crater dats\论文样本"
    output_folder = r"D:\crater dats\experiment\ablation\V+SI+I+R"  # New folder to save processed images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Weight parameters for edge extraction objective function
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
        # Save the contour image
        output_img_path = os.path.join(output_folder, f"processed_{filename}")
        draw_best_polys(enhanced_array, contours, output_img_path)
        # Correct the position and size of the crater
        outside_contour = np.squeeze(contours[0], axis=1)

        print(f"Processed image saved to {output_img_path}")
