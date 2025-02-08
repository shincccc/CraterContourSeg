import csv
import os
import numpy as np
from osgeo import gdal

def scale_to_ubyte(array):
    """
    Scale the input array to 0-255 range and convert to unsigned byte.
    """
    # Convert array to float for calculations
    array_float = array.astype(np.float32)
    
    # Find the minimum and maximum values in the array
    min_val = np.min(array_float)
    max_val = np.max(array_float)
    
    # Perform linear scaling to map the value range to 0-255
    scaled_array = ((array_float - min_val) / (max_val - min_val)) * 255
    
    # Convert the result to integers
    scaled_array = np.floor(scaled_array).astype(np.uint8)
    
    # Ensure the values are within the 0-255 range
    scaled_array = np.clip(scaled_array, 0, 255)
    
    return scaled_array

def find_row_by_id(csv_file_path, target_id):
    """
    Find a row in the CSV file by matching the 'id' field, 
    and return a list of [lat, long, diam] for that row.
    """
    # Open the CSV file and read the content
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        print(target_id)
        
        # Iterate through each row in the CSV
        for row in reader:
            # If the 'id' column value matches the target_id
            if row['id'] == target_id:
                # Extract the 2nd, 3rd, and 4th columns and return them as a list
                result = [float(row['lat']), float(row['long']), float(row['diam'])]
                return result
    
    return [None, None, None]  # Return [None, None, None] if no matching id is found

def tile_concat(dem_file_path):
    """
    Concatenate DEM tiles into a single array by stacking them.
    """
    # Initialize a list to store the concatenated arrays
    tile_list = []

    # Mapping of tile filenames to their positions
    tile_files = {
        'HMC_13E10_dt5.tif': 'top_left',   # Top-left
        'HMC_13E20_dt5.tif': 'top_right',  # Top-right
        'HMC_13E30_dt5.tif': 'bottom_left',# Bottom-left
        'HMC_13E40_dt5.tif': 'bottom_right' # Bottom-right
    }

    # Load all four tile datasets
    for tile_file, position in tile_files.items():
        tile_path = os.path.join(dem_file_path, tile_file)

        # Open each tile file
        dataset = gdal.Open(tile_path)
        tile_array = dataset.ReadAsArray().astype(np.float16)
        tile_list.append((position, tile_array))

    # Concatenate the tiles in a "田" (field) shape pattern (vertical and horizontal stacking)
    
    # 1. Vertical stacking: Top-left and Bottom-left
    top_tile = tile_list[0][1]  # Top-left
    bottom_tile = tile_list[2][1]  # Bottom-left
    left_column = np.vstack((top_tile, bottom_tile))  # Vertical stacking

    # 2. Vertical stacking: Top-right and Bottom-right
    top_tile = tile_list[1][1]  # Top-right
    bottom_tile = tile_list[3][1]  # Bottom-right
    right_column = np.vstack((top_tile, bottom_tile))  # Vertical stacking

    # 3. Horizontal stacking: Combine left and right columns
    all_tiles_array = np.hstack((left_column, right_column))  # Horizontal stacking

    return all_tiles_array
