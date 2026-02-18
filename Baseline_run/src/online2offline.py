import numpy as np
import matplotlib.pyplot as plt

import os
import glob


def read_svc_file(file_path):
    
    """
    Read a single .svc handwriting file and return its samples as a NumPy array.

    Input
    -----
    file_path : str
        Path to a .svc file. The first line is ignored. Each subsequent valid line
        must contain 7 whitespace-separated integers:
        (x, y, timestamp, pen_status, azimuth, altitude, pressure).

    Output
    ------
    data : np.ndarray of shape (N, 7), dtype=int
        Array of samples with columns:
        [x, y, timestamp, pen_status, azimuth, altitude, pressure].
        Lines that do not have exactly 7 values are skipped.
    """

    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  
            parts = line.strip().split()
            if len(parts)!= 7:
                continue
            x, y, timestamp, pen_status, azimuth, altitude, pressure = map(int, parts)
            data.append((x, y, timestamp, pen_status, azimuth, altitude, pressure))
    return np.array(data)



def plot_handwriting(data, output_path):

    """
    Render a handwriting trajectory to a PNG image using pen pressure for stroke width.

    Input
    -----
    data : np.ndarray of shape (N, 7)
        Handwriting samples in the format:
        [x, y, timestamp, pen_status, azimuth, altitude, pressure].
        pen_status: 1 = pen down (draw), 0 = pen up (lift).
        pressure is assumed to be in [0, 1023] for linewidth scaling.

    output_path : str
        File path to save the output image (e.g., .png).

    Output
    ------
    None
        Saves a PNG image to `output_path`. Does not return a value.

    Notes
    -----
    - The trajectory is rotated 90° counterclockwise: (x, y) -> (-y, x).
    - Consecutive pen-down points are connected with line segments.
    - Line width is scaled by pressure: 0.1 + (pressure/1023)*5.
    """


    plt.figure(figsize=(10, 10))

    prev_x, prev_y = None, None
    current_line = []

    for x, y, _, pen_status, _, _, pressure in data:
        rotated_x = -y  
        rotated_y = x    

        if pen_status == 1: 
            if prev_x is not None and prev_y is not None:
                line_width = 0.1 + (pressure / 1023) * 5  
                plt.plot([prev_x, rotated_x], [prev_y, rotated_y], color='black', linewidth=line_width, solid_capstyle='round')
            prev_x, prev_y = rotated_x, rotated_y
        else:  
            prev_x, prev_y = None, None

    plt.axis('off')  
    plt.gca().set_aspect('equal', adjustable='box') 

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    
    """
    Convert all .svc files in a folder into pressure-aware handwriting PNG images.

    Input
    -----
    Uses hard-coded paths:
      - input_dir  = './Dataset/house'
      - output_dir = './image/house'

    Output
    ------
    None
        For each .svc/.SVC file found in input_dir, saves a corresponding .png
        image into output_dir and prints the saved file path.
    """

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


        data = read_svc_file(file_path)

        plot_handwriting(data, output_path)
        print(f"saved: {output_path}")

if __name__ == '__main__':
    main()



