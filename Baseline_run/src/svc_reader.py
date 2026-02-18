import os
import glob
import numpy as np


def read_svc_file(file_path):

    """
    Read a single .svc handwriting file and return its samples as a NumPy array.

    Input
    -----
    file_path : str
        Path to a .svc file. The first line (often the number of points) is skipped.
        Each valid subsequent line must contain 7 whitespace-separated integers:
        x, y, timestamp, pen_status, azimuth, altitude, pressure.

    Output
    ------
    data : np.ndarray of shape (N, 7), dtype=int
        Array of handwriting samples with columns:
        [x, y, timestamp, pen_status, azimuth, altitude, pressure].
        Lines that do not have exactly 7 values are ignored.
    """
    
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # skipping the first line - it is number of data points
            # separating every words
            parts = line.strip().split()
            if len(parts)!= 7:
                continue
            x, y, timestamp, pen_status, azimuth, altitude, pressure = map(int, parts)
            data.append((x, y, timestamp, pen_status, azimuth, altitude, pressure))
    return np.array(data)


def read_all_svc_files(file_path):

    """
    Read all .svc/.SVC files from a directory into a dictionary of NumPy arrays.

    Input
    -----
    file_path : str
        Directory containing .svc (or .SVC) files.

    Output
    ------
    dataset : dict[str, np.ndarray]
        Dictionary mapping each file's base name (filename without extension)
        to a NumPy array of shape (N, 7) produced by `read_svc_file`.
        Example: dataset["sample_001"] -> array([[x, y, t, status, az, alt, p], ...]).
    """
    
    svc_files = glob.glob(os.path.join(file_path, "*.svc")) + glob.glob(os.path.join(file_path, "*.SVC"))
    svc_files.sort()
    
    dataset = {}
    for file in svc_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        dataset[base_name] = read_svc_file(file)
        
    return dataset
        
    
    

