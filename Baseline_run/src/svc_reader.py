import os
import glob
import numpy as np


def read_svc_file(file_path):
    """Read a single .svc file and return as numpy array"""

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
    """Read all .svc files in a folder and return as dictionary of numpy arrays"""
    
    svc_files = glob.glob(os.path.join(file_path, "*.svc")) + glob.glob(os.path.join(file_path, "*.SVC"))
    svc_files.sort()
    
    dataset = {}
    for file in svc_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        dataset[base_name] = read_svc_file(file)
        
    return dataset
        
    
    

