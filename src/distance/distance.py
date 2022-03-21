import os
import numpy as np

def load_top_view_config(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        M = np.zeros((3, 3))
        for i in range(3):
            M[i] = np.array(lines[i].split(', '))

        w_scale = float(lines[3])
        h_scale = float(lines[4])
    
    # print('M:', M)
    # print('w_scale:', w_scale)
    # print('h_scale:', h_scale)
    return M, w_scale, h_scale

if __name__ == '__main__':
    # load_top_view_config('../config/top_view_config.txt')
    load_top_view_config('src/distance/distance.txt')
