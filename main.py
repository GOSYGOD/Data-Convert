# coding=utf-8

import os
from pathlib import Path
import argparse
import numpy as np

def Parse_config():

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--file_path', type=str, default=None, help='specify the folder path')

    args = parser.parse_args()
    # args.file_path = '.'
    return args






def main():
    args = Parse_config()
    lidar_file = '000008.bin'
    calib_file = '000008.txt'
    image_file = '000008.png'

    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32).reshape(3, 4)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32).reshape(3, 4)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32).reshape(3, 3)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32).reshape(3, 4)


    points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    R0 = np.concatenate((R0,np.array([0,0,0]).reshape(1,3)), axis=0)
    R0 = np.concatenate((R0,np.array([0,0,0,1]).reshape(4,1)), axis=1)

    Tr_velo_to_cam = np.concatenate((Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)), axis=0)

    x = points[0,:].reshape(-1,1)
    x[3] = 1
    r1 = np.matmul(P2, R0)
    r2 = np.matmul(r1, Tr_velo_to_cam)
    image_points = np.matmul(r2, x)
    image_points = (image_points / image_points[-1]).astype(int)



    a = 1




if __name__ == '__main__':

    main()

