# coding=utf-8

import numpy as np
from pathlib import Path

def Read_calib(calib_file):
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

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


class Calibration(object):
    def __init__(self, calib_file):
        if Path(calib_file).exists():
            calib = Read_calib(calib_file)
        else:
            print("Calib file doesn't exist.")

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

    def cart_to_hom(self, pts):
        """
        @param pts: (N, 3 or 2)
        @return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        @param pts_lidar: (N, 3)
        @return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar, img_num='P2'):
        """
        @param pts_lidar: (N, 3)
        @return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        # return pts_img, pts_depth

        pts_hom = self.cart_to_hom(pts_lidar)
        if img_num == 'P2':
            pts2 = self.cart_to_hom(pts_lidar)
            rect_img = np.dot(np.dot(self.R0, self.V2C), pts2.T)
            pts_img2 = np.dot(self.P2, self.cart_to_hom(rect_img.T).T)
        return pts_img, pts_depth

if __name__=='__main__':
    pts = np.array([[1,2,3],[3,2,1]])
    cal = Calibration('000008.txt')
    result,_ = cal.lidar_to_img(pts)
    print(result)

