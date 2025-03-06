import cv2
import numpy as np


class VAP_Vein():
    def __init__(self, id=0):
        self.idn = id
        self.length = 1
        self.vap_point_list = []
        self.mid_point = None
        self.tip_points = []
        self.branch_points = []

    def Build_Up(self):

        self.mid_point = self.vap_point_list[int(len(self.vap_point_list) / 2)]

        p_list = [[p.y, p.x] for p in self.vap_point_list] # remove branch points
        if (len(p_list)>0):
            my_array = np.array(p_list)
            self.length= float(cv2.arcLength(my_array, False))


if __name__ == "__main__":
    plist= [[0,0],[1,1],[2,2]] # düz piksel 1, çapraz piksel 1.414 mutlaka iki piksel olması gerekiyor.
    my_array = np.array(plist)
    print(float(cv2.arcLength(my_array, False)))