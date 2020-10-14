import os
import numpy as np
import cv2

class splitbase():
    def __init__(self,
                 srcpath,
                 gap=100,
                 subsize=1024,
                 padding=True):
        self.srcpath = srcpath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.padding = padding

    def SplitSingle(self, name, rate, extent):
        img = cv2.imread(os.path.join(self.srcpath, name + extent))
        assert np.shape(img) != ()

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img

        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        patch_list = []
        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                subimg = img[up: (up + self.subsize), left: (left + self.subsize)]
                h, w, c = np.shape(subimg)
                if (self.padding):
                    outimg = np.zeros((self.subsize, self.subsize, 3))
                    outimg[0:h, 0:w, :] = subimg
                patch_list.append(outimg)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide
        return patch_list