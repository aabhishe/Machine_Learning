# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 15:00:30 2018
reading image in and filteing RGB from a color image
@author: alok_
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
Christ_the_Redeemer_pic = Image.open('C:/Users/alok_/OneDrive/Machine Learning/Machine_Learning/Python/Neural Net/Christ_the_Redeemer.jpg')
plt.imshow(Christ_the_Redeemer_pic)

Christ_the_Redeemer_pic_arr = np.asarray(Christ_the_Redeemer_pic)
Christ_the_Redeemer_pic_arr.shape

Christ_the_Redeemer_pic_red = Christ_the_Redeemer_pic_arr.copy()

Christ_the_Redeemer_pic_red[:, :, 1] = 0    # setting green pixels to zero
Christ_the_Redeemer_pic_red[:, :, 2] = 0    # setting blue pixels to zero

plt.imshow(Christ_the_Redeemer_pic_red)

Christ_the_Redeemer_pic_blue = Christ_the_Redeemer_pic_arr.copy()

Christ_the_Redeemer_pic_blue[:, :, 0] = 0    # setting red pixels to zero
Christ_the_Redeemer_pic_blue[:, :, 1] = 0    # setting green pixels to zero

plt.imshow(Christ_the_Redeemer_pic_blue)

Christ_the_Redeemer_pic_green = Christ_the_Redeemer_pic_arr.copy()

Christ_the_Redeemer_pic_green[:, :, 0] = 0    # setting red pixels to zero
Christ_the_Redeemer_pic_green[:, :, 2] = 0    # setting blue pixels to zero

plt.imshow(Christ_the_Redeemer_pic_green)



