# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 08:15:34 2022

@author: Hugo Poinard
"""

import pandas as pd
import numpy as np
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import cv2 as cv

img_rgb = cv.imread('sheet2.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)


template = cv.imread('diese sheet.png',0)
#template = cv.resize(template, (10,31), interpolation = cv.INTER_AREA)
w, h = template.shape[::-1]

res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.6
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)

cv.imshow('Résultat',img_rgb)