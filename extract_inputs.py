# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 09:08:50 2022

@author: Hugo Poinard
"""

import pandas as pd
import numpy as np
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import cv2 


"""Functions used for sheet music pre-processing"""
def extract_peaks(scan, threshold):
    scan = np.array(scan)
    max_scan = scan.max()
    list_peaks = []
    for i in range(len(scan)):
        if scan[i] > threshold * max_scan:
            list_peaks.append(i)
    #list_peaks = remove_duplicates(list_peaks)
    return list_peaks

def vertical_scan(image):
    """Scanning partition vertically to find the staff position"""
    histogram_vertical = np.zeros(image.shape[0])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram_vertical[i] += image[i][j]
    return histogram_vertical


def remove_duplicates(list_x):
    """A same peak can have multiple values if the staff line is wide enough"""
    list_peaks = []
    x_0 = list_x[0]
    agregat = [x_0]
    for i in range(1,len(list_x)):
        if list_x[i] == x_0 + 1:
            agregat.append(list_x[i])
            x_0 = list_x[i]
        else:
            list_peaks.append(agregat[int(len(agregat)/2)])
            x_0 = list_x[i]
            agregat = [x_0]
    list_peaks.append(agregat[int(len(agregat)/2)])
    return list_peaks

def group_peaks(list_peaks):
    """Group the peak values by group of 5 (the number of lines of a staff)"""
    if len(list_peaks)%5 != 0:
        return "Uncorrect threhsold for peak detection"
    else:
        peaks = [[list_peaks[i*5 + k] for k in range(5)] for i in range(len(list_peaks)//5)]
        return peaks
    
def select_staff(image, peak_group, extra_width):
    list_staff_images = [image[group[0] - int((group[-1]-group[0])*extra_width) : group[-1]+int((group[-1]-group[0])*extra_width)][:] for group in peak_group]
    return list_staff_images

def convert_BnW_to_binary(image):
    image_height, image_length = image.shape
    image_binary = np.zeros([image_height, image_length])
    for i in range(image_height):
        for j in range(image_length):
            if blackAndWhiteImage[i][j] == 0:
                image_binary[i][j] = 1
    return image_binary

originalImage = cv2.imread("C:/Users/Gloria Sok Cheng/Desktop/Projet OMR/sheet/menuet.PNG")
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
thresh, blackAndWhiteImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('B&W image', blackAndWhiteImage)
binary_image = convert_BnW_to_binary(blackAndWhiteImage)
scan = vertical_scan(binary_image)
peaks = extract_peaks(scan, 0.5)
peaks_unique = remove_duplicates(peaks)
group_of_peaks = group_peaks(peaks_unique)
list_output = select_staff(blackAndWhiteImage, group_of_peaks, 0.4)
for i in range(len(list_output)):
    cv2.imshow('output '+str(i), list_output[i])




