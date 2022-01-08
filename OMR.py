# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:20:54 2021

@author: Hugo Poinard
"""

import pandas as pd
import numpy as np
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import cv2 



"""Fonctions utilisées pour la détection des notes"""
#filtre pour enlever la portée de la partition
def image_value(image,x,y):
    try:
        value = image[x][y]
        return value
    except IndexError:
        return 0
    
def filter_staff(image):
    height = image.shape[0]
    length = image.shape[1]
    filtered_image = np.zeros(image.shape)
    threshold_filter = 1.5
    for x in range(height):
        for y in range(length):
            somme = image[x][y] + 0.5*image_value(image,x+1,y) + 0.5*image_value(image,x-1,y) + 0.2*image_value(image,x,y-1) + 0.2*image_value(image,x,y+1)
            if somme > threshold_filter:
                filtered_image[x][y] = 1
                
    return filtered_image


#Convertit en une image binaire en une image B&W
def convert_binary_image(image):
    new_image = np.zeros(image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] == 0:
                new_image[x][y] = 255
            else:
                new_image[x][y] = 0
    return new_image

#A partir de la position de la portée (liste) et d'une coordonnée verticale x, renvoie la note associée
def return_note(staff, x):
    dict_note = {staff[0]:"fa", int((staff[0]+staff[1])/2):"mi", staff[1]:"ré", int((staff[1]+staff[2])/2):"do", staff[2]:"si", int((staff[2]+staff[3])/2):"la",staff[3]:"sol", int((staff[3]+staff[4])/2):"fa", staff[4]:"mi"} 
    list_position = list(dict_note.keys())
    closest_note = min(list_position, key=lambda y:abs(y-x))
    return dict_note[closest_note]
    



#Opening the sample sheet music image and converting it to a binary Black and White image (0 -> White, 1 -> Black)
originalImage = cv2.imread("sheet1.png")
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Black white image', blackAndWhiteImage)

image_height = blackAndWhiteImage.shape[0]
image_length = blackAndWhiteImage.shape[1]


image = np.zeros([image_height, image_length])
for i in range(image_height):
    for j in range(image_length):
        if blackAndWhiteImage[i][j] == 0:
            image[i][j] = 1
            


"""Détection de la portée"""
#On scanne la partition verticalement pour trouver la position de la portée. 
histogram_vertical = np.zeros(image_height)

for i in range(image_height):
    for j in range(image_length):
        histogram_vertical[i] += image[i][j]

        
ind_staff = np.argpartition(histogram_vertical, -5)[-5:]
ind_staff.sort()

print(ind_staff)
plt.figure()
plt.plot(histogram_vertical)

amplitude_staff = ind_staff[-1] - ind_staff[0]




filtered_image = filter_staff(image)
filtered_image = convert_binary_image(filtered_image)
cv2.imshow('Image without staff', filtered_image) 

"""Détection des symboles"""
#On scanne la partition horizontalement pour détecter les symbôles
histogram_horizontal = np.zeros(image_length)

for i in range(image_length):
    for j in range(image_height):
        histogram_horizontal[i] += filtered_image[j][i]
        
plt.figure()
plt.plot(histogram_horizontal)


threshold = 5
#On stocke la position horizontale [début - fin] des symbôles détectés
detection = False
symbols = []
start = 0
end = 0
for i in range(image_length):
    if histogram_horizontal[i] > threshold:
        if detection:
            end += 1
        else:
            start = i
            end = i
        detection = True
    else:
        if detection :
            detection = False
            symbols.append([start, end])

"""Détection et reconnaissance des notes"""
#On extraie les notes depuis les symbôles
notes = []
notes_position = []
threshold_largeur_note = 10
for symbol in symbols[4:]:
    if symbol[1] - symbol[0] > threshold_largeur_note:
        histogram_notes = np.zeros(image_height)
        for i in range(image_height):
            for j in range(symbol[0], symbol[1]):
                histogram_notes[i] += filtered_image[i][j]
        notes.append(histogram_notes)
        detection = False
        symbols = []
        start = 0
        end = 0
        for i in range(image_height):
            if histogram_notes[i] > threshold:
                if detection:
                    end += 1
                else:
                    start = i
                    end = i
                    detection = True
            else:
                if detection :
                    detection = False
        notes_position.append((end+start)/2)    
      
recognized_notes = []
for i in range(len(notes_position)):
    recognized_notes.append(return_note(ind_staff, notes_position[i]))

print(recognized_notes)



        

        
"""
class SheetMusic:
    def _init_(self):
        self.clef = "
        self.key = "G_Major"
        self.time_signature = "4/4"
        
    def transpose(self, new_key):
        self.key = new_key
        """