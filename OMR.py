import pandas as pd
import numpy as np
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import cv2 
from statistics import mean

"""Fonctions used for notes detection"""

def image_value(image,x,y):
    try:
        value = image[x][y]
        return value
    except IndexError:
        return 0
    
def filter_staff(image):
    """filter to remove staff on partition"""
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

def convert_binary_image(image):
    """convert to a binary image"""
    new_image = np.zeros(image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] == 0:
                new_image[x][y] = 255
            else:
                new_image[x][y] = 0
    return new_image

def return_note(staff, x):
    """From the staff position and a x vertical coordinate returns the associated note"""
    dict_note = {staff[0]:"fa", int((staff[0]+staff[1])/2):"mi", staff[1]:"re", int((staff[1]+staff[2])/2):"do", staff[2]:"si", int((staff[2]+staff[3])/2):"la",staff[3]:"sol", int((staff[3]+staff[4])/2):"fa", staff[4]:"mi"} 
    list_position = list(dict_note.keys())
    closest_note = min(list_position, key=lambda y:abs(y-x))
    return dict_note[closest_note]


class OMR():
    def __init__(self, originalImage):
        self.originalImage = originalImage

    def preprocessing(self, debug=False):
        # Opening the sample sheet music image and 
        # converting it to a binary Black and White image (0 -> White, 1 -> Black)
        grayImage = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2GRAY)
        thresh, blackAndWhiteImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

        if debug:
            while True:
                cv2.imshow('Black white image', blackAndWhiteImage)
                # Quit
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
        
        self.image_height, self.image_length = blackAndWhiteImage.shape

        # invert binaries of blackAndWhiteImage
        self.image = np.zeros([self.image_height, self.image_length])
        for i in range(self.image_height):
            for j in range(self.image_length):
                if blackAndWhiteImage[i][j] == 0:
                    self.image[i][j] = 1

    def staff_detection(self, debug=False):
        """Scanning partition vertically to find the staff position"""
        histogram_vertical = np.zeros(self.image_height)

        for i in range(self.image_height):
            for j in range(self.image_length):
                histogram_vertical[i] += self.image[i][j]

                
        self.ind_staff = np.argpartition(histogram_vertical, -5)[-5:]
        self.ind_staff.sort()

        self.filtered_image = filter_staff(self.image)
        
        if debug:
            print(self.ind_staff)
            plt.figure()
            plt.plot(histogram_vertical)
            plt.show()

            while True:
                filtered_image_converted = convert_binary_image(self.filtered_image)
                cv2.imshow('Image without staff', filtered_image_converted) 
                # Quit
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break

        amplitude_staff = self.ind_staff[-1] - self.ind_staff[0]

        return amplitude_staff
    
    def bar_detection(self, debug=False):
        """Scanning vertically to detect vertical bar which delimits the measures
        Computed after staff_detection"""
        bar_x = []
        for x in range(self.image_length):
            if all([self.image[j][x] == 1 for j in range(self.ind_staff[0], self.ind_staff[-1])]):
                bar_x.append(x)
        
        if debug:
            while True:
                image_with_bars = self.originalImage
                # add vertical bars with red lines
                for x in bar_x:
                    image_with_bars = cv2.line(image_with_bars, (x,self.ind_staff[0]), (x,self.ind_staff[-1]), (0, 0, 255), 1)
                cv2.imshow('Image with vertical bars', image_with_bars)
                # Quit
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break

        # get center of bars
        bar_center_x = []
        i = 0
        while i < len(bar_x):
            bar = [bar_x[i]]
            while i+1 < len(bar_x) and bar_x[i+1] == bar_x[i] + 1:
                bar.append(bar_x[i+1])
                i += 1
            i += 1
            center_x = mean(bar)
            bar_center_x.append(center_x)
        
        if debug:
            print(len(bar_center_x), "vertical bars detected")
            
        return bar_center_x

    def symbols_detection(self):
        """Scanning the partition horizontally to detect the symbols"""
        histogram_horizontal = np.zeros(self.image_length)

        for i in range(self.image_length):
            for j in range(self.image_height):
                histogram_horizontal[i] += self.filtered_image[j][i]
                
        plt.figure()
        plt.plot(histogram_horizontal)
        plt.show()


        self.threshold = 5
        # saving the horizontal position [start - end] for detected symbols
        detection = False
        symbols = []
        start = 0
        end = 0
        for i in range(self.image_length):
            if histogram_horizontal[i] > self.threshold:
                if detection:
                    end += 1
                else:
                    start = i
                    end = i
                detection = True
            else:
                if detection:
                    detection = False
                    symbols.append([start, end])
        
        return symbols

    def notes_recognition(self, symbols):
        """Detection and recognition of notes from symbols"""
        notes = []
        notes_position = []
        threshold_largeur_note = 10
        for symbol in symbols[4:]:
            if symbol[1] - symbol[0] > threshold_largeur_note:
                histogram_notes = np.zeros(self.image_height)
                for i in range(self.image_height):
                    for j in range(symbol[0], symbol[1]):
                        histogram_notes[i] += self.filtered_image[i][j]
                notes.append(histogram_notes)
                detection = False
                symbols = []
                start = 0
                end = 0
                for i in range(self.image_height):
                    if histogram_notes[i] > self.threshold:
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
            recognized_notes.append(return_note(self.ind_staff, notes_position[i]))
        
        return recognized_notes

    def main(self):
        self.preprocessing()
        self.staff_detection()
        bar_x = self.bar_detection(debug=True)
        print(bar_x)
        symbols = self.symbols_detection()
        recognized_notes = self.notes_recognition(symbols)
        return recognized_notes
        

if __name__ == '__main__':
    originalImage = cv2.imread("sheet/sheet1.png")
    # originalImage = cv2.imread("sheet/sheet2.png")
    # originalImage = cv2.imread("sheet/Mary-Had-a-Little-Lamb-9.png")

    omr = OMR(originalImage)
    recognized_notes = omr.main()
    print(recognized_notes)
