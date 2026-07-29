
import numpy as np
from numpy import random


def generate_blank():
    arr = np.zeros((8, 8), dtype = int)
    return arr

arr = (generate_blank())

def generate_horizontal_line(arr):
    row_index = random.randint(0, 7)
    
    arr[row_index] = 1
    return arr

print(generate_horizontal_line(arr))

def generate_vertical_line():
    new_sample = generate_blank()
    col_index = random.randint(0, 7)
    
    new_sample[:, col_index] = 1
    return new_sample

vertical_image = generate_vertical_line()
print(vertical_image)


def generate_image_dataset(samples_per_class=100):

    X = []
    y = []

    for i in range(samples_per_class):
        new_sample = generate_blank()
        flatten = new_sample.flatten()
        X.append(flatten)
        y.append(0)


    
        
    for i in range(samples_per_class):
        horizontal_image = generate_horizontal_line(generate_blank())
        flatten_horizontal_image = horizontal_image.flatten()
        X.append(flatten_horizontal_image)
        y.append(1)

    for i in range(samples_per_class):
        vertical_image = generate_vertical_line()
        flatten_vertical_image = vertical_image.flatten()
        X.append(flatten_vertical_image)
        y.append(2)


    X = np.array(X)
    y= np.array(y)
    return X,y







        

    
