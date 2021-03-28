#Author : Khettab Billel
# March 23th, 2021

import numpy as np
import math
from itertools import product
import os
import json

CELL_HEIGHT = 32.0
CELL_WIDTH = 32.0

GRID_W = 80
GRID_H = 45

IMAGE_HEIGHT = 720.0
IMAGE_WIDTH = 1280.0

INPUT_DATASET_PATH = os.path.join(os.path.dirname(__file__),"full_dataset_test_lsnet/") # Dataset labelme format (Line or line strip)
OUTPUT_DATASET_PATH = os.path.join(os.path.dirname(__file__),"lsnet_format/")

def equal(a, b):
    return a[0] == b[0] and a[1] == b[1]

def get_grid_lines_intersections(annot) :
    if annot["imageHeight"] != IMAGE_HEIGHT :
        return None

    if annot["imageWidth"] != IMAGE_WIDTH :
        return None

    shapes = annot["shapes"]
    intersections = []
    for shape in shapes :
        points = shape["points"]
        num_points = len(points)
        for i in range(num_points-1) :
            X1 = points[i][0]
            Y1 = points[i][1]

            X2 = points[i+1][0]
            Y2 = points[i+1][1]

            if i == 0 :
                x_list = [X1,X2]
                y_list = [Y1,Y2]
            else :
                x_list = [X2]
                y_list = [Y2]

            #FIND INTERSECTIONS WITH THE VERTICAL GRID AND THE HOPRIZONTAL GRID
            if X1 == X2 :
                y = min(Y1, Y2)
                Y = max(Y1, Y2)
                y_list .append(math.floor(y/CELL_HEIGHT) + float(y % CELL_HEIGHT != 0 )*CELL_HEIGHT)
                y = y_list[-1]
                x_list.append(X1)
                y += CELL_HEIGHT
                while y <= Y :
                    y_list.append(y)
                    x_list.append(X1)
                    y += CELL_HEIGHT
            else :
                x = min(X1, X2)
                X = max(X1, X2)
                if Y1 == Y2 :
                    x_list .append((math.floor(x/CELL_WIDTH) + float(x % CELL_WIDTH != 0) )*CELL_WIDTH)
                    y_list.append(Y1)
                    x = x_list[-1] + CELL_WIDTH
                    
                    while x <= X :
                        x_list.append(x)
                        y_list.append(Y1)
                        x += CELL_WIDTH
                else :
                    a = (Y2 - Y1 ) / (X2 - X1)
                    b = Y1 - a*X1

                    x_list.append((math.floor(x/CELL_WIDTH) + float(x % CELL_WIDTH != 0) )*CELL_WIDTH)
                    y_list.append(a*x_list[0] + b)
                    x = x_list[-1] + CELL_WIDTH
            
                    while x <= X :
                        x_list.append(x)
                        y_list.append(a*x + b)
                        x += CELL_WIDTH

                    y = min(Y1, Y2)
                    Y = max(Y1, Y2)
                
                    y_list.append((math.floor(y/CELL_HEIGHT) + float(y % CELL_HEIGHT != 0) )*CELL_HEIGHT)
                    y = y_list[-1]
                    x_list.append((y - b) / a)
                    y += CELL_HEIGHT
                    while y <= Y :
                        y_list.append(y)
                        x_list.append((y - b) / a)
                        y += CELL_HEIGHT

            flist = list(zip(x_list, y_list))
            intersections.append(flist)
    
        return intersections

def crop_outbound(intersections) :
    new_intersections = []
    for line in intersections :
        nline = line.copy()
        for x in line :
            if x[0] < 0 or x[0] > IMAGE_WIDTH or x[1] < 0 or x[1] > IMAGE_HEIGHT :
                nline.remove(x)
        new_intersections.append(nline)

    return new_intersections

def gen_grid(intersections) :
    proba_grid = np.array([0.0 , 1.0]) * np.ones(shape = (GRID_H, GRID_W, 2))
    reg_grid = np.zeros(shape = (GRID_H, GRID_W, 4))
    for line in intersections :
        for x, y in line : 
            i = int(x / CELL_WIDTH)
            j = int(y / CELL_HEIGHT)
            
            if x % CELL_WIDTH == 0 :
                if i == 0 :
                    I = [i]
                    MI = [0.0]
                elif i == GRID_W :
                    I = [i-1]
                    MI = [1.0]
                else :
                    I = [i, i-1]
                    MI = [0.0, 1.0]
            else :
                I = [i]
                MI = [(x - i * CELL_WIDTH)/CELL_WIDTH]

            if y % CELL_HEIGHT == 0 :
                if j == 0 :
                    J = [j]
                    MJ = [0.0]
                elif j == GRID_H :
                    J = [j-1]
                    MJ = [1.0]
                else :
                    J = [j, j-1]
                    MJ = [0.0, 1.0]
            else :
                J = [j]
                MJ = [(y - j * CELL_HEIGHT)/CELL_HEIGHT]

            it = list(zip(product(J, I), product(MJ,MI)))
            for packed in it :
                mx = packed[1][1]
                my = packed[1][0]

                i = packed[0][0]
                j = packed[0][1] 

                if proba_grid[i][j][0] == 1.0 :
                    reg_grid[i][j][2] = mx 
                    reg_grid[i][j][3] = my
                else :
                    proba_grid[i][j][0] = 1.0
                    proba_grid[i][j][1] = 0.0

                    reg_grid[i][j][0] = mx 
                    reg_grid[i][j][1] = my

    return proba_grid, reg_grid


def convert2LSNET() :
    files = os.listdir(INPUT_DATASET_PATH)
    for file in files  :
        data = json.load(open(INPUT_DATASET_PATH + file))
        intersections = crop_outbound(get_grid_lines_intersections(data))
        proba_grid, reg_grid = gen_grid(intersections)
        with open(OUTPUT_DATASET_PATH + "proba_" + file, "w") as fout:
            json.dump(proba_grid.tolist(), fout)

        with open(OUTPUT_DATASET_PATH + "seg_" + file), "w") as fout :
            json.dump(reg_grid.tolist(), fout)


    


if __name__ == '__main__':
    convert2LSNET()