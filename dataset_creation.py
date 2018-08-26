import os

import pandas as pd
import numpy as np
import cv2
import sys
from random import random




def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random()

            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]

    return output


def add_image(image, token):
    token += 1
    cv2.imwrite('positive/res%d.png' % token, image)

    return token


def gen_positive():

    df = pd.read_csv('data/aorta_labels.csv', 'a', delimiter=',')
    token = 0
    for ind, row in df.iterrows():
        image = cv2.imread('class/%s' % (row['filename']), 1)

        x_min = row['xmin']
        x_max = row['xmax']
        y_min = row['ymin']
        y_max = row['ymax']

        for delta_x in range(-5, 5, 2):
            for delta_y in range(-8, 8, 2):
                x_min = row['xmin'] + delta_x
                x_max = row['xmax'] + delta_x
                y_min = row['ymin'] + delta_y
                y_max = row['ymax'] + delta_y


                cropped_image = image[y_min: y_min + (x_max - x_min), x_min: x_max]
                blur = cv2.GaussianBlur(sp_noise(cropped_image, 0.03), (5, 5), 0)

                median = cv2.medianBlur(cropped_image, 5)


                token = add_image(blur, token)
                token = add_image(cropped_image, token)
                token = add_image(median, token)


def gen_negative():
    token = 0
    for filename in os.listdir('norm'):

        token += 1
        if filename.startswith('no'):
            image = cv2.imread('norm/%s' % filename)

            resized_image = cv2.resize(image, (60, 60))
            cv2.imwrite('negative/res%d.png' % token, resized_image)





if __name__ == '__main__':
    gen_positive()

