from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt

def error_map(img1, img2, epsilon=1.):
    # # preprocessing: img should be in Gray scale
    # Original_img_path = './I1_viewport_0_1280x1440x8_cf1.png'
    # Test_img_path1 = './001_fov1.png'

    # Original_img = cv2.imread(Original_img_path, cv2.IMREAD_GRAYSCALE)
    # Test_img1 = cv2.imread(Test_img_path1, cv2.IMREAD_GRAYSCALE)
    
    # # the main fuction to caculate Error Map
    # error_1 = error_map(Original_img, Test_img1)


    assert img1.shape == img2.shape, 'Two inputs should have the same shape!'
    assert len(img1.shape) == 2, 'Inputs should be the grayscale.'

    # Higher value means lower distance
    # range: [0, 1]
    return np.log(1/(((img1-img2)**2)+epsilon/(255**2)))/np.log((255**2)/epsilon)

def affinity_matrix(img):
    # the image should be in Gray scale
    img_shape =img.shape
    h = img_shape[0]
    w = img_shape[1]
    # center point position (x0,y0)
    x0 = int(w/2)
    y0 = int(h/2)
    # the distance maximum between  
    z_max = np.sqrt(((w-1)/2)**2 + ((h-1)/2)**2)
    x = np.arange(0, w, 1) # x: column
    y = np.arange(0, h, 1) # y: row
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.sqrt((xx - x0)**2 +(yy-y0)**2)
    # Affinity matrix
    A = 1 - (z/z_max)
    return A

