import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

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
    A = torch.from_numpy(A).float()
    return A

# overal fix sized image
view_img_path = '/ha/Nud_GCN/Database/OriginalViewport/I1.png'
view_img = cv2.imread(view_img_path)
padding = 80
view_img = cv2.copyMakeBorder(view_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT)
A = affinity_matrix(view_img)