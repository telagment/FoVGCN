import os
import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def error_map(img1, img2, epsilon=1.):
    # # preprocessing: img should be in Gray scale
    # Original_img_path = '/content/drive/MyDrive/Machine Learning/Nud_GCN/Database/OriginalViewport/I1.png'
    # Test_img_path1 = '/content/drive/MyDrive/Machine Learning/Nud_GCN/Database/nud_viewport_train/001.png'

    # Original_img = cv2.imread(Original_img_path, cv2.IMREAD_GRAYSCALE)
    # Test_img1 = cv2.imread(Test_img_path1, cv2.IMREAD_GRAYSCALE)
    
    # the main fuction to caculate Error Map
    # error_1 = error_map(Original_img, Test_img1)


    assert img1.shape == img2.shape, 'Two inputs should have the same shape!'
    assert len(img1.shape) == 2, 'Inputs should be the grayscale.'

    error = np.log(1/(((img1-img2)**2)+epsilon/(255**2)))/np.log((255**2)/epsilon)
    error = torch.from_numpy(error).float()
    # error = error.type(torch.DoubleTensor)
    error = error.unsqueeze(0)

    # Higher value means lower distance
    # range: [0, 1]
    return error
# Original_img = cv2.imread(Original_img_path, cv2.IMREAD_GRAYSCALE)
# Test_img = cv2.imread(Test_img_path, cv2.IMREAD_GRAYSCALE)
# padding = 80
# Original_img = cv2.copyMakeBorder(Original_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT)
# Test_img = cv2.copyMakeBorder(Test_img, 0, 0, padding, padding, cv2.BORDER_CONSTANT)


# idx_lists = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16']
# for idx_list in enumerate(idx_lists):
#     dis_imgs_dir_name = os.path.join(disimgspath(), str(idx_list))
#     imgs_list = os.listdir(dis_imgs_dir_name)
#     print(imgs_list)
disimg_path = "/ha/Nud_GCN/Database/nud_dis_errormap"
refimg_path = "/ha/Nud_GCN/Database/OriginalViewport"
error_map_path = '/ha/Nud_GCN/Database/error_map'
idx_list = "I1"
disimgs_list = glob(f"{disimg_path}/{idx_list}/*.png")
refimg = glob(f"{refimg_path}/{idx_list}.png")
refimg = cv2.imread(refimg)
print('disimgs_list', disimgs_list)
print('refimgs_list', refimg) 
for idx in range(disimg_path):
    disimg = cv2.imread(idx)
    
