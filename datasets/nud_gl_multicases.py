import os
import six
import torch
import random
import numbers
import numpy as np
import cv2
import math
import scipy.io as scio
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as tF
import itertools
import operator

# list of group image path
def data_test_choose(test_case, data_path):
    test_case_list = list(range((test_case-1) * 32 + 1, test_case * 32 + 1)) # 32 is the number of sample in each distorted image group 
    # print('test_case_list', len(test_case_list), test_case_list)
    test_group = []
    for i, name in enumerate(test_case_list):
        if test_case_list[i] in range(10):
            test_name = data_path + '/' + '00%s'%(test_case_list[i]) + '.pt'
            # test_group += test_name
            test_group.append(test_name)
            # print(test_name)
        elif test_case_list[i] in range(10,100):
            test_name = data_path + '/' + '0%s'%(test_case_list[i]) + '.pt'
            # test_group += test_name
            test_group.append(test_name)
            # print(test_name)
        else: 
            test_name = data_path + '/' + '%s'%(test_case_list[i]) + '.pt'
            # test_group += test_name
            test_group.append(test_name)
            # print(test_name)
    # print('test_group: ', test_group)
    return test_group

def data_train_choose(train_case, data_path):
    train_cases = list(range((train_case - 1) * 32 + 1, train_case * 32 + 1)) # 32 is the number of sample in each distorted image group 
    train_case_flip_1 = [element * 2 for element in train_cases]
    train_case_flip_2 = [element * 3 for element in train_cases]
    train_cases_list = train_cases + train_case_flip_1 + train_case_flip_2
    # print('train_case_list: ', train_cases_list)
    train_group = []
    for i, name in enumerate(train_cases_list):
        if train_cases_list[i] in range(10):
            train_name = data_path + '/' + '00%s'%(train_cases_list[i]) + '.pt'
            # train_group += train_name
            train_group.append(train_name)
            # print(tes_name)
        elif train_cases_list[i] in range(10,100):
            train_name = data_path + '/' + '0%s'%(train_cases_list[i]) + '.pt'
            # train_group += train_name
            train_group.append(train_name)
            # print(test_name)
        else: 
            train_name = data_path + '/' + '%s'%(train_cases_list[i]) + '.pt'
            # train_group += train_name
            train_group.append(train_name)
            # print(test_name)
    # print('train_group', train_group)
    return train_group

#get the affinity matrix
def get_aff_path():
    set_root = 'Database/attention_weightmatrix/A_binary_quandratic/A_linear_e.pt'
    return set_root

#/content/drive/MyDrive/Machine Learning/ha_/Nud_GCN/VGCN_PyTorch/Database/affinity_matrix/Affinity_modified/A_Gaussian_01.pt
#/content/drive/MyDrive/Machine Learning/ha_/Nud_GCN/VGCN_PyTorch/Database/affinity_matrix/Affinity_modified/A_360_corner_0.0001.pt


#Retangular_Nud
def preprocessing(errormap):
    tensor_split_0 = torch.split(errormap, 360, 1)

    error_I_II = tensor_split_0[0]
    error_IV_III = tensor_split_0[1]

    error_I = torch.split(error_I_II, 360, 2)[0]
    error_II = torch.split(error_I_II, 360, 2)[1]
    error_IV = torch.split(error_IV_III, 360, 2)[0]
    error_III = torch.split(error_IV_III, 360, 2)[1]

    I = error_I
    II = torch.rot90(error_II, 3, (2, 1))
    III = torch.rot90(error_III, 2, (2, 1))
    IV = torch.rot90(error_IV, 1, (2, 1))

    error_input = torch.cat((I, II, III, IV), 2)
    return error_input 

def get_labelspath(is_training):
    sets_root = 'Database/'
    if is_training:
        sets = [
            'nud_fovall_label'
        ]
    else:
        sets = [
            'nud_fovall_label'
        ]
    return [os.path.join(sets_root, set) for set in sets]

def get_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])
def get_data_path(data_path, train_case, test_case):
    test_set_list = []
    train_set_list = []
    test_case_1 = test_case[0]
    test_case_2 = test_case[1]
    print('train_case: ', train_case)
    print('test_case: ', test_case)

    test_group_1 = data_test_choose(test_case_1, data_path)
    test_group_2 = data_test_choose(test_case_2, data_path)
    test_set_list = test_group_1 + test_group_2
    for i, sample in enumerate(train_case):
      # get test set
        train_group = data_train_choose(sample, data_path)
        train_set_list += train_group
        # train_set_list.append(train_group)
        # print('Train: ', train_case)

    # print(len(test_set_list), test_set_list)
    # print(len(train_set_list), train_set_list)
    return train_set_list, test_set_list

# output: data(error tensor), label, A
class ImgDataset(data.Dataset):
    def __init__(self, errtensor_path, errtensor_name, aff_path, transform, is_training, label_path, shuffle=False):
        self.errtensor_path = errtensor_path
        self.errtensor_name = errtensor_name
        self.aff_path = aff_path
        self.nSamples = len(self.errtensor_path)
        self.indices = range(self.nSamples)
        if shuffle:
            random.shuffle(self.indices)
        self.transform = transform
        self.is_training = is_training
        self.label_path = label_path

    def __getitem__(self, index):
        errtensorpath = self.errtensor_path[index]
        errtensorname = self.errtensor_name[index]
        label = []
        errtensor = torch.load(errtensorpath)
        errtensor = errtensor.squeeze(0)
        
        # get the label
        labelname_dir = os.path.splitext(str(errtensorname))
        labelname = labelname_dir[0] + '.mat'
        labelname = os.path.join(self.label_path, labelname)
        label_content = scio.loadmat(labelname)
        label = label_content['score'][0]  
        label = torch.from_numpy(label).float()       

        # get affinity matrix 
        A = torch.load(str(self.aff_path))
        A = A.squeeze(0)
        return errtensor, label, errtensorname, A
    
    def __len__(self):
        return self.nSamples

def get_dataset(data_path, train_case, test_case, is_training): #errtensor_name
    errtensor_path_list = []
    errtensor_name_list = []
    datasets_list = []
    if is_training == True:
        err_sets_path = get_data_path(data_path, train_case, test_case)[0] # get the path of errormaps for training 
    else:
        err_sets_path = get_data_path(data_path, train_case, test_case)[1] # get the path of errormaps for testing
        
    aff_path = get_aff_path() # get the path of affinity matrix 
    labels_path = get_labelspath(is_training)   # get the label paths

    transform = get_transform()
    for err_set_path in err_sets_path:
        errtensor_name = os.path.basename(str(err_set_path))
        errtensor_name_list.append(errtensor_name)
        errtensor_path_list.append(err_set_path)
        # print('errtensor_path_list: ', len(errtensor_path_list))
        # print(errtensor_path_list)
    
    datasets_list.append(
        ImgDataset(
            errtensor_path = errtensor_path_list,
            errtensor_name = errtensor_name_list,
            aff_path = aff_path,
            transform = transform,
            is_training = is_training,
            label_path = labels_path[0]
        )
    )
    return data.ConcatDataset(datasets_list)
