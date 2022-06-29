import os
import time
import argparse
import torch
import math
import numpy as np
import cv2
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable
from torchvision import models
import scipy.io as scio
from scipy import stats
import torch.nn as nn
from torchvision import models
import random
import datetime
from datetime import datetime
import utils
from datasets.nud_gl_multicases import get_dataset
from model.nud_model import FovGCN
import itertools
import operator

import gc
# del variables #delete unnecessary variables 
gc.collect()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Training settings
parser = argparse.ArgumentParser(description='VR Image Quality Assessment')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--start_case', type=int, default=119)
parser.add_argument('--end_case', type=int, default=120)
parser.add_argument('--total_epochs', type=int, default=200, help = 'default 20')
parser.add_argument('--total_iterations', type=int, default=10000, help = 'default 10000')
parser.add_argument('--batch_size', '-b', type=int, default=12, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-4, metavar=' LR', help='learning rate (default: 0.01)')
parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=4)
parser.add_argument('--save', '-s', default='model/checkpoint', type=str, help='directory for saving')
parser.add_argument('--skip_training', default=False, action='store_true')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to load the weight')

args = parser.parse_args()

seed = [7021, 9042, 9042, 8264]
torch.manual_seed(seed[0])
torch.cuda.manual_seed_all(seed[1])
np.random.seed(seed[2])
random.seed(seed[3])

kwargs = {'num_workers': args.number_workers}

def train(epoch, iteration):
    torch.cuda.empty_cache()
    model.train()
    # scheduler.step()
    end = time.time()
    score_list = []
    label_list = []
    name_list = []
    train_losses = []
    running_loss = 0
    test_loss = 0
    total = 0
    correct = 0
    srocc = plcc = rmse = 0
    # rmse_mean = rmse_sum = 0
    log = [0 for _ in range(1)]
    for batch_idx, batch in enumerate(train_loader):
        errtensor, label, _, A = batch
        # print('errtensor_: ', errtensor.shape)
        errtensor = Variable(errtensor.cuda(), requires_grad=True)
        label = Variable(label.cuda(), requires_grad=True)
        A = Variable(A.cuda(), requires_grad=True)
        optimizer.zero_grad()
        
        score, label, loss, rmse = model(errtensor, label, A, requires_loss=True)
        loss.backward()
        optimizer.step()

        score = score.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        loss = loss.cpu().detach().numpy()
        rmse = rmse.cpu().detach().numpy()

        score_list.append(score)
        label_list.append(label)

        # rmse_sum += rmse 
        running_loss += loss.item()
        total += label
        iteration += 1
    # rmse_mean = rmse_sum/len(train_loader)
    train_loss = running_loss/len(train_loader)
    # train_losses.append(train_loss)
    # print('Train Loss: %.3f: '%(train_loss))
    #print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

    score_list = np.reshape(np.asarray(score_list), (-1,))
    label_list = np.reshape(np.asarray(label_list), (-1,))
    name_list = np.reshape(np.asarray(name_list), (-1,))

    srocc = stats.spearmanr(label_list, score_list)[0]
    plcc = stats.pearsonr(label_list, score_list)[0]
   
    # print('Train: SROCC: %.4f, PLCC: %.4f, RMSE: %.4f\n' % (srocc, plcc, rmse))
    return srocc, plcc, rmse, train_loss
    #return srocc, plcc, rmse, train_loss, accu

def eval():

    model.eval()
    log = 0
    score_list = []
    label_list = []
    name_list = []
    running_loss = []
    eval_losses = []
    iteration = 0
    # rmse_mean = rmse_sum = 0

    for batch_idx, batch in enumerate(test_loader):
        errtensor, label, errtensorname , A = batch
        errtensor = Variable(errtensor.cuda(), )
        label = Variable(label.cuda())
        A = Variable(A.cuda())

        score, label , loss, rmse = model(errtensor, label, A, requires_loss=True)

        score = score.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        loss = loss.cpu().detach().numpy()
        rmse = rmse.cpu().detach().numpy()

        score_list.append(score)
        label_list.append(label)
        name_list.append(errtensorname[0])

        # rmse_sum += rmse 
        running_loss += loss
        iteration += 1

        ## release memory
        torch.cuda.empty_cache()

    # rmse_mean = rmse_sum/len(train_loader)
    test_loss = running_loss/len(test_loader)

    eval_losses.append(test_loss)

    score_list = np.reshape(np.asarray(score_list), (-1,))
    label_list = np.reshape(np.asarray(label_list), (-1,))
    name_list = np.reshape(np.asarray(name_list), (-1,))

    srocc = stats.spearmanr(label_list, score_list)[0]
    plcc = stats.pearsonr(label_list, score_list)[0]

    # print('Test: SROCC: %.4f, PLCC: %.4f, RMSE: %.4f\n' % (srocc, plcc, rmse))
    return srocc, plcc, rmse, test_loss, score_list, label_list
    # return srocc, plcc, rmse, test_loss, accu

test_cases_all = []
all_case = list(range(1,17))
# print(type(data[0]), data)
test_cases = itertools.combinations(all_case, 2)
data_path = 'Database/errormap/data_full'
# list_case = list(rang(args.start_case, args.end_case + 1))

for i, sample in enumerate(test_cases):
    test_case = list(sample)
    test_cases_all.append(test_case)

test_sample = test_cases_all[args.start_case:args.end_case]
# print('test_cases_all: ', len(test_cases_all), test_cases_all)
# print('test_sample: ', len(test_sample), test_sample)

srocc_train_final = []
plcc_train_final = []
rmse_train_final = []
srocc_test_final = []
plcc_test_final = []
rmse_test_final = []
best_case_SROCC_list = []
best_case_PLCC_list = []
best_case_RMSE_list = []
best_case_SROCC = 0
best_case_PLCC = 0
best_case_RMSE = 0

for case, sample in enumerate(test_sample):
    test_case = test_sample[case]
    train_case = [i for i in all_case if i != sample[0]]
    train_case = [i for i in train_case if i != sample[1]]
    print(train_case)
    print(test_case)
    if not args.skip_training:
        train_set = get_dataset(data_path, train_case, test_case, is_training=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_set = get_dataset(data_path, train_case, test_case, is_training=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, **kwargs)

    model = FovGCN().cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not args.skip_training: # training 
        best = -100
        srocc_train_list = []
        plcc_train_list = []
        rmse_train_list = []

        srocc_test_list = []
        plcc_test_list = []
        rmse_test_list = []

        score_list = []
        label_list = []
        for epoch in range(args.start_epoch, args.total_epochs+1):
            iteration = (epoch-1) * len(train_loader) + 1
            log_train = train(epoch, iteration)
            log_test = eval()

            srocc = log_test[0]
            plcc = log_test[1]
            loss_test = log_test[2]

            srocc_train_list.append(log_train[0])
            plcc_train_list.append(log_train[0])
            rmse_train_list.append(log_train[0])

            srocc_test_list.append(srocc)
            plcc_test_list.append(plcc)
            rmse_test_list.append(loss_test)

            current_cc = srocc + plcc + (1 - loss_test)
            # print('current_cc: ', current_cc)
            actual_case = case + args.start_case
            # print('Case: %d , Epoch: %d ' %(actual_case, epoch))
            # print('Training: SROCC: %.4f, PLCC: %.4f, RMSE: %.4f' % (log_train[0], log_train[1], log_train[2]))
            # print('Testing: SROCC: %.4f, PLCC: %.4f, RMSE: %.4f\n' % (srocc, plcc, loss_test))
            # print('Epoch %d, Training Loss %f, Training Accuracy %f, Testing Loss %f, Testing Accuracy %f' % (epoch, loss_train, loss_test))
            # print('Epoch %d, Training Loss %f, Training Accuracy %f, Testing Loss %f, Testing Accuracy %f' % (epoch, float(loss_train), float(accu_train), float(loss_test), float(accu_test)))

            # create and save .mat file
            now = datetime.now()
            result_detail_name = now.strftime("%m_%d_") + str(actual_case) + '_detail' +'.mat'
            # result = now.strftime("%m_%d_") + str(actual_case) +'.mat'
            mat_dir_1 = 'results/binarycase/result_detail'
            # mat_dir_2 = '/content/drive/MyDrive/Machine Learning/ha_/Nud_GCN/VGCN_PyTorch/results/result'
            mat_name_1 = os.path.join(mat_dir_1, result_detail_name)
            # mat_name_2 = os.path.join(mat_dir_2, result)
            # scio.savemat(mat_name_2, {'SROCC_train': log_train[0], 'PLCC_train': log_train[1], 'RMSE_train': log_train[2],
            #                           'SROCC_test': srocc, 'PLCC_test': plcc, 'RMSE_test': loss_test})
            scio.savemat(mat_name_1, {'score': log_test[4], 'label': log_test[5],
                                      'SROCC_train_list': srocc_train_list,'PLCC_train_list': plcc_train_list, 'RMSE_Train_list': rmse_train_list, 
                                      'SROCC_test_list': srocc_test_list,'PLCC_test_list': plcc_test_list, 'RMSE_test_list': rmse_test_list})
            if current_cc > best:
                best = current_cc
                best_case_SROCC = srocc
                best_case_PLCC = plcc
                best_case_RMSE = loss_test
                print('Case: %d , Epoch: %d , best_SROCC: %4f , best_PLCC: %4f, best_RMSE: %4f' %(actual_case, epoch, best_case_SROCC, best_case_PLCC, best_case_RMSE))
                best_case_SROCC_list.append(best_case_SROCC)
                best_case_PLCC_list.append(best_case_PLCC)
                best_case_RMSE_list.append(best_case_RMSE)

        best_case_SROCC_list.append(best_case_SROCC)
        best_case_PLCC_list.append(best_case_PLCC)
        best_case_RMSE_list.append(best_case_RMSE)

        # mat_path = '/content/drive/MyDrive/MachineLearning/ha_/Nud_GCN/VGCN_PyTorch/results/linear/' 
        mat_path = 'results/binarycase/' 
        # /content/drive/MyDrive/MachineLearning/ha_/Nud_GCN/VGCN_PyTorch/results/anpha
        now = datetime.now()
        mat_name = mat_path + str(args.start_case) + '_' + str(args.end_case) +'best.mat'
        scio.savemat(mat_name, {'SROCC': best_case_SROCC_list,'PLCC': best_case_PLCC_list, 'RMSE': best_case_RMSE_list})

        if current_cc > best:
            best = current_cc
            name_checkpoint = 'checkpoint' + str(actual_case)
            checkpoint = os.path.join(args.save, name_checkpoint)
            utils.save_model(model, checkpoint, epoch, is_epoch=True)
        
    else: # testing
        print('Test Load pre-trained model!')
        # utils.load_model(model, args.resume)
        model.load_state_dict(torch.load(args.resume))
        eval()

    
