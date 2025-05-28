import os
import json
import torch
import numpy as np
import pandas
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
import itertools
import SimpleITK as sitk
from resnet import ResNet18
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.svm import SVR
import csv

from pypinyin import pinyin, Style

def compare_posig(pos_sig_list, pos_sig):
    def cal_l1_dist(p1, p2):
        p1 = [int(p1[i]) for i in range(5)]
        p2 = [int(p2[i]) for i in range(5)]
        return np.sum(np.abs(np.array(p1) - np.array(p2)))
    _pos_seg = [ps[0] for ps in pos_sig_list]
    result = []
    for ps in pos_sig:
        min_dist, min_index = 10**7, None
        for i in range(len(_pos_seg)):
            dist = cal_l1_dist(ps, _pos_seg[i])
            if dist < min_dist:
                min_dist = dist
                min_index = i
        result.append(min_index)
    return result

def read_data_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    result_list = []
    sig1_list, sig2_list = [], []
    for name in data:
        print(name)
        tmp_list = []
        for nid in data[name]:
            pos_sig = []
            for pid in data[name][nid]:
                feature = data[name][nid][pid]['feature']
                feature = [float(f) for f in feature]
                xmin = int(data[name][nid][pid]['xmin'])
                ymin = int(data[name][nid][pid]['ymin'])
                xmax = int(data[name][nid][pid]['xmax'])
                ymax = int(data[name][nid][pid]['ymax'])
                z = int(data[name][nid][pid]['z'])
                sig_1 = float(data[name][nid][pid]['signal']['0']) * 10000
                sig_2 = float(data[name][nid][pid]['signal']['1'])
                img_path = data[name][nid][pid]['img_path']
                pos_sig.append([xmin, ymin, xmax, ymax, z, [sig_1, sig_2], img_path, feature])
                sig1_list.append(sig_1)
                sig2_list.append(sig_2)
                # print(sig_1, sig_2)
            tmp_list.append(pos_sig)

        result_list.append([name, tmp_list])

    print('signal range', min(sig1_list), max(sig1_list), min(sig2_list), max(sig2_list))
    print('signal range', min(sig1_list), max(sig1_list), min(sig2_list), max(sig2_list))
    return result_list

def extract_xy(feature_list, name_list, signal):
    x_list, y_list, n_list = [], [], []
    z_gap_list = [['name', 'nodule-id', 'max-z', 'ct-num']]
    for i in range(len(name_list)):
        name = name_list[i]
        for i, nid in enumerate(feature_list[i]):
            combinations = list(itertools.combinations(nid, 2))
            nodule_num = len(nid)
            max_z_gap = 0
            for combined in combinations:
                nid_0, nid_1 = combined[0], combined[1]
                x0 = (nid_0[0] + nid_0[2]) / 2
                y0 = (nid_0[1] + nid_0[3]) / 2
                z0 = nid_0[4]
                x1 = (nid_1[0] + nid_1[2]) / 2
                y1 = (nid_1[1] + nid_1[3]) / 2
                z1 = nid_1[4]
                if abs(z1 - z0) >= 20:
                    print('z >= 20: ', name)
                    continue
                max_z_gap = max(max_z_gap, abs(z1 - z0))

                x0, y0, z0, x1, y1, z1 = x0 / 10, y0 / 10, z0 / 10, x1 / 10, y1 / 10, z1 / 10

                _f = [x0, y0, z0] + nid_0[-1]
                _s = [nid_0[5][i] for i in signal] + [nid_1[5][i] for i in signal]
                _f = _f + _s
                x_list.append(_f)
                y_list.append([x1, y1, z1])
                n_list.append(name)

            z_gap_list.append([name, i, max_z_gap, nodule_num])

    with open('./output/max-z-2.csv', 'a') as file:
        writer = csv.writer(file)
        for i in range(len(z_gap_list)):
            writer.writerow(z_gap_list[i])

    return x_list, y_list, n_list


def extract_xy_2(feature_list, name_list, signal):
    x_list, y_list, n_list = [], [], []
    for i in range(len(name_list)):
        name = name_list[i]
        for nid in (feature_list[i]):
            combinations = list(itertools.combinations(nid, 3))
            for combined in combinations:
                nid_0, nid_1, nid_2 = combined[0], combined[1], combined[2]
                x0 = (nid_0[0] + nid_0[2]) / 2
                y0 = (nid_0[1] + nid_0[3]) / 2
                z0 = nid_0[4]
                x1 = (nid_1[0] + nid_1[2]) / 2
                y1 = (nid_1[1] + nid_1[3]) / 2
                z1 = nid_1[4]
                x2 = (nid_2[0] + nid_2[2]) / 2
                y2 = (nid_2[1] + nid_2[3]) / 2
                z2 = nid_2[4]
                if abs(z1 - z0) >= 20 or abs(z2 - z0) >= 20:
                    print('z >= 20: ', name)
                    continue

                x0, y0, z0, x1, y1, z1 = x0 / 10, y0 / 10, z0 / 10, x1 / 10, y1 / 10, z1 / 10
                x2, y2, z2 = x2 / 10, y2 / 10, z2 / 10

                _f = [x0, y0, z0, x1, y1, z1] + nid_0[-1] + nid_1[-1]
                _s = [nid_0[5][i] for i in signal] + [nid_1[5][i] for i in signal] + [nid_2[5][i] for i in signal]
                _f = _f + _s
                x_list.append(_f)
                y_list.append([x2, y2, z2])
                n_list.append(name)
    return x_list, y_list, n_list

def write_result(name, pred, label, file_name):
    csv_name = file_name + '.csv'
    with open(os.path.join('./output/csv', csv_name), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'label', 'pred'])
        for i in range(len(name)):
            writer.writerow([name[i], label[i], pred[i]])
    return

def train_xgb(signal=[0]):
    data_list = read_data_json('./json/new_norm_data_dict_20250515.json')
    random.seed(142)
    random.shuffle(data_list)
    train_list = data_list[:int(len(data_list) * 0.8)]
    valid_list = data_list[int(len(data_list) * 0.8):]
    train_name = [d[0] for d in train_list]
    valid_name = [d[0] for d in valid_list]
    train_list = [d[1] for d in train_list]
    valid_list = [d[1] for d in valid_list]
    train_x, train_y, _train_name = extract_xy(train_list, train_name, signal)
    valid_x, valid_y, _valid_name = extract_xy(valid_list, valid_name, signal)

    train_x, train_y, valid_x, valid_y = np.array(train_x), np.array(train_y), np.array(valid_x), np.array(valid_y)

    for i in [2]:
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror',
                                   n_estimators=60,
                                   learning_rate=0.1,
                                   max_depth=6,
                                   random_state=42)
        xgb_reg.fit(train_x.astype(float), train_y[:, i].astype(float))
        y_pred_xgb = xgb_reg.predict(train_x.astype(float))
        mse_xgb = mean_squared_error(train_y[:, i].astype(float), y_pred_xgb.astype(float))
        mse_ori = mean_squared_error(train_y[:, i].astype(float), train_x[:, i].astype(float))
        print('dimention:', i)
        print('train predict mse:', mse_xgb)
        print('train original mse:', mse_ori)

        y_pred_xgb = xgb_reg.predict(valid_x.astype(float))
        mse_xgb = mean_squared_error(valid_y[:, i].astype(float), y_pred_xgb.astype(float))
        mse_ori = mean_squared_error(valid_y[:, i].astype(float), valid_x[:, i].astype(float))
        print('dimention:', i)
        print('valid predict mse:', mse_xgb)
        print('valid original mse:', mse_ori)

    file_name = 'xgb_throx' if signal[0] == 0 else 'xgb_flow'
    write_result(_valid_name, y_pred_xgb, valid_y[:, i], file_name)

    return


def train_svm(signal=[1]):
    data_list = read_data_json('./json/new_norm_data_dict.json')
    random.seed(142)
    random.shuffle(data_list)
    train_list = data_list[:int(len(data_list) * 0.8)]
    valid_list = data_list[int(len(data_list) * 0.8):]
    train_name = [d[0] for d in train_list]
    valid_name = [d[0] for d in valid_list]
    train_list = [d[1] for d in train_list]
    valid_list = [d[1] for d in valid_list]
    train_x, train_y, _train_name = extract_xy(train_list, train_name, signal)
    valid_x, valid_y, _valid_name = extract_xy(valid_list, valid_name, signal)

    train_x, train_y, valid_x, valid_y = np.array(train_x), np.array(train_y), np.array(valid_x), np.array(valid_y)

    for i in [2]:
        svm_reg = SVR(kernel='linear', C=0.0001, gamma='auto')
        svm_reg.fit(train_x, train_y[:, i])
        y_pred_svm = svm_reg.predict(train_x)
        mse_svm = mean_squared_error(train_y[:, i].astype(float), y_pred_svm.astype(float))
        mse_ori = mean_squared_error(train_y[:, i].astype(float), train_x[:, i].astype(float))
        print('dimention:', i)
        print('train predict mse:', mse_svm)
        print('train original mse:', mse_ori)

        y_pred_svm = svm_reg.predict(valid_x)
        mse_svm = mean_squared_error(valid_y[:, i].astype(float), y_pred_svm.astype(float))
        mse_ori = mean_squared_error(valid_y[:, i].astype(float), valid_x[:, i].astype(float))
        print('dimention:', i)
        print('valid predict mse:', mse_svm)
        print('valid original mse:', mse_ori)

    file_name = 'svm_throx' if signal[0] == 0 else 'svm_flow'
    write_result(_valid_name, y_pred_svm, valid_y[:, i], file_name)
    return


if __name__ == '__main__':
    train_xgb()
    # train_svm()






































