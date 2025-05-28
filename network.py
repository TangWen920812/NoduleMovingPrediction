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
import csv

OMEGA = 30

class SirenActivation(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, input):
        return torch.sin(OMEGA * input)

    @staticmethod
    def sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / OMEGA, np.sqrt(6 / num_input) / OMEGA)

    @staticmethod
    def first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)

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

# def read_data_json(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     result_list = []
#     sig1_list, sig2_list = [], []
#     for name in data:
#         feature = data[name]['feature']
#         feature = [float(f) for f in feature]
#         tmp_list = []
#         for pid in data[name]:
#             if pid == 'feature': continue
#             pos_sig = []
#             for nid in data[name][pid]['position']:
#                 xmin = int(data[name][pid]['position'][nid]['xmin'])
#                 ymin = int(data[name][pid]['position'][nid]['ymin'])
#                 xmax = int(data[name][pid]['position'][nid]['xmax'])
#                 ymax = int(data[name][pid]['position'][nid]['ymax'])
#                 z = int(data[name][pid]['position'][nid]['z'])
#                 sig_1 = float(data[name][pid]['signal']['0']['signal']) * 1000
#                 sig_2 = -float(data[name][pid]['signal']['1']['signal']) * 1000
#                 img_path = data[name][pid]['position'][nid]['img_file']
#                 pos_sig.append([xmin, ymin, xmax, ymax, z, [sig_1, sig_2], img_path])
#                 sig1_list.append(sig_1)
#                 sig2_list.append(sig_2)
#             if len(tmp_list) == 0:
#                 tmp_list = [[p] for p in pos_sig]
#             else:
#                 insert_index = compare_posig(tmp_list, pos_sig)
#                 for i in range(len(insert_index)):
#                     tmp_list[insert_index[i]].append(pos_sig[i])
#
#         for i in range(len(tmp_list)):
#             for j in range(len(tmp_list[i])):
#                 tmp_list[i][j].append(feature)
#         result_list.append([name, tmp_list])
#     print('signal range', min(sig1_list), max(sig1_list), min(sig2_list), max(sig2_list))
#     return result_list

def read_data_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    result_list = []
    sig1_list, sig2_list = [], []
    for name in data:
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


class FeatureMLP(nn.Module):
    def __init__(self, channel, cls_num):
        super(FeatureMLP, self).__init__()
        self.sin = SirenActivation()
        self.layer_0 = nn.Sequential(
            nn.Linear(channel, 64),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 128)
        )
        self.layer_1 = nn.Sequential(
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )
        self.layer_2 = nn.Sequential(
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512)
        )
        self.layer_3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512*2, 64)
        )
        self.skip_1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.InstanceNorm1d(256),
        )
        self.skip_2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.InstanceNorm1d(512),
        )
        self.img_feature_extractor = ResNet18(512)

        self.layer_out0 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        self.layer_out1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        self.layer_out2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        self.layer_out3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 5),
            # self.sin,
            nn.LeakyReLU(),
            nn.Linear(5, 5, bias=True),

        )
    def forward(self, x, img):
        img_f = self.img_feature_extractor(img)
        x1 = self.layer_0(x)
        x2 = self.layer_1(x1)
        x2 = x2 + self.skip_1(x1)
        x3 = self.layer_2(x2)
        x3 = x3 + self.skip_2(x2)
        feature = self.layer_3(torch.cat([x3, img_f], dim=-1))
        pred_x = self.layer_out0(feature)
        pred_y = self.layer_out1(feature)
        pred_z = self.layer_out2(feature)
        pred_l1 = self.layer_out3(feature)
        return torch.cat([pred_x, pred_y, pred_z], dim=-1), pred_l1

class FeatureData(Dataset):
    def __init__(self, feature_list, name_list, signal=[0], if_train=True):
        # feature, signal, position
        self.feature_list = feature_list
        self.name_list = name_list
        self.signal = signal
        self.if_train = if_train
        if True:
            self.feature_list, self.name_list = self.valid_data_process()

    def valid_data_process(self):
        delta = [[], [], []]
        feature_list, name_list = [], []
        self.item_dic = {}
        item = 0
        for i in range(len(self.name_list)):
            name = self.name_list[i]
            for nid in (self.feature_list[i]):
                combinations = list(itertools.combinations(nid, 2))
                for combined in combinations:
                    nid_0, nid_1 = combined[0], combined[1]
                    x0 = (nid_0[0] + nid_0[2]) / 2
                    y0 = (nid_0[1] + nid_0[3]) / 2
                    z0 = nid_0[4]
                    x1 = (nid_1[0] + nid_1[2]) / 2
                    y1 = (nid_1[1] + nid_1[3]) / 2
                    z1 = nid_1[4]
                    delta_x = int(x0 - x1)
                    delta_y = int(y0 - y1)
                    delta_z = int(z0 - z1)
                    delta[0].append(delta_x)
                    delta[1].append(delta_y)
                    delta[2].append(delta_z)
                    if abs(delta_z) >= 20: continue

                    feature_list.append([[nid_0, nid_1]])
                    name_list.append(name)
                    self.item_dic[item] = [delta_x, delta_y, delta_z]
                    item += 1

        for i in range(3):
            value_counts = Counter(delta[i])
            unique_values, counts = zip(*value_counts.items())
            plt.bar(unique_values, counts, color='skyblue')
            plt.title('Value Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.savefig('./' + str(i) + '.jpg')
            plt.clf()

        times_num = [0, 0, 0]
        for i in range(3):
            thres = 5 if i != 2 else 2.5
            num = np.array([self.item_dic[k][i] for k in self.item_dic])
            neg = (np.abs(num) <= thres).sum()
            pos = (np.abs(num) > thres).sum()
            print(i, 'neg: ', neg, 'pos: ', pos)
            times_num[i] = neg // pos

        if self.if_train:
            new_add, new_add_name = [], []
            times = max(times_num)
            for i in range(len(feature_list)):
                if abs(self.item_dic[i][0]) > 5 or abs(self.item_dic[i][1]) > 5 or abs(self.item_dic[i][2]) > 2.5:
                    new_add += [feature_list[i]] * times
                    new_add_name += [name_list[i]] * times
            feature_list += new_add
            name_list += new_add_name

        return feature_list, name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        all_feature = self.feature_list[item]
        nid_feature = random.choice(all_feature)
        feature = nid_feature[0][-1]
        if False:
            x_y = random.sample(nid_feature, k=2)
        else:
            x_y = nid_feature
        x, y = x_y[0], x_y[1]
        in_signal = [x[-3][i] for i in self.signal]
        in_pos = [p for p in x[:5]]
        out_signal = [y[-3][i] for i in self.signal]
        out_pos = [p for p in y[:5]]

        in_img_path = x[-2]
        in_img = sitk.GetArrayFromImage(sitk.ReadImage(in_img_path))[0]
        in_msk = np.zeros_like(in_img)
        in_msk[in_pos[1]: in_pos[3], in_pos[0]: in_pos[2]] = 1
        in_img[in_img > 500] = 500
        in_img[in_img < -1024] = -1024
        in_img = (in_img + 1024) / (500 + 1024)
        in_img = np.array([in_img, in_msk])

        x0, y0, z0 = (in_pos[0] + in_pos[2]) / 2, (in_pos[1] + in_pos[3]) / 2, in_pos[4]
        x1, y1, z1 = (out_pos[0] + out_pos[2]) / 2, (out_pos[1] + out_pos[3]) / 2, out_pos[4]
        label = [0, 0, 0]
        if x0 - x1 < -5:
            label[0] = 1
        elif x0 - x1 > 5:
            label[0] = 2
        if y0 - y1 < -5:
            label[1] = 1
        elif y0 - y1 > 5:
            label[1] = 2
        if z0 - z1 < -2.5:
            label[2] = 1
        elif z0 - z1 > 2.5:
            label[2] = 2

        in_pos = [p / 10 for p in x[:5]]
        out_pos = [p / 10 for p in y[:5]]
        input = in_pos + feature + in_signal + out_signal

        return torch.from_numpy(np.array(input)), torch.from_numpy(in_img), \
               torch.from_numpy(np.array(label)), torch.from_numpy(np.array(out_pos)),  name


class FeatureData2(Dataset):
    def __init__(self, feature_list, name_list, signal=[0], if_train=True):
        # feature, signal, position
        self.feature_list = feature_list
        self.name_list = name_list
        self.signal = signal
        self.if_train = if_train
        if True:
            self.feature_list, self.name_list = self.valid_data_process_2()

    def valid_data_process_2(self):
        delta = [[], [], []]
        feature_list, name_list = [], []
        self.item_dic = {}
        item = 0
        for i in range(len(self.name_list)):
            name = self.name_list[i]
            for nid in (self.feature_list[i]):
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
                    delta_x = int(x0 - x2)
                    delta_y = int(y0 - y2)
                    delta_z = int(z0 - z2)
                    delta[0].append(delta_x)
                    delta[1].append(delta_y)
                    delta[2].append(delta_z)
                    if abs(delta_z) >= 20: continue

                    feature_list.append([[nid_0, nid_1, nid_2]])
                    name_list.append(name)
                    self.item_dic[item] = [delta_x, delta_y, delta_z]
                    item += 1

        for i in range(3):
            value_counts = Counter(delta[i])
            unique_values, counts = zip(*value_counts.items())
            plt.bar(unique_values, counts, color='skyblue')
            plt.title('Value Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.savefig('./' + str(i) + '.jpg')
            plt.clf()

        times_num = [0, 0, 0]
        for i in range(3):
            thres = 5 if i != 2 else 2.5
            num = np.array([self.item_dic[k][i] for k in self.item_dic])
            neg = (np.abs(num) <= thres).sum()
            pos = (np.abs(num) > thres).sum()
            print(i, 'neg: ', neg, 'pos: ', pos)
            times_num[i] = neg // pos

        if self.if_train:
            new_add, new_add_name = [], []
            times = max(times_num)
            for i in range(len(feature_list)):
                if abs(self.item_dic[i][0]) > 5 or abs(self.item_dic[i][1]) > 5 or abs(self.item_dic[i][2]) > 2.5:
                    new_add += [feature_list[i]] * times
                    new_add_name += [name_list[i]] * times
            feature_list += new_add
            name_list += new_add_name

        return feature_list, name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        all_feature = self.feature_list[item]
        nid_feature = random.choice(all_feature)
        feature = nid_feature[0][-1]
        if False:
            x_y = random.sample(nid_feature, k=2)
        else:
            x_y = nid_feature
        x0, x1, y = x_y[0], x_y[1], x_y[2]
        in_signal = [x0[-3][i] for i in self.signal] + [x0[-3][i] for i in self.signal]
        in_pos = [p for p in x0[:5]] + [p for p in x1[:5]]
        out_signal = [y[-3][i] for i in self.signal]
        out_pos = [p for p in y[:5]]

        in_img_path = x0[-2]
        in_img = sitk.GetArrayFromImage(sitk.ReadImage(in_img_path))[0]
        in_msk = np.zeros_like(in_img)
        in_msk[in_pos[1]: in_pos[3], in_pos[0]: in_pos[2]] = 1
        in_img[in_img > 500] = 500
        in_img[in_img < -1024] = -1024
        in_img = (in_img + 1024) / (500 + 1024)
        in_img = np.array([in_img, in_msk])

        _x0, _y0, _z0 = (in_pos[0] + in_pos[2]) / 2, (in_pos[1] + in_pos[3]) / 2, in_pos[4]
        _x1, _y1, _z1 = (out_pos[0] + out_pos[2]) / 2, (out_pos[1] + out_pos[3]) / 2, out_pos[4]
        label = [0, 0, 0]
        if _x0 - _x1 < -5:
            label[0] = 1
        elif _x0 - _x1 > 5:
            label[0] = 2
        if _y0 - _y1 < -5:
            label[1] = 1
        elif _y0 - _y1 > 5:
            label[1] = 2
        if _z0 - _z1 < -2.5:
            label[2] = 1
        elif _z0 - _z1 > 2.5:
            label[2] = 2

        in_pos = [p / 10 for p in x0[:5]] + [p / 10 for p in x1[:5]]
        out_pos = [p / 10 for p in y[:5]]
        input = in_pos + feature + in_signal + out_signal

        return torch.from_numpy(np.array(input)), torch.from_numpy(in_img), \
               torch.from_numpy(np.array(label)), torch.from_numpy(np.array(out_pos)),  name

def L1_loss(pred, out_pos, input, index=None):
    if index is not None:
        pred, out_pos, input = pred[:, index], out_pos[:, index], input[:, index]
    loss = torch.abs(pred + input.detach() - out_pos).mean()
    return loss

def Cross_Entropy_loss(pred, label):
    loss = 0
    for i in range(3):
        _pred = torch.softmax(pred[:, i*3:(i+1)*3], dim=-1)
        _label = label[:, i].long()
        _loss = F.cross_entropy(_pred, F.one_hot(_label, num_classes=3).float())
        # print(_pred, F.one_hot(_label, num_classes=3), _loss)
        loss += _loss
    return loss.mean() / 3

def write_result(name, pred, label, file_name):
    csv_name = file_name + '.csv'
    with open(os.path.join('./output/csv', csv_name), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'label', 'pred'])
        for i in range(len(name)):
            writer.writerow([name[i], label[i], pred[i]])
    return

def train(signal=[0]):
    max_epoches = 200
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2, 3"

    data_list = read_data_json('./json/new_norm_data_dict_noremove.json')
    random.seed(142)
    random.shuffle(data_list)
    train_list = data_list[:int(len(data_list)*0.8)]
    valid_list = data_list[int(len(data_list)*0.8):]
    train_name = [d[0] for d in train_list]
    valid_name = [d[0] for d in valid_list]
    train_list = [d[1] for d in train_list]
    valid_list = [d[1] for d in valid_list]
    train_data = FeatureData2(train_list, train_name, signal=signal, if_train=False)
    train_data_t = FeatureData2(train_list, train_name, signal=signal, if_train=False)
    valid_data = FeatureData2(valid_list, valid_name, signal=signal, if_train=False)
    net = FeatureMLP(len(signal) * 2 + 38, None) #32, 38

    train_dataloader = DataLoader(dataset=train_data, batch_size=9, shuffle=True, num_workers=3, drop_last=True)
    valid_dataloader = DataLoader(dataset=train_data_t, batch_size=1, shuffle=False, num_workers=1)

    max_step = len(train_data) * max_epoches
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    # resume
    weights_dict = torch.load(os.path.join('./saved_model', 'signal0_time2/69_0.19354655.pth'))
    net.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(net).cuda()
    best_acc = 10**5
    for ep in range(0, max_epoches):
        model.train()
        for iter, pack in enumerate(train_dataloader):
            break
            input, img, label, out_pos, _ = pack[0].cuda().float(), pack[1].cuda().float(), pack[2].cuda().float(), \
                                           pack[3].cuda().float(), pack[4]
            pred, pred_l = model(input, img)
            loss_ce = Cross_Entropy_loss(pred, label)
            loss_l1 = L1_loss(pred_l, out_pos, input[:, :5], index=4)
            loss = loss_ce * 0.0 + loss_l1
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_data), '/', max_step,
                      'loss ce:', loss_ce.item(), 'loss l1:', loss_l1.item())

        print('')
        if ep % 1 == 0:
            acc = validation(valid_dataloader, model, signal)
            return
            print('')
            if acc < best_acc:
                torch.save(model.module.state_dict(),
                           os.path.join('./saved_model', str(ep) + '_' + str(acc) + '.pth'))
                best_acc = acc

def validation(valid_dataloader, model, signal):
    model.eval()
    pred_list, label_list = [[], [], []], [[], [], []]
    pred_l1, ori_l1 = [[], [], []], [[], [], []]
    outpred, out_label, name_list = [], [], []
    for iter, pack in enumerate(valid_dataloader):
        input, img, label, out_pos, name = pack[0].cuda().float(), pack[1].cuda().float(), pack[2].cuda().float(), \
                                           pack[3].cuda().float(), pack[4]
        pred, pred_l = model(input, img)
        name_list += list(name)
        for i in range(3):
            _pred = torch.softmax(pred[:, i*3:(i+1)*3], dim=-1).detach().cpu().numpy()
            _label = label[:, i].detach().cpu().numpy()
            pred_list[i] += list(np.argmax(_pred, axis=1))
            label_list[i] += list(_label)

        pred_x = (pred_l[:, 0] + pred_l[:, 2]).detach().cpu().numpy() / 2
        pred_y = (pred_l[:, 1] + pred_l[:, 3]).detach().cpu().numpy() / 2
        pred_z = pred_l[:, 4].detach().cpu().numpy()
        ori_x = (input[:, 0] + input[:, 2]).detach().cpu().numpy() / 2
        ori_y = (input[:, 1] + input[:, 3]).detach().cpu().numpy() / 2
        ori_z = input[:, 4].detach().cpu().numpy()
        label_x = (out_pos[:, 0] + out_pos[:, 2]).detach().cpu().numpy() / 2
        label_y = (out_pos[:, 1] + out_pos[:, 3]).detach().cpu().numpy() / 2
        label_z = out_pos[:, 4].detach().cpu().numpy()
        pred_l1[0] += list((pred_x + ori_x - label_x)**2)
        pred_l1[1] += list((pred_y + ori_y - label_y)**2)
        pred_l1[2] += list((pred_z + ori_z - label_z)**2)
        ori_l1[0] += list((ori_x - label_x)**2)
        ori_l1[1] += list((ori_y - label_y)**2)
        ori_l1[2] += list((ori_z - label_z)**2)

        outpred += list((pred_z + ori_z) )
        out_label += list(label_z)

    avg_acc = 0
    for i in range(3):
        acc = accuracy_score(label_list[i], pred_list[i])
        print('acc: ',  acc)
        avg_acc += acc

        print('l1: ', 'pred', np.mean(pred_l1[i]), 'ori', np.mean(ori_l1[i]))

    # write_result(name_list, outpred, out_label, 'dl_throx')
    return np.mean(pred_l1[-1])


if __name__ == '__main__':
    train()




































