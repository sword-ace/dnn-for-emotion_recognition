# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import random
import time
import torch
import torch.nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import math
import helper
import detectiveNN

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed=2021):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_dataset_loaders(path=None, valid_rate=0.1, batch_size=32, num_workers=0, pin_memory=False):
    trainset = Dataset_process(path=path, train=True)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = Dataset_process(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, feature_type='text', target_names=None,
                        tensorboard=False):
    assert not train_flag or optimizer != None
    losses, preds, labels = [], [], []
    if train_flag:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train_flag: optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() if isinstance(d, torch.Tensor) else d for d in data[:-1]] if cuda_flag else data[:-1]


        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if feature_type == 'multi':
            dataf = torch.cat((textf, acouf, visuf), dim=-1)
        elif feature_type == 'text':
            # dataf = textf
            dataf = textf
        elif feature_type == 'acouf':
           dataf = acouf

        else:
            dataf = torch.cat((textf, acouf), dim=-1) #acouf


        log_prob = model(dataf, qmask, seq_lengths)
        label = torch.cat([label[j][:seq_lengths[j]] for j in range(len(label))])
        loss = loss_f(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            loss.backward()
            
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    labels = np.array(labels)
    preds = np.array(preds)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(metrics.classification_report(labels, preds, target_names=target_names, digits=4))
    all_matrix.append(["ACC"])
    for i in range(len(target_names)):
        all_matrix[-1].append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds]



lr = 0.001 
l2 = 0.002
dropout = 0.5
epochs = 80
batch_size = 30 
valid_rate = 0
patience = 10
status = 'train'
output_dir = '/content/output'

data_path = '/content/drive/MyDrive/DetectiveNN/data/iemocap/iemocap_data.pkl'


output_path = '/content/model_out'
base_model = 'GRU' 
base_layer = 1
feature_type = 'multi' 
no_cuda = False
class_weight = True
seed = 100 


cuda_flag = torch.cuda.is_available() and not no_cuda

#  dataset intro
n_classes, n_speakers, hidden_size, input_size = 6, 2, 100, None
target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']

if feature_type == 'multi':
    input_size = 456 #1024
elif feature_type == 'acouf':
    input_size = 100
elif feature_type == 'text':
    input_size = 100
elif feature_type == 'v':
    input_size = 256
else:
    input_size = 100
    # print('Error: feature_type not set.')
    # exit(0)

seed_everything(seed)
model = DetectiveNN(base_model=base_model,
                    base_layer=base_layer,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_speakers=n_speakers,
                    n_classes=n_classes,
                    dropout=dropout,
                    cuda_flag=cuda_flag)

#######################################part for modify loss function ###########################################
class_weights = torch.FloatTensor([1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089, 1 / 0.148392305, 1 / 0.140051123, 1 / 0.24875555])


if cuda_flag:
    print('Running on GPU')
    # torch.cuda.set_device(0) # test
    class_weights = class_weights.cuda()
    model.cuda()
else:
    print('Running on CPU')
loss_f = FocalLoss(gamma=gamma, alpha=class_weights if class_weight else None)


name = 'DetectiveNN'
print('{} with {} as base model'.format(name, base_model))
print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
print('Running on the {} features........'.format(feature_type))
print("load the best model....")





optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
train_loader, valid_loader, test_loader = get_dataset_loaders(path=data_path, valid_rate=valid_rate, batch_size=batch_size, num_workers=0)

# if status == 'train':
all_test_fscore, all_test_acc = [], []
all_train_fscore, all_train_acc = [], []
# all_valid_fscore, all_valid_acc = [], []
best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
patience2 = 0
for e in range(epochs):
    start_time = time.time()

    train_loss, train_acc, train_fscore, _, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=train_loader, epoch=e, train_flag=True,
                                                                    optimizer=optimizer, cuda_flag=cuda_flag, feature_type=feature_type,
                                                                    target_names=target_names)
    valid_loss, valid_acc, valid_fscore, _, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=valid_loader, epoch=e, cuda_flag=cuda_flag,
                                                                    feature_type=feature_type, target_names=target_names)
    test_loss, test_acc, test_fscore, test_metrics, _ = train_or_eval_model(model=model, loss_f=loss_f, dataloader=test_loader, epoch=e,
                                                                            cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names)
    all_test_fscore.append(test_fscore)
    all_test_acc.append(test_acc)
    # all_valid_fscore.append(valid_fscore)
    # all_valid_acc.append(valid_acc)
    all_train_fscore.append(train_fscore)
    all_train_acc.append(train_acc)


    if valid_rate > 0:
        eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
    else:
        eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore
    if e == 0 or best_eval_fscore < eval_fscore:
        patience = 0
        best_epoch, best_eval_fscore = e, eval_fscore
        if not os.path.exists(output_path): os.makedirs(output_path)
        save_model_dir = os.path.join(output_path, 'f1_{}_{}.pkl'.format(name, e).lower())
        torch.save(model.state_dict(), save_model_dir)
        model.load_state_dict(torch.load(save_model_dir))
    else:
        patience += 1

    if best_eval_loss is None:
        best_eval_loss = eval_loss
        best_epoch2 = 0
    else:
        if eval_loss < best_eval_loss:
            best_epoch2, best_eval_loss = e, eval_loss
            patience2 = 0
            if not os.path.exists(output_path): os.makedirs(output_path)
            save_model_dir = os.path.join(output_path, 'best_{}_{}.pkl'.format(name, e).lower())
            torch.save(model.state_dict(), save_model_dir)
            model.load_state_dict(torch.load(save_model_dir))
            # checkpoint = torch.load(os.path.join(model.state_dict(), f'{best_epoch2}.pkl'))
            # model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch = checkpoint['epoch']
         # If there's a previously saved best model, delete it
            # if save_model_dir and os.path.isfile(save_model_dir):
                # os.remove(save_model_dir)
        else:
            patience2 += 1


    print(
        'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore,
                    round(time.time() - start_time, 2)))
    print(test_metrics[0])
    print(test_metrics[1])
    print('\n')

    if patience >= patience and patience2 >= patience:
         print('Early stoping...', patience, patience2)
         break
    if patience >= patience2:
         print('Early stoping...', patience, patience2)
         break

print('Final Test performance...')
print('Early stoping...', patience, patience2)
print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch, best_eval_fscore,                                                                                         all_test_acc[best_epoch] if best_epoch >= 0 else 0,                                                                                          all_test_fscore[best_epoch] if best_epoch >= 0 else 0))
print('Eval-metric: Loss, Epoch: {}, Accuracy: {}, F1-Score: {}'.format(best_epoch2,
                                                                            all_test_acc[best_epoch2] if best_epoch2 >= 0 else 0,
                                                                            all_test_fscore[best_epoch2] if best_epoch2 >= 0 else 0))