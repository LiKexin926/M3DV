#-*- coding: UTF-8 -*-
from model import generate_model
import numpy as np
import read
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os
from sklearn import metrics
import math
import resnet

class data_prepare(data.Dataset):
    def __init__(self,img_path,label_path):
        '''
        :在这初始化一些参数
        '''
        self.img_path = img_path#图像读取路径
        self.label_path = label_path#标记读取路径
        #图像尺寸
        self.img_rows = 32
        self.img_cols = 32
        self.img_dims = 32
        
        self.length=0#数据长度
        self.img=[]#数据存储
        self.label=[]#标记存储
        self.data_get()#数据获取

    def data_get(self):
        #数据获取
        self.img,a,self.label,b=read.new_data(self.img_path,self.label_path)
        self.length=len(self.label)

    def __getitem__(self, index):#读取数据
        data = []
        label = []
        data.append(self.img[index])#读取第index个图像
        label.append(self.label[index])#读取第index个标记
        return torch.from_numpy(np.array(data)).float(),torch.from_numpy(np.array(label)).long()
        
    def __len__(self):
        #返回数据长度
        return self.length

class test_data_prepare(data.Dataset):
    def __init__(self,img_path,label_path):
        '''
        :在这初始化一些参数
        '''
        self.img_path = img_path#图像读取路径
        self.label_path = label_path#标记读取路径
        #图像尺寸
        self.img_rows = 32
        self.img_cols = 32
        self.img_dims = 32
        
        self.length=0#数据长度
        self.img=[]#数据存储
        self.label=[]#标记存储
        self.data_get()#数据获取

    def data_get(self):
        #数据获取
        a,self.img,b,self.label=read.new_data(self.img_path,self.label_path)
        self.length=len(self.label)

    def __getitem__(self, index):#读取数据
        data = []
        label = []
        data.append(self.img[index])#读取第index个图像
        label.append(self.label[index])#读取第index个标记
        return torch.from_numpy(np.array(data)).float(),torch.from_numpy(np.array(label)).long()
        
    def __len__(self):
        #返回数据长度
        return self.length
        
def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.cuda().train()
    for i, (data,label) in enumerate(train_loader):
        # if is_mixup:
        #     inputs, targets_a, targets_b, lam = mixup_data(data.cuda(), labels.cuda(), alpha=1.0)
        #     inputs, targets_a, targets_b = map(Variable, (inputs,
        #                                                   targets_a, targets_b))
        #     outputs = model(inputs)
        #     optimizer.zero_grad()
        #     loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        #     loss.backward()
        #     optimizer.step()
        # else:
        data_cuda = data.cuda()
        label_cuda = label.cuda()
        # compute output
        outputs = model(data_cuda)
        loss = criterion(outputs,label_cuda.squeeze())
        print('In epoch {}, batch {}, loss is {}'.format(epoch, i, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def save_checkpoint(states,  output_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    
if __name__ == "__main__":
    data_path = "../train_val"
    label_path = "../train_val.csv"
    test_path = "../train_val"
    test_label="../train_val.csv"
    final_output_dir="model"
    
    num_epoches = 60
    batch_size = 30
    learning_rate = 0.001
    
    data_set = data_prepare(data_path,label_path)
    test_set = test_data_prepare(test_path,test_label)
    train_loader = data.DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
        )
    
    model, parameters = generate_model(models='resnet', model_depth=200, n_classes=2, shortcut='B', sample_size=32, sample_duration=32, pretrain_path=False, ft_begin_index=0)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss(size_average=True)
    file1=open('acc_train.txt','a')
    file2=open('acc_test.txt','a')
    for epoch in range(num_epoches):
        # if num_epoches > 80:
        #     is_mixup = False
        # else:
        #     is_mixup = True
        print('epoch:'+str(epoch))
        train(train_loader, model, criterion, optimizer, epoch)
        
        model.cuda().eval()
        label_all=[]
        predicted=[]
        for i, (data,label) in enumerate(train_loader):
            data = data.cuda()
            for key in label.numpy():
                label_all.append(key)
            # compute output
            output = model(data)
            predict=torch.argmax(F.softmax(output,dim=1),dim=1).cpu().numpy()
            for meb in predict:
                predicted.append(meb)
        accuracy = metrics.accuracy_score(label_all, predicted, normalize=True, sample_weight=None)
        print('Training accuracy: %0.2f%%' % (accuracy*100))
        file1.write('epoch='+str(epoch)+":"+str(accuracy*100)+'\n')
        
        label_all=[]
        predicted=[]
        for i, (data,label) in enumerate(test_loader):
            data = data.cuda()
            for key in label.numpy():
                label_all.append(key)
            # compute output
            output = model(data)
            predict=torch.argmax(F.softmax(output,dim=1),dim=1).cpu().numpy()
            for meb in predict:
                predicted.append(meb)
        accuracy = metrics.accuracy_score(label_all, predicted, normalize=True, sample_weight=None)
        print('Test accuracy: %0.2f%%' % (accuracy*100))
        file2.write('epoch='+str(epoch)+":"+str(accuracy*100)+'\n')
        if(accuracy*100>70):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, final_output_dir, str(epoch)+'.pth')
            
    file1.close()
    file2.close()
    
    