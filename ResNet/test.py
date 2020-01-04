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
import pandas as pd
import csv

import resnet

class test_data_prepare(data.Dataset):
    def __init__(self,csv_path,img_path):
        self.img_path = img_path
        self.csv_path = csv_path
        self.img_rows = 32
        self.img_cols = 32
        self.img_dims = 32
        
        self.length=0
        self.img =[]
        self.data_get()

    def data_get(self):
        self.img = read.new_test(self.csv_path,self.img_path)
        self.length=len(self.img)

    def __getitem__(self, index):
        data = []
        data.append(self.img[index])
        return torch.from_numpy(np.array(data)).float()

    def __len__(self):
        return self.length

def reload(model, checkpoint):
    new_state_dict = model.state_dict()
    for k in checkpoint.keys():
        if k in new_state_dict.keys():
            new_state_dict[k] = checkpoint[k]
    model.load_state_dict(new_state_dict)

    return model

if __name__ == "__main__":
    csv_path = 'submission.csv'
    test_path = "../test"
    pretrained_path = 'model.pth'
    final_path = 'submission.csv'
    is_pretrained = True
    batch_size = 24

    test_set = test_data_prepare(csv_path,test_path)
    test_loader = data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
        )
    
    model, parameters = generate_model(models='resnet', model_depth=200, n_classes=2, shortcut='B', sample_size=32, sample_duration=32, pretrain_path=False, ft_begin_index=0)
    
    if is_pretrained:
        assert os.path.isfile(pretrained_path), "Model does not exist!"
        checkpoint = torch.load(pretrained_path)
        model = reload(model, checkpoint['state_dict'])
        
    criterion = nn.CrossEntropyLoss(size_average=True)
        
    model.cuda().eval()
    predicted=[]
    for i, data in enumerate(test_loader):
        data = data.cuda()
        # compute output
        output = model(data).cpu().detach()
        predict=F.softmax(output,dim=1).cpu().numpy()
        for meb in predict:
            predicted.append(meb[1])
    csv_file = csv.reader(open(csv_path))
    name = []
    for cont in csv_file:
        if cont[0]!='Id':
            name.append(cont[0])
    dataframe = pd.DataFrame({'Id':name,'Predicted':predicted})
    dataframe.to_csv(final_path,index=False)
    
    
    
