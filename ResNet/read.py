import numpy as np
from matplotlib import pyplot as plt
import os
import csv
from sklearn.model_selection import train_test_split
from mylib import rotation, reflection, crop, random_center, _triple

class Transform:
    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
#读取voxel和seg
def train_load(filepath):
    files_name = os.listdir(filepath) 
    files_name.sort()
    voxel = []
    seg = []
    for i in files_name:
        tmp = np.load(filepath+'/'+i)
        voxel.append(tmp['voxel'])
        seg.append(tmp['seg'])
    return np.array(voxel),np.array(seg)
    
#读取标签
def label_load(filepath):
    csv_file = csv.reader(open(filepath))
    label = []
    for cont in csv_file:
        if cont[1]!='lable':
            label.append(int(cont[1]))
    return np.array(label)
    
def test_load(csv_filepath,filepath):
    csv_file = csv.reader(open(csv_filepath))
    name = []
    for cont in csv_file:
        if cont[0]!='Id':
            name.append(cont[0])
    voxel = []
    seg = []
    for i in name:
        tmp = np.load(filepath+'/'+i+'.npz')
        voxel.append(tmp['voxel'])
        seg.append(tmp['seg'])
    return np.array(voxel),np.array(seg)
    
#训练数据读取，数据增强
def new_train(filepath):
    voxel_train,seg_train = train_load(filepath)
    train_del = []
    train_cropedtrans = []
    for i in range(len(voxel_train)):
        seg_train[i] = seg_train[i].astype(int)
        deleted = np.multiply(voxel_train[i],seg_train[i]).astype(np.float32)
        deleted/=128.-1.
        train_del.append(deleted)#对应元素相乘，抠出节点
    
    for i in range(len(train_del)):
        train_cropedtrans.append(crop(train_del[i],(50,50,50),(32,32,32)))
    
    for i in range(len(train_del)):
        train_cropedtrans.append(np.squeeze(Transform(32,5)(train_del[i])))
    return np.array(train_cropedtrans)
    
def new_label(filepath):
    train_label = label_load(filepath)
    train_label = np.hstack((train_label,train_label))
    return(train_label)

def new_test(csv_filepath,filepath):
    #测试数据读取
    voxel_test,seg_test = test_load(csv_filepath,filepath)
    test_del = []
    test_croped = []
    for i in range(len(voxel_test)):
        seg_test[i] = seg_test[i].astype(int)
        deleted = np.multiply(voxel_test[i],seg_test[i]).astype(np.float32)
        deleted/=128.-1.
        test_del.append(deleted)#对应元素相乘，抠出节点
    
    for i in range(len(test_del)):
        test_croped.append(crop(test_del[i],(50,50,50),(32,32,32)))
    return np.array(test_croped)
    
def new_data(filepath1,filepath2):
    X_train_new = new_train(filepath1)
    train_label = new_label(filepath2)
    train,val,train_label,val_label =train_test_split(X_train_new,train_label,test_size=80,shuffle=True)
    return train,val,train_label,val_label



