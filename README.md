# M3DV
当前共两个文件夹，运行代码时需要把测试集、训练集放在相对代码的上一级文件中。

# ResNet:
该文件存放使用ResNet模型的代码，由于模型过大，存放在了https://pan.baidu.com/s/1WB72GWAI2kSb9cjqfS-G-w 中，下载后放在ResNet文件夹下命名为'model.pth'即可。

read.py为读取数据集相关代码，mylib.py为数据预处理的相关代码，resnet.py和model.py为ResNet网络的相关代码。

运行train.py，训练网络，并输出其每一代的loss和在训练集、验证集上的精确度。（为了确保顺利训练，需要创建空文件夹'model'和空文本'acc_train.txt'，'acc_test.txt'）通过改变final_output_dir，可以指定不同的模型存放位置。

运行test.py，测试模型在test集上的表现。输出结果存放在submission.csv中。通过改变res_path，可以指定不同的结果存放位置。

# DenseNet：
该文件存放使用DenseNet模型的代码。model.h5是训练的得到的模型。

read.py为读取数据集相关代码，mylib.py为数据预处理的相关代码，densenet.py为DenseNet网络的相关代码。

运行train.py，训练网络，并输出其每一代的loss和在训练集、验证集上的精确度。通过改变final_output_dir，可以指定不同的模型存放位置。

运行test.py，测试模型在test集上的表现。输出结果存放在submission.csv中。通过改变res_path，可以指定不同的结果存放位置。
