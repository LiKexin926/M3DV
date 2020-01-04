# M3DV
当前共两个文件夹，运行代码时需要把测试集、训练集放在相对代码的上一级文件中。

# ResNet:
该文件存放使用ResNet模型的代码，由于模型过大，存放在了https://pan.baidu.com/s/1WB72GWAI2kSb9cjqfS-G-w 中，下载后放在ResNet文件夹下命名为'model.pth'即可。

运行train.py训练网络，并输出其每一代的loss和在训练集、测试集上的精确度。（为了确保顺利训练，需要创建空文件夹'model'和空文本'acc_train.txt'和'acc_test.txt'）

运行test.py，测试在test集上的表现。输出结果存放在submission.csv中。
