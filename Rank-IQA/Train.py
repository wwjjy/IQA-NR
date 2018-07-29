from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, Input
from keras.models import Model
import random
import numpy as np
import h5py
import xlrd
from ResNet import ResNet
from svr import srocc


#
model_path = 'my_model.h5'
my_model = ResNet(model_path)
my_model.show_summary()

# basemodel.summary()
# mymodel.summary()

# 下来读取预训练的数据
data = xlrd.open_workbook('结果.xlsx')
table = data.sheets()[6]
nrows = table.nrows
ncols = table.ncols

raw_data = []
for i in range(1, nrows):
    raw_data.append(table.row_values(i, 0, 2))
random.shuffle(raw_data)

train = raw_data[0:]
test = raw_data[70:]

# print(train)

train_data = []
for i in train:
    train_data.append([i[0], i[1]])
print(train_data)
# print(train_lable)

# print(train_data[0][0])
# 这里确定文件名和等级没问题之后，开始读取数据
# 这里先确定文件所在的路径
File_path = u'C:/Users/SAMA/Desktop/code/all/'
# 这里再逐一读取文件并分块以及对应标签
finetune_data = []
finetune_label = []
for i in train_data:
    # 合成文件的名字
    file = File_path + i[0] + '.mat'
    print(file)
    # 下一步读取其中图像数据
    feature = h5py.File(file)
    train_set_data = feature['image']['oct'][:]

    # 读取数据后进行校正
    train_set_data = train_set_data.T
    train_set_data = 255 - (train_set_data ** 0.3 * 255)
    # print(train_set_data.shape)

    # 数据进行分块并与对应标签一起保存
    for k in np.arange(0, 480, 32):
        for j in np.arange(0, 1536, 32):
            finetune_data.append(train_set_data[k: k + 32, j: j + 32].reshape(32 * 32))
            if i[1] == 1:
                finetune_label.append([1, 0, 0])
            elif i[1] == 2:
                finetune_label.append([0, 1, 0])
            else:
                finetune_label.append([0, 0, 1])
finetune_label = np.array(finetune_label)

# 开始准备finetune
finetune_data = np.array(finetune_data)
finetune_data = finetune_data.reshape(finetune_data.shape[0], 32, 32, 1)
epoch = 100001
ID = np.arange(0, finetune_data.shape[0]).tolist()
for i in range(epoch):
    train_data_ID = random.sample(ID, 50)
    acc = my_model.train_on_batch(finetune_data[train_data_ID], finetune_label[train_data_ID])
    if (i+1) % 100 == 0:

        my_model.save_parameter(model_path)
        print("epoch %d, acc= %f" % (i + 1, acc[1]))
        srocc()
