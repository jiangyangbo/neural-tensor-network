#coding=utf-8
#!/usr/bin/python

import math
import numpy as np
from sklearn.datasets import load_digits

#配置库
import torch
import torch.utils.data as Data
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


# 配置参数
torch.manual_seed(1) #设置随机数种子，确保结果可重复                                                                                                      
batch_size = 200 #批处理大小 #,100, 200
learning_rate = 1e-2  #学习率
num_epoches = 1000      #训练次数        # 1000 ,2000                                             


import pdb

def get_data():
  digits = load_digits()
  #pdb.set_trace()
  L = int(math.floor(digits.data.shape[0] * 0.2))
  X_train = digits.data[:L]
  y_train = digits.target[:L]
  X_test = digits.data[L + 1:]
  y_test = digits.target[L + 1:]
  return X_train, y_train, X_test, y_test

#from neural_tensor_layer import NeuralTensorLayer
##定义neural Tensor Layer
class NeuralTensorLayer(nn.Module):
    def __init__(self,  output_dim, input_dim):
        super(NeuralTensorLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # W : k*d*d
        #k = self.output_dim
        #d = self.input_dim
        self.weight_W = nn.Parameter(torch.Tensor(output_dim, input_dim, input_dim))
        self.weight_V = nn.Parameter(torch.Tensor(2 * input_dim, output_dim))
        #self.b = nn.Parameter(torch.Tensor(1,input_dim))
        self.b = nn.Parameter(torch.Tensor(1,output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = .1 / math.sqrt(self.input_dim)  # 1.0 0.5
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        k = self.output_dim
        batch_size = input1.size(0)
        pdb.set_trace()
        input = torch.cat((input1, input2), 1) # 纵轴拼接
        feed_forward_product = torch.mm(input, self.weight_V)
        tmp_mm = torch.mm(input1, self.weight_W[0])
        bilinear_tensor_products = Variable(torch.zeros(batch_size, k))
        bilinear_tensor_products[:, 0] = torch.sum(input2 * tmp_mm , 1)
        #print(bilinear_tensor_products)
        for i in range(k)[1:]:
            btp =torch.sum(input2  * torch.mm(input1, self.weight_W[i]), 1)
            bilinear_tensor_products[:, i] = btp
        pdb.set_trace()
        output = torch.tanh(bilinear_tensor_products + feed_forward_product + self.b)
        result = output
        return result


class Model(nn.Module):
     def __init__(self):
         super(Model, self).__init__()
         self.ntn = NeuralTensorLayer(32, 64)
         self.fc = nn.Linear(32, 16)
         self.fc2 = nn.Linear(16,10)
        
     def forward(self,  input1, input2):
        out = self.ntn(input1, input2)
        #pdb.set_trace()
        out = self.fc(out)
        out = self.fc2(out)
        #pdb.set_trace()
        return out
        
        

def main():
    #pdb.set_trace()

    X_train, Y_train, X_test, Y_test = get_data()
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.int64)
    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)

    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.int64)
    X_test =torch.from_numpy(X_test)
    Y_test=torch.from_numpy(Y_test)
    #pdb.set_trace()
    #数据的批处理，尺寸大小为batch_size, 
    #在训练集中，shuffle 必须设置为True, 表示次序是随机的
    train_dataset = Data.TensorDataset(data_tensor=X_train, target_tensor=Y_train)
    test_dataset = Data.TensorDataset(data_tensor=X_test, target_tensor=Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Model()
    print model
    
    criterion = nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 开始训练
    loss_h = []
    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1): #批处理
            #pdb.set_trace()
            img, label = data
            img = Variable(img)
            label = Variable(label)
            # 前向传播 
            out = model(img, img)
            #pdb.set_trace()
            loss = criterion(out, label) # loss 
            running_loss += loss.data[0] * label.size(0) # total loss , 由于loss 是batch 取均值的，需要把batch size 乘回去
            _, pred = torch.max(out, 1) # 预测结果
            num_correct = (pred== label).sum() #正确结果的num
            accuracy = (pred == label).float().mean() #正确率
            running_acc += num_correct.data[0] # 正确结果的总数
            # 后向传播
            optimizer.zero_grad() #梯度清零，以免影响其他batch
            loss.backward() # 后向传播，计算梯度
            optimizer.step() #梯度更新
            #print 'loss', loss

 
        #打印一个循环后，训练集合上的loss 和 正确率
        print('Train Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
         epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
                train_dataset))))
    #import matplotlib.pyplot as plt
    #plt.plot(loss_h)
    #模型测试， 由于训练和测试 BatchNorm, Dropout配置不同，需要说明是否模型测试
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:  #test set 批处理
        img, label = data
        
        img = Variable(img, volatile=True) # volatile 确定你是否不调用.backward(), 测试中不需要
        label = Variable(label, volatile=True)
        #img = torch.t(img)
        out = model(img, img)  # 前向算法 
        loss = criterion(out, label)   # 计算 loss
        eval_loss += loss.data[0] * label.size(0)  # total loss
        _, pred = torch.max(out, 1)  # 预测结果
        num_correct = (pred == label).sum()  # 正确结果
        eval_acc += num_correct.data[0] #正确结果总数

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_dataset)), eval_acc * 1.0 / (len(test_dataset))))


main()
