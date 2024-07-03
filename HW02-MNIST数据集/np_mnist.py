# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""
'''
@ student: Zhuoyang Liu
'''
#作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集
#完成方法：采用numpy实现MLP网络，使用交叉熵作为损失函数，使用sigmoid作为激活函数，使用softmax输出层
import numpy as np
from  tqdm  import tqdm
'''
One-hot编码是一种将类别型数据转化为数值型数据的常用方法，
主要用于机器学习和深度学习领域的数据预处理阶段。
它的基本思想是为每一个类别分配一个独特的二进制向量，
该向量除了表示当前类别的位置为1（即“热”）之外，其余所有位置均为0。
'''
# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码

# 定义激活函数，可用的有sigmoid、relu、softmax
def sigmoid(x):
    '''
    sigmoid函数
    '''
    return 1. / (1. + np.exp(-x))

def sigmoid_prime(x):
    '''
    sigmoid函数的导数
    '''
    return sigmoid(x) * (1. - sigmoid(x))

def relu(x):
    '''
    relu函数
    '''
    return np.maximum(0, x)

def relu_prime(x):
    '''
    relu函数的导数
    '''
    return (x > 0).astype(np.float)

#输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    '''
    return np.exp(x)/np.sum(np.exp(x))

def f_prime(x):
    '''
    softmax函数的导数
    '''
    return f(x)*(1-f(x))

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    #y_pred是softmax输出，softmax输出的和为1
    eps = 1e-7 # 防止log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps) # 确保y_pred在(eps, 1 - eps)范围内
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return y_pred - y_true

# 定义权重初始化函数
def init_weights(shape):
    '''
    初始化权重
    '''
    if isinstance(shape, tuple) and len(shape) > 0:
        fan_in = shape[0]
    else:
        raise ValueError("Shape must be a non-empty tuple.")
        
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/fan_in), size=shape)

count=0
total=0
# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        '''
        初始化网络结构
        '''
        #input_size=784, hidden_size=256, output_size=10, lr=0.01
        self.W1 = init_weights((input_size, hidden_size))#784*256
        self.b1 = init_weights((1,hidden_size))  #1*256
        self.W2 = init_weights((hidden_size, output_size))#256*10
        self.b2 = init_weights((1,output_size))  #1*10
        self.lr = lr#学习率
    def forward(self, x):
        '''
        前向传播
        self.z1  = np.matmul(x, self.W1) + self.b1 # z1:1*256
        self.a1  = sigmoid(self.z1) # a1:1*256
        self.z2 = np.matmul(self.a1, self.W2) + self.b2 # z2:1*10
        self.a2 = f(self.z2)
        return self.a2
        '''

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        global count
        global total
        self.shapex=x_batch.shape
        self.shapey=y_batch.shape
        #print("x的形状：",self.shapex)
        #print("y的形状：",self.shapey)
        batch_size = 0
        batch_loss = 0
        batch_acc = 0
        #初始化三个参数
        self.grads_W2 = np.zeros_like(self.W2)
        self.grads_b2 = np.zeros_like(self.b2)
        self.grads_W1 = np.zeros_like(self.W1)
        self.grads_b1 = np.zeros_like(self.b1)

        for x, y in zip(x_batch, y_batch):
            #x_batch:64*784,y_batch:64*10
            #x:1*784,y:1*10
            ## forward
            #self.w1:784*256
            self.z1  = np.matmul(x, self.W1) + self.b1 # z1:1*256
            self.a1  = sigmoid(self.z1) # a1:1*256
            #self.w2:256*10
            # z2:1*10
            self.z2 = np.matmul(self.a1, self.W2) + self.b2 
            # a2:1*10
            self.a2 = f(self.z2) 
            ##getloss
            #y:1*10
            self.loss= loss_fn(y, self.a2)#y:1*10, self.a2:1*10
            #self.delta_L:1*10
            self.delta_L = loss_fn_prime(y, self.a2)
            ##backward
            #self.w2:#256*10
            #self.delta_L:#1*10
            #self.a1:#1*256
            #self.delta_l:1*256
            self.delta_l = np.matmul(self.delta_L,(self.W2).T) * self.a1 * (1-self.a1)
            ##addgrads
            #self.a1:#1*256
            #self.delta_L:1*10
            #self.grads_W2:256*10
            self.grads_W2+=np.matmul((self.a1).T, self.delta_L)
            #self.grads_b2:#1*10
            #self.delta_L:#1*10
            self.grads_b2 += self.delta_L
            #self.grads_W1:#784*256
            #self.delta_l:#1*256
            #x:1*784
            self.shapex=np.array([x]).shape
            self.shapey=self.delta_l.shape
            #print("x的形状：",self.shapex)
            #print("self.delta_l的形状：",self.shapey)
            self.grads_W1 += np.matmul(np.array([x]).T, self.delta_l)
            #print("self.grads_W1:",end="")
            #print(self.grads_W1)
            #self.grads_b1:#1*256
            #self.delta_l:#1*256
            self.grads_b1 += self.delta_l
            #print("self.grads_b1:",end="")
            #print(self.grads_b1)
            ##update    
            batch_size += 1
            batch_loss += self.loss
            self.index_y = np.argmax(y)
            self.index_max = np.argmax(self.a2)
            if self.index_y == self.index_max:
                batch_acc+=1
                count+=1
                total+=1
            else:
                batch_acc+=0
                total+=1
            #batch_acc += 1 if (y == (1 if self.a2>0.5 else 0)) else 0
        ##grad_average
        self.grads_W2 /= batch_size
        self.grads_b2 /= batch_size
        self.grads_W1 /= batch_size
        self.grads_b1 /= batch_size
        ##loss_average
        batch_loss /= batch_size
        batch_acc /= batch_size
        #print("loss:{} batch_acc:{} batch_size:{} lr:{}".format(batch_loss, batch_acc, batch_size, self.lr))
        ##update_weights
        self.W2 -= self.lr * self.grads_W2
        self.b2 -= self.lr * self.grads_b2
        self.W1 -= self.lr * self.grads_W1
        self.b1 -= self.lr * self.grads_b1
    def test(self, x_batch, y_batch):
        '''
        测试
        '''
        global count
        global total
        self.correct_count=0
        self.total_count=0
        for x, y in zip(x_batch, y_batch):
            #前向传播，得出预测值
            self.z1  = np.matmul(x, self.W1) + self.b1 # z1:1*256
            self.a1  = sigmoid(self.z1) # a1:1*256
            self.z2 = np.matmul(self.a1, self.W2) + self.b2 
            self.a2 = f(self.z2) 
            #预估值与真实值对比：
            self.index_y = np.argmax(y)
            self.index_max = np.argmax(self.a2)
            if self.index_y == self.index_max:
                count+=1
                self.correct_count+=1
                total+=1
                self.total_count+=1
            else:
                count+=0
                self.total_count+=1
                total+=1
        return self.correct_count/self.total_count


if __name__ == '__main__':
    ##training_network
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.01)
    for epoch in range(10):
        count=0
        total=0
        losses = []
        accuracies = []
        val_accuracies = []
        p_bar = tqdm(range(0, len(X_train), 1))
        for i in p_bar:
            # 获取一个批次的训练数据
            batch_X = X_train[i:i+1]#每次选取1个训练数据进行训练
            batch_y = y_train[i:i+1]
            #print("epoch:",epoch)
            #print("i:",i)
            net.step(batch_X, batch_y)#训练一次，batch_x为64个训练数据，batch_y为64个标签
            ##update_loss
            losses.append(net.loss)
            ##update_accuracy
            accuracies.append(net.a2)
        avg_train_loss = np.mean(losses)
        avg_train_accuracy = count/total
        print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}")
        count=0
        total=0
        for i in range(len(X_val)):
        # 获取一个批次的测试数据
            batch_X = X_val[i:i+1]#每次选取1个训练数据进行测试
            batch_y = y_val[i:i+1]
            test_result=net.test(batch_X,batch_y)
            val_accuracies.append(test_result)  # 假设 net.test 返回的是准确率
        avg_val_accuracy = count/total
        print(f"Epoch: {epoch + 1}, Val Acc: {avg_val_accuracy:.4f}")
    #实测测试集准确率达到97.38%，超过了94%的准确率，说明模型训练效果不错。
p_bar = tqdm(range(0, len(X_test), 1))
count=0
total=0
for i in p_bar:
    # 获取一个批次的训练数据
    batch_X = X_train[i:i+1]#每次选取1个训练数据进行训练
    batch_y = y_train[i:i+1]
    test_result=net.test(batch_X,batch_y)
    val_accuracies.append(test_result)  # 假设 net.test 返回的是准确率
avg_train_loss = np.mean(losses)
avg_train_accuracy = count/total
print(f"Test:, Test Loss: {avg_train_loss:.4f}, Test Acc: {avg_train_accuracy:.4f}")