# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# 定义超参数
batch_size = 128 #批处理大小128
learning_rate = 0.001 #学习率
num_epochs = 100 #epoch的数量

# 定义数据预处理方式
# 普通的数据预处理方式
#在PyTorch库中，torchvision.transforms.Compose 类是用来组合多个图像变换操作的容器。
#transforms.Compose 将单个变换 transforms.ToTensor() 组合在一起。
#transforms.ToTensor() 是一个常用的预处理步骤，
#它将 PIL Image 或 numpy.ndarray 类型的数据转换为 PyTorch Tensor 格式，
#同时会将数据归一化到 [0, 1] 区间内（对于 PIL 图像，默认是按照 [0, 255] 范围内的像素值来进行归一化的）
#这样做的目的是为了满足深度学习模型对输入数据格式的要求。
#当你想要对图像数据应用一系列预处理操作时，可以将这些操作放入一个 Compose 对象中，
#从而在每次调用 transform(image) 时一次性完成所有预处理步骤。
#例如，如果还需要对图像进行其他的预处理操作，比如裁剪、归一化、翻转等，可以这样做：
'''
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像尺寸调整为224x224
    transforms.ToTensor(),           # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 数据归一化
])
'''
transform = transforms.Compose([
    # 随机裁剪为 32x32，填充为 4
    transforms.RandomCrop(32,padding=4),
    #随机水平翻转，发生的概率为0.5
    transforms.RandomHorizontalFlip(p=0.5),
    #随机垂直翻转，发生的概率为0.5
    #transforms.RandomVerticalFlip(p=0.5),
    # 随机旋转，随机角度为 15
    transforms.RandomRotation(15),
    # 转换为 Tensor
    transforms.ToTensor(),
    # 归一化，其中mean和std的数值是根据 CIFAR-10 数据集计算得到的，预先训练好的模型需要这些数值。
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
])

# 数据增强的数据预处理方式
# transform = transforms.Compose(


# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(nn.Module):
    '''
    定义卷积神经网络,3个卷积层,2个全连接层
    '''
    def __init__(self):
        super(Net, self).__init__()
        #nn.Conv2d 是一个二维卷积层的类，用于处理图像数据，通常应用于计算机视觉任务。
        #第一个参数 3 表示输入通道数（input channels）或输入特征图的数量，这里表示输入图像有 3 个颜色通道（红绿蓝 RGB）。
        #第二个参数 6 表示输出通道数（output channels）或输出特征图的数量，这里表示输出图像有 6 个颜色通道。
        #第三个参数 5 表示卷积核的大小（kernel size），这里表示卷积核大小为 3x3。
        #第四个参数 stride 表示卷积步长（stride），这里表示卷积步长为 1。
        #第五个参数 padding 表示填充（padding），这里表示在输入张量周围添加 0 填充。
        #第六个参数 dilation 表示卷积核之间的间距（dilation），这里表示卷积核之间的间距为 1。
        #第七个参数 groups 表示分组卷积的数量，这里表示不进行分组卷积。
        #在卷积操作中，输入张量的大小（height x width）需要满足卷积核大小和步长的要求，否则会报错。
        #输入的训练图像是32x32
        self.conv1 = nn.Sequential(
        # 输入通道数in_channels=3，输出通道数out_channels=64
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        # 对输入batch的每一个特征通道进行normalize,64表示输出的channel数量
        nn.BatchNorm2d(64), 
        nn.ReLU()
        )
        #self.pool 是一个二维最大池化层（Max Pooling Layer），同样定义在 PyTorch 的 nn 模块中。
        #nn.MaxPool2d(2, 2) 创建了一个具有固定大小池化核的二维最大池化层。
        #第一个参数 2 表示池化核的大小（kernel size），这里表示池化核大小为 2x2。
        #第二个参数 2 表示池化步长（stride），这里表示池化步长为 2。
        #在池化操作中，输入张量的大小（height x width）需要满足池化核大小和步长的要求，否则会报错。
        #这样做的主要目的是降低数据的空间维度（如高度和宽度），从而减少计算量，同时保留最重要的特征信息。
        #在实例化 nn.MaxPool2d(2, 2) 后，self.pool 将会在前向传播过程中对输入的三维张量（batch_size x 6 x new_height x new_width）应用 2x2 的最大池化操作，并生成新的三维张量（batch_size x 6 x new_height / 2 x new_width / 2）。
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(64, 16, 5)
        self.conv2=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv7=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        #nn.Linear 是一个全连接层，用于处理二维数据，通常应用于分类任务。
        self.fc1 = nn.Linear(512*2*2, 240)
        self.fc2=nn.Linear(240,10)
    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        #32x32
        x=self.conv1(x)
        x=self.pool(x)
        #16x16
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.pool(x)
        #8x8
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.pool(x)
        #4x4
        x=self.conv6(x)
        x=self.conv7(x)
        x=self.pool(x)
        #2x2
        x = x.view(-1, 512*2*2)
        x = F.tanh(self.fc1(x))
        x=F.softmax(self.fc2(x),dim=1)
        return x

# 实例化模型
model = Net()

#调增硬件加速
#mlu是Mobile Large Unified Memory，即寒武纪MLU处理器
#如果安装了寒武纪MLU处理器，可以使用torch.mlu.is_available()来判断是否可以使用MLU加速。
use_mlu = False
try:
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
model = model.to(device)

# 定义损失函数和优化器
# 使用交叉熵损失函数，后面可以更换
criterion = nn.CrossEntropyLoss()
#使用Adam优化器，后面可以更换
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 打印训练信息
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))

    # 测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))