import sys
import os
from torch.utils.data import dataloader
import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

torch.cuda.set_device(7)
batch_size = 1

class plate_cnn_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.PReLU(),
            nn.Conv2d(64,128,3,2,1),
            nn.PReLU(),
            nn.Conv2d(128,128,3,2,1),
            nn.PReLU(),
            nn.Conv2d(128,64,3,2,1),
            nn.PReLU(),
            nn.Conv2d(64,16,3,1,1),
            nn.PReLU(),
            nn.Conv2d(16,4,3,1,1),
            nn.PReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(340, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 32),
            nn.PReLU(),
            nn.Linear(32, 8),
            nn.PReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        y = self.conv(x).reshape(batch_size, -1,)
        # print(y.shape)
        return self.fc(y)

class PlatePic(data.Dataset):

    def list_all_files(self, root):
        files = []
        list = os.listdir(root)
        for i in range(len(list)):
            element = os.path.join(root, list[i])
            if os.path.isdir(element):
                files.extend(self.list_all_files(element))
            elif os.path.isfile(element):
                files.append(element)
        return files

    def __init__(self, root):
        super().__init__()
        if not os.path.exists(root):
            raise ValueError('没有找到文件夹')
        self.files = self.list_all_files(root)

        self.X = []
        self.y = []
        self.labels = [os.path.split(os.path.dirname(file))[-1] for file in self.files]

        for i, file in enumerate(self.files):
            src_img = cv2.imread(file)
            if src_img.ndim != 3:
                continue
            resize_img = cv2.resize(src_img, (136, 36))
            self.X.append(resize_img)
            self.y.append([0 if self.labels[i] == 'no' else 1])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __getitem__(self, index):
        tf = transforms.ToTensor()
        # print(torch.Tensor(self.y[index]).shape)
        return tf(self.X[index]), torch.FloatTensor(self.y[index])

    def __len__(self) -> int:
        return len(self.X)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean=0.0, std=0.1)
        m.bias.data.fill_(0)

def train(epoch, lr):
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    loss_history = []

    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output, target)
        loss.backward()

        if loss_history and loss_history[-1] < loss.data:
            lr *= 0.7
        loss_history.append(loss.data)

        optimizer.step()

        if batch_idx % 2000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def get_accuracy(model, train_model_path):
    tot = len(train_loader.dataset)
    right = 0

    with torch.no_grad():
        for (input, target) in train_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)

            for idx in range(len(output)):
                if (output[idx] > 0.5 and target[idx] > 0.5) or \
                (output[idx] < 0.5 and target[idx] < 0.5):
                    right += 1

        acc = right / tot
        print('accuracy : %.3f' % acc)
        
        global best_acc
        if acc > best_acc:
            best_acc = acc
            torch.save(model, train_model_path)

if __name__ == '__main__':
    data_dir = '../images/cnn_plate_train'
    train_model_path = 'plate.pth'

    # model = plate_cnn_net()
    model = torch.load(train_model_path)
    model = model.cuda()
    # model.apply(weights_init)

    print("Generate Model.")

    batch_size = 1
    dataset = PlatePic(data_dir)
    train_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size,
                    num_workers=14, pin_memory=True, drop_last=True)

    global best_acc
    best_acc = 0.0
    for epoch in range(0, 30):
        lr = 0.001
        train(epoch, lr)
        get_accuracy(model, train_model_path)
    
    torch.save(model, train_model_path)

    print("Finish Training")
