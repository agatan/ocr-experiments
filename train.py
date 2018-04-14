import os

import torch
import torch.optim as optim
from torch.autograd import Variable

import torchvision.transforms as transforms

from loss import Loss
from ssocr import SSOCR
from data import ListDataset


CUDA = False


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
trainset = ListDataset(root='train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)
testset = ListDataset(root='test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

net = SSOCR()
if os.path.exists('net.pth'):
    net.load_state_dict(torch.load('net.pth'))

if CUDA:
    net.cuda()

criterion = Loss()
optimizer = optim.Adam(net.parameters())


def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, mask) in enumerate(trainloader):
        if CUDA:
            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
            mask = mask.cuda()
        inputs = Variable(inputs)
        loc_targets = Variable(loc_targets)
        mask = Variable(mask)

        optimizer.zero_grad()
        loc_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        print(f'train_loss: {loss.data[0]}, average: {train_loss / (batch_idx + 1)}')

def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, mask) in enumerate(testloader):
        if CUDA:
            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
            mask = mask.cuda()
        inputs = Variable(inputs)
        loc_targets = Variable(loc_targets)
        mask = Variable(mask)

        loc_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, mask)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

start_epoch = 0
for epoch in range(start_epoch, start_epoch+10):
    train(epoch)
    test(epoch)
    torch.save(net.state_dict(), 'net.pth')
