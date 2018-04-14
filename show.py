import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageDraw

from ssocr import SSOCR
from data import ListDataset


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
testset = ListDataset(root='test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

net = SSOCR()
p = 'net16.pth'
if os.path.exists(p):
    net.load_state_dict(torch.load(p))

n = np.random.randint(0, 100)
img = Image.open('test/{}.png'.format(n))
x = transform(img).unsqueeze(0)
x = Variable(x, volatile=True)

loc_preds, conf_preds = net(x)
boxes = testset.encoder.decode(loc_preds.data.squeeze(0), conf_preds.data.squeeze(0), [300, 200])
print(boxes)
if boxes is None:
    import sys
    sys.exit(1)
draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
