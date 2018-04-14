import os
import json

import torch
import torch.utils.data as data
from PIL import Image

from encoder import DataEncoder


class ListDataset(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.encoder = DataEncoder()
        self.input_size = [300, 200] # W, H

        self.fnames = []
        self.boxes = []

        i = 0
        while True:
            f = os.path.join(self.root, f'{i}.json')
            i += 1
            if not os.path.isfile(f):
                break
            with open(f, 'r') as fp:
                info = json.load(fp)
            self.fnames.append(info['file'])
            box = []
            for b in info['boxes']:
                xmin = float(b['left'])
                ymin = float(b['top'])
                xmax = xmin + float(b['width'])
                ymax = xmin + float(b['height'])
                box.append([xmin, ymin, xmax, ymax])
            self.boxes.append(torch.Tensor(box))

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        img = self.transform(img)
        return img, boxes

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]

        w, h = self.input_size
        n_imgs = len(imgs)
        inputs = torch.zeros(n_imgs, 3, h, w)
        loc_targets = []
        masks = []
        for i in range(n_imgs):
            inputs[i] = imgs[i]
            loc_target, mask = self.encoder.encode(boxes[i], self.input_size)
            loc_targets.append(loc_target)
            masks.append(mask)
        return inputs, torch.stack(loc_targets), torch.stack(masks)

    def __len__(self):
        return len(self.fnames)


def main():
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = ListDataset(root='out', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    for images, loc_targets, mask in dataloader:
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break
