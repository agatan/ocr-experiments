import json
import glob

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def _charset():
    alphas = ([chr(x) for x in range(ord('a'), ord('z')+1)])
    digits =  list('0123456789')
    hiras = [chr(x) for x in range(ord('あ'), ord('ゔ')+1)]
    return alphas + digits + hiras

CHARSET = _charset()
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}

def char2idx(c):
    return CHAR2IDX[c]

def text2idx(text: str):
    return np.array([char2idx(c) for c in text])

def idx2char(i):
    if i == 0:
        return '<UNK>'
    return CHARSET[i - 1]

def random_fontname():
    fonts = list(glob.glob("/Users/naomichi/Library/Fonts/*.ttf"))
    return np.random.choice(fonts)


def drawn_bb(fontname, fontsize, text, total_width, total_height):
    dummy = Image.new('RGB', (total_width, total_height))
    draw = ImageDraw.Draw(dummy)
    font = ImageFont.truetype(fontname, fontsize, encoding='unic')
    w, h = draw.textsize(text, font)
    del draw
    del dummy
    return w, h, font

def make_image(width, height, bgcolor):
    image = Image.new('RGB', (width, height), bgcolor)
    draw = ImageDraw.Draw(image)

    used = np.zeros((width, height), dtype='int32')
    charset = _charset()

    required_boxes = np.random.randint(3, 10)
    boxes = []
    while len(boxes) < required_boxes:
        text_len = np.random.randint(3, 12)
        text = ''.join((np.random.choice(charset) for _ in range(text_len)))

        top, left = np.random.randint(0, height), np.random.randint(0, width)
        w, h, f = drawn_bb(random_fontname(), np.random.randint(8, 24), text, width, height)
        if top + h > height or left + w > width:
            continue
        if used[left:left+w, top:top+h].sum() > 0:
            continue
        used[left:left+w, top:top+h] = 1
        draw.text((left, top), text, font=f, fill=(np.random.randint(0, 30), np.random.randint(0, 30), np.random.randint(0, 30)))
        boxes.append(dict(
            text=text,
            top=top,
            left=left,
            width=w,
            height=h,
        ))

    del draw
    return image, boxes


def main():
    import argparse
    import os.path
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1000)
    parser.add_argument('-o', default='out')
    args = parser.parse_args()
    if not os.path.isdir(args.o):
        os.makedirs(args.o)
    for i in tqdm(range(0, args.n), total=args.n):
        image, boxes = make_image(300, 200, (255, 255, 255))
        with open(os.path.join(args.o, f'{i}.json'), 'w') as f:
            json.dump(dict(file=f'{i}.png', boxes=boxes), f)
        with open(os.path.join(args.o, f'{i}.png'), 'wb') as f:
            image.save(f)


if __name__ == '__main__':
    main()
