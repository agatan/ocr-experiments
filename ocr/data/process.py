"""Process raw dataset.
"""
import argparse
import csv
import logging
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
from faker import Faker


_faker = Faker("ja_JP")


def _random_from_charset():
    n = np.random.randint(1, 3)
    return "".join(np.random.choice(CHARSET, n))


def _random_text():
    choices = [
        _random_from_charset,
        _random_from_charset,
        _random_from_charset,
        _faker.address,
        _faker.first_name,
        _faker.last_name,
        _random_from_charset,
        _faker.address,
        _faker.first_name,
        _faker.last_name,
        _random_from_charset,
        _faker.address,
        _faker.first_name,
        _faker.last_name,
        _faker.first_romanized_name,
        _faker.last_romanized_name,
        _faker.phone_number,
        _faker.company,
        _faker.email,
        _faker.url,
    ]
    fn = np.random.choice(choices)
    return fn()


def _charset():
    # alphas = ([chr(x) for x in range(ord('a'), ord('z')+1)])
    # digits = list("あいうえお")
    with open(
        os.path.join(os.path.dirname(__file__), "..", "..", "charlist.txt"), "r"
    ) as f:
        chars = list(f.read().strip())
    chars.append(" ")
    # hiras = [chr(x) for x in range(ord('あ'), ord('ゔ')+1)]
    # return alphas + digits + hiras
    return chars


CHARSET = _charset()
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}


def vocab():
    return len(CHARSET) + 2


def char2idx(c):
    x = CHAR2IDX.get(c, None)
    if x is None:
        x = 0
    return x


def text2idx(text: str):
    return np.array([char2idx(c) for c in text])


def idx2char(i):
    if i == 0:
        return "<UNK>"
    return CHARSET[i - 1]


def random_fontname():
    fonts = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W0.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W1.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W2.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W9.ttc",
        "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",
        "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
    ]
    return np.random.choice(fonts)


def drawn_bb(fontname, fontsize, text, total_width, total_height):
    dummy = Image.new("RGB", (total_width, total_height))
    draw = ImageDraw.Draw(dummy)
    font = ImageFont.truetype(fontname, fontsize, encoding="unic")
    w, h = draw.textsize(text, font)
    del draw
    del dummy
    return w, h, font


def make_image(width, height, bgcolor):
    image = Image.new("RGB", (width, height), bgcolor)
    draw = ImageDraw.Draw(image)

    used = np.zeros((width, height), dtype="int32")

    required_boxes = np.random.randint(3, 10)
    boxes = []
    while len(boxes) < required_boxes:
        text = _random_text()
        if not all((ord(c) < 128 for c in text)) and np.random.rand() < 0.5:
            text = "\n".join(text)

        top, left = np.random.randint(0, height), np.random.randint(0, width)
        w, h, f = drawn_bb(
            random_fontname(), np.random.randint(16, 32), text, width, height
        )
        if top + h > height or left + w > width:
            continue
        if used[left : left + w, top : top + h].sum() > 0:
            continue
        used[left : left + w, top : top + h] = 1
        draw.text(
            (left, top),
            text,
            font=f,
            fill=(
                np.random.randint(0, 30),
                np.random.randint(0, 30),
                np.random.randint(0, 30),
            ),
        )
        boxes.append(
            dict(text=text, ymin=top, xmin=left, xmax=left + w, ymax=top + h, angle=0)
        )

    del draw
    return image, boxes


def _dofn(n: int, n_images: int, out: str):
    np.random.seed(int(time.time() * n) % (2 ** 32))
    with open(os.path.join(out, f"annotations-{n}.csv"), "w") as f:
        w = csv.DictWriter(
            f, fieldnames=["image", "xmin", "ymin", "xmax", "ymax", "angle", "text"]
        )
        for i in range(n_images):
            image, coordinates = make_image(832, 512, (255, 255, 255))
            image_file = os.path.join(out, "images", f"{n}-{i}.png")
            image.save(image_file)
            for coor in coordinates:
                coor["image"] = os.path.join("images", f"{n}-{i}.png")
            w.writerows(coordinates)


def main():
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--n_batches", default=4, type=int)
    args = parser.parse_args()

    logger.info("process raw dataset, and save them in data/processed/")

    Path("data/processed/images").mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor() as pool:
        tasks = [
            pool.submit(_dofn, i, args.batch_size, "data/processed")
            for i in range(args.n_batches)
        ]
        for t in tqdm(tasks, total=args.n_batches):
            t.result()

    csvfiles = [
        os.path.join("data/processed", f)
        for f in os.listdir("data/processed")
        if f.endswith(".csv")
    ]
    with open("data/processed/annotations.csv", "w") as fp:
        fp.write("image,xmin,ymin,xmax,ymax,angle,text\n")
        for c in csvfiles:
            with open(c, "r") as cf:
                fp.write(cf.read())
            os.remove(c)


if __name__ == "__main__":
    main()
