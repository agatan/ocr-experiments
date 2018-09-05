"""Evaluate ocr.
"""
import json
import logging


def main():
    logger = logging.getLogger(__name__)
    logger.info("dump dummy metrics")
    with open("metrics.json", "w") as fp:
        m = dict(precision=0.98, recall=0.87)
        json.dump(m, fp)
    with open("out.txt", "w") as fp:
        for i in range(1, 10):
            fp.write(f'{i},{"positive" if i % 2 == 0 else "negative"}\n')


if __name__ == "__main__":
    main()
