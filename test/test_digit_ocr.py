import sys
import cv2
from pathlib import Path

sys.path.append(".")
import mk8dx_digit_ocr


# Positive
def test_detect_digit_gray():
    for img_path in Path("test/data").glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        ret, num = mk8dx_digit_ocr.detect_digit(img)  # , verbose=True)
        gt = int(img_path.stem)
        assert ret and num == gt, f"ret={ret}, num={num}, img_path={str(img_path)}"


def test_detect_digit_bgr():
    for img_path in Path("test/data").glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        ret, num = mk8dx_digit_ocr.detect_digit(img)
        gt = int(img_path.stem)
        assert ret and num == gt, f"ret={ret}, num={num}, img_path={str(img_path)}"


# Negative
def test_detect_digit_bgra_negative():
    for img_path in Path("test/data/negative").glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        ret, num = mk8dx_digit_ocr.detect_digit(img)
        assert not ret, f"ret={ret}, num={num}, img_path={str(img_path)}"


def test_detect_digit_gray_negative():
    for img_path in Path("test/data/negative").glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        ret, num = mk8dx_digit_ocr.detect_digit(img)  # , verbose=True)
        assert not ret, f"ret={ret}, num={num}, img_path={str(img_path)}"


def test_detect_digit_bgr_negative():
    for img_path in Path("test/data/negative").glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        ret, num = mk8dx_digit_ocr.detect_digit(img)
        assert not ret, f"ret={ret}, num={num}, img_path={str(img_path)}"
