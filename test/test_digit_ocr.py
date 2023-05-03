import pytest
import cv2
import mk8dx_digit_ocr

def test_detect_digit():
    img1 = cv2.imread("data/001.png", cv2.IMREAD_GRAYSCALE)
    ret, num = mk8dx_digit_ocr.detect_digit(img1)
    assert ret and num, f"ret={ret}, num={num}"

    img2 = cv2.imread("data/002.png", cv2.IMREAD_GRAYSCALE)
    ret, num = mk8dx_digit_ocr.detect_digit(img2)
    assert ret and num, f"ret={ret}, num={num}"

