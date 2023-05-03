import sys
import cv2

sys.path.append(".")
import mk8dx_digit_ocr


def test_detect_digit_gray():
    img1 = 255 - cv2.imread("test/data/001.png", cv2.IMREAD_GRAYSCALE)
    ret, num = mk8dx_digit_ocr.detect_digit(img1)
    assert ret and num == 20780, f"ret={ret}, num={num}"

    img2 = cv2.imread("test/data/002.png", cv2.IMREAD_GRAYSCALE)
    ret, num = mk8dx_digit_ocr.detect_digit(img2)
    assert ret and num == 9526, f"ret={ret}, num={num}"


def test_detect_digit_bgr():
    img1 = 255 - cv2.imread("test/data/001.png", cv2.IMREAD_COLOR)
    ret, num = mk8dx_digit_ocr.detect_digit(img1)
    assert ret and num == 20780, f"ret={ret}, num={num}"

    img2 = 255 - cv2.imread("test/data/002.png", cv2.IMREAD_COLOR)
    ret, num = mk8dx_digit_ocr.detect_digit(img2)
    assert ret and num == 9526, f"ret={ret}, num={num}"


def test_detect_digit_bgra():
    img1 = 255 - cv2.imread("test/data/001.png", cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    ret, num = mk8dx_digit_ocr.detect_digit(img1)
    assert ret and num == 20780, f"ret={ret}, num={num}"

    img2 = 255 - cv2.imread("test/data/002.png", cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
    ret, num = mk8dx_digit_ocr.detect_digit(img2)
    assert ret and num == 9526, f"ret={ret}, num={num}"
