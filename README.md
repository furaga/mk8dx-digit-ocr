# mk8dx-digit-ocr

Python3 library for 7 segment digit OCR for Mario Kart 8 Deluxe.

Licensed under the Apache License.

## Project pages

GitHub project https://github.com/furaga/mk8dx-digit-ocr

## Installation

Just run `pip3 install git+https://github.com/furaga/mk8dx-digit-ocr.git` in your Python venv or directly on your system.

For manual install, git clone the github repository and copy the directory mk8dx_digt_ocr in your python project root.

Requires: opencv-python (from pip)

## Usage

See python scripts in the [test](https://github.com/furaga/mk8dx-digit-ocr/tree/master/test) directory.

The following is a simple example.

```
$ python
>>> import mk8dx_digit_ocr
>>> import cv2
>>> img = cv2.imread("test/data/14660.png")
>>> mk8dx_digit_ocr.detect_digit(img)
(True, 14660)
```

<img src="https://raw.githubusercontent.com/furaga/mk8dx-digit-ocr/master/doc/14660.png">

Note that `img` must be a cropped image that contains a single line of number. 

## Problems?

Please check on Github project issues, and if nobody else have experienced it before, you can file a new issue.