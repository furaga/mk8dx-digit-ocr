import cv2
import numpy as np

MIN_ROD_CONTRAST = 32


def is_digit_or_segment(roi):
    top, left, right, bottom, area = roi
    height, width = bottom + 1 - top, right + 1 - left
    h_per_w = height / width
    if area / (height * width) > .99 or area / (height * width) < .1:
        return False
    if area / (height * width) > .85 and .5 <= h_per_w <= 2:
        return False
    if h_per_w < .3 or h_per_w > 10:
        return False
    return True


def rod_thresholding(roi, image_height, image_width, min_area_ratio=.0025, max_area_ratio=.2,
                     min_height_per_width=.5, max_height_per_width=10,
                     min_height_ratio=.15):
    top, left, right, bottom, area = roi
    image_area = image_height * image_width
    if (bottom + 1 - top) / image_height <= min_height_ratio:
        if area / image_area <= min_area_ratio:
            return False
        if area / image_area >= max_area_ratio:
            return False
    h_per_w = (bottom + 1 - top) / (right + 1 - left)
    if h_per_w <= min_height_per_width:
        return False
    if h_per_w >= max_height_per_width:
        return False
    if h_per_w <= 4 and area / ((bottom + 1 - top) * (right + 1 - left)) > .85:
        return False
    return True


SEGMENTS2DIGIT = {0b0000000: -1, 0b0000001: -1, 0b0000010: 1, 0b0000011: -1, 0b0000100: -1,
                  0b0000101: -1, 0b0000110: -1, 0b0000111: -1, 0b0001000: -1, 0b0001001: -1,
                  0b0001010: -1, 0b0001011: -1, 0b0001100: -1, 0b0001101: -1, 0b0001110: -1,
                  0b0001111: -1, 0b0010000: 1, 0b0010001: -1, 0b0010010: 1, 0b0010011: -1,
                  0b0010100: -1, 0b0010101: -1, 0b0010110: -1, 0b0010111: -1, 0b0011000: -1,
                  0b0011001: -1, 0b0011010: 4, 0b0011011: 3, 0b0011100: -1, 0b0011101: 2,
                  0b0011110: -1, 0b0011111: -1, 0b0100000: -1, 0b0100001: -1, 0b0100010: -1,
                  0b0100011: -1, 0b0100100: -1, 0b0100101: -1, 0b0100110: -1, 0b0100111: -1,
                  0b0101000: -1, 0b0101001: -1, 0b0101010: 4, 0b0101011: 5, 0b0101100: -1,
                  0b0101101: -1, 0b0101110: -1, 0b0101111: 6, 0b0110000: -1, 0b0110001: -1,
                  0b0110010: 4, 0b0110011: -1, 0b0110100: -1, 0b0110101: -1, 0b0110110: -1,
                  0b0110111: 0, 0b0111000: 4, 0b0111001: -1, 0b0111010: 4, 0b0111011: 9,
                  0b0111100: -1, 0b0111101: -1, 0b0111110: -1, 0b0111111: 8, 0b1000000: -1,
                  0b1000001: -1, 0b1000010: -1, 0b1000011: -1, 0b1000100: -1, 0b1000101: -1,
                  0b1000110: -1, 0b1000111: -1, 0b1001000: -1, 0b1001001: -1, 0b1001010: -1,
                  0b1001011: 3, 0b1001100: -1, 0b1001101: -1, 0b1001110: -1, 0b1001111: 6,
                  0b1010000: -1, 0b1010001: -1, 0b1010010: 7, 0b1010011: 3, 0b1010100: -1,
                  0b1010101: 2, 0b1010110: -1, 0b1010111: 0, 0b1011000: -1, 0b1011001: 2,
                  0b1011010: 3, 0b1011011: 3, 0b1011100: 2, 0b1011101: 2, 0b1011110: -1,
                  0b1011111: 8, 0b1100000: -1, 0b1100001: -1, 0b1100010: 7, 0b1100011: 5,
                  0b1100100: -1, 0b1100101: -1, 0b1100110: -1, 0b1100111: 0, 0b1101000: -1,
                  0b1101001: 5, 0b1101010: 5, 0b1101011: 5, 0b1101100: -1, 0b1101101: 6,
                  0b1101110: 6, 0b1101111: 6, 0b1110000: 7, 0b1110001: -1, 0b1110010: 7,
                  0b1110011: 0, 0b1110100: -1, 0b1110101: 0, 0b1110110: 0, 0b1110111: 0,
                  0b1111000: -1, 0b1111001: 9, 0b1111010: 9, 0b1111011: 9, 0b1111100: -1,
                  0b1111101: 8, 0b1111110: 8, 0b1111111: 8}


def ison(roi_bin, is_vertical: bool, ret_val: int) -> int:
    if is_vertical:
        thresh = roi_bin.shape[0] * .6
        if (np.count_nonzero(roi_bin, axis=0) > thresh).any():
            return ret_val
    else:
        thresh = roi_bin.shape[1] * .7
        if (np.count_nonzero(roi_bin, axis=1) > thresh).any():
            return ret_val
    return 0


def rod2digit(rod, rod_bin):
    height_rod, width_rod = rod_bin.shape
    _, labels, stats = cv2.connectedComponentsWithStats(rod_bin)[:3]
    # Remove tiny blobs.
    for i, stat in enumerate(stats):
        left, top, w, h = stat[:4]
        if max(w, h) < min(width_rod, height_rod) * .15:
            rod_bin[i == labels] = 0

    if height_rod / width_rod > 3:
        read_digit = 1
    else:
        vline1, vline2 = width_rod // 3, width_rod * 2 // 3
        hline1, hline2, hline3 = height_rod // 3, height_rod // 2, height_rod * 2 // 3
        segments = [rod_bin[:hline1, vline1:vline2], rod_bin[:hline2, :vline1], rod_bin[:hline2, vline2:],
                    rod_bin[hline1:hline3, vline1:vline2], rod_bin[hline2:,
                                                                   :vline1], rod_bin[hline2:, vline2:],
                    rod_bin[hline3:, vline1:vline2]]
        is_verticals = [False, True, True, False, True, True, False]
        ret_vals = [64, 32, 16, 8, 4, 2, 1]
        read_digit = SEGMENTS2DIGIT[sum(
            list(map(ison, segments, is_verticals, ret_vals)))]
    if -1 == read_digit and np.max(np.count_nonzero(rod_bin, axis=0)) / max(1, np.max(np.count_nonzero(rod_bin, axis=1))) > 3.5:
        read_digit = 1

    return read_digit


def resize_show_img(img, target_width=256):
    ratio = target_width / img.shape[1]
    return cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)


def detect_digit(roi_gray, verbose=False):
    roi_gray = roi_gray.copy()

    if verbose:
        cv2.imshow("input", resize_show_img(roi_gray))

    height, width = roi_gray.shape

    roi_bin = cv2.adaptiveThreshold(roi_gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                    max(int(min(height, width) / 4), 1) * 2 + 1, 2)

    if verbose:
        cv2.imshow("adaptiveThreshold", resize_show_img(roi_bin))

    _, labels, stats = cv2.connectedComponentsWithStats(roi_bin)[:3]

    # 数字は外枠に接していない想定
    comp_means = []
    for i, stat in enumerate(stats):
        mask = (255 * (labels == i)).astype(np.uint8)
        comp_means.append(cv2.mean(roi_gray, mask=mask)[0])

    comp_mean_threshold = np.max(comp_means) - 30
    thr_mask = np.zeros_like(roi_gray)
    thr_mask[roi_gray >= comp_mean_threshold] = 255
    kernel = np.ones((5, 5), np.uint8)
    thr_mask = cv2.morphologyEx(thr_mask, cv2.MORPH_CLOSE, kernel)

    if verbose:
        cv2.imshow("char mask", resize_show_img(thr_mask))

    roi_bin[thr_mask <= 0] = 0

    if verbose:
        cv2.imshow("char masked", resize_show_img(roi_bin))

    _, labels, stats = cv2.connectedComponentsWithStats(roi_bin)[:3]

    for i, stat in enumerate(stats):
        area = stat[-1]
        if area < width * height * 0.05 and max(stat[2], stat[3]) < min(width, height) * 0.2:
            roi_bin[i == labels] = 0
            continue
        if 0 == min(stat[0], stat[1]) or width == stat[0] + stat[2] or height == stat[1] + stat[3]:
            roi_bin[i == labels] = 0
            continue

    if verbose:
        cv2.imshow("filter components", resize_show_img(roi_bin))

    roi_bin_big = resize_show_img(roi_bin)
    kernel_size = int(5 * 256 / roi_bin.shape[1])
    kernel_size = max(5, 2 * (kernel_size // 2) + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    roi_bin_big = cv2.morphologyEx(roi_bin_big, cv2.MORPH_CLOSE, kernel)
    roi_bin_big = cv2.resize(
        roi_bin_big, (roi_bin.shape[1], roi_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

    _, labels, stats = cv2.connectedComponentsWithStats(roi_bin_big)[:3]

    if verbose:
        cv2.imshow(f"connect segments", resize_show_img(roi_bin_big))

    rods = []  # ROD = Region of a Digit
    for i, stat in enumerate(stats):
        left, top, w, h = stat[:4]
        if w < width and h < height and max(w, h) > min(width, height) / 4:
            rods.append((left, top, left + w, top + h, None))

    rods = sorted(rods, key=lambda r: r[0])
    if len(rods) <= 0:
        if verbose:
            cv2.waitKey(0)
        return False, 0

    num = 0
    if verbose:
        print("rods", rods)
    for i, rod in enumerate(rods):
        left, top, right, bottom, _ = rod
        cropped = roi_bin_big[top:bottom, left:right]
        digit = rod2digit(rod, cropped)
        if verbose:
            print("digit =", digit)
            cv2.imshow(f"char {i}", resize_show_img(cropped))
        num = 10 * num + digit

    if verbose:
        #        print("result =", num)
        cv2.waitKey(0)

    return True, num


if __name__ == "__main__":
    from pathlib import Path
    all_img_paths = Path("data/digital_testdata").glob("*.png")

    gt = {
        "001.png": 20780,
        "002.png": 9526,
    }

    n_NG = 0

    verbose = False
    for img_path in all_img_paths:
        if img_path.name not in gt:
            continue
#        if img_path.name != "008_6.png":
        # if img_path.name != "003_8.png":
        #     continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        ret, value = detect_digit(255 - img, verbose)
        print(ret, value)
        OK = "OK" if gt[img_path.name] == value else "NG"
        if OK != "OK":
            n_NG += 1
        print(f"{OK} {img_path.name}: {value}")

    print("# of NG =", n_NG)
