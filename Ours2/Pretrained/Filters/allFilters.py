# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def gaussian_low_pass_filter(f_shift, D0):

    h, w = f_shift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = np.exp(- dis_square / (2 * D0 ** 2))

    return template * f_shift


def gaussian_high_pass_filter(f_shift, D0):

    h, w = f_shift.shape
    x, y = np.mgrid[0:h, 0:w]
    center = (int((h - 1) / 2), int((w - 1) / 2))
    dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
    template = 1 - np.exp(- dis_square / (2 * D0 ** 2))

    return template * f_shift


def circle_low_pass_filter(f_shift, radius_ratio):

    rows, cols = f_shift.shape
    filter = np.zeros(f_shift.shape, np.uint8)
    crow, ccol = int((f_shift.shape[0] - 1) / 2), int((f_shift.shape[1] - 1) / 2)
    radius = int(radius_ratio * f_shift.shape[0] / 2)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt(np.power((i - crow), 2) + np.power((i - crow), 2)) <= radius:
                filter[i, j] = 1
            else:
                filter[i, j] = 0
    return filter * f_shift


def circle_high_pass_filter(f_shift, radius_ratio):
    rows, cols = f_shift.shape
    filter = np.ones(f_shift.shape, np.uint8)

    crow, ccol = int((f_shift.shape[0] - 1) / 2), int((f_shift.shape[1] - 1) / 2)
    radius = int(radius_ratio * f_shift.shape[0] / 2)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt(np.power((i - crow), 2) + np.power((i - crow), 2)) < radius:
                filter[i, j] = 0
            else:
                filter[i, j] = 1
    return filter * f_shift


def square_high_pass_filter(f_shift, length):

    rows, cols = f_shift.shape
    crow, ccol = int(f_shift.shape[0] / 2), int(f_shift.shape[1] / 2)
    mask = np.ones((rows, cols))
    mask[crow - length:crow + length, ccol - length:ccol + length] = 0
    return f_shift * mask


def square_low_pass_filter(f_shift, length):

    rows, cols = f_shift.shape
    crow, ccol = int(f_shift.shape[0] / 2), int(f_shift.shape[1] / 2)
    mask = np.zeros((rows, cols))
    mask[crow - length:crow + length, ccol - length:ccol + length] = 1

    return f_shift * mask


def i_fft(f_shift):

    i_shift = np.fft.ifftshift(f_shift)
    i_img = np.fft.ifftn(i_shift)
    i_img = np.abs(i_img)
    return i_img


def get_low_high_f_by_gaussian_filter(img, D0):

    f = np.fft.fftn(img)
    f_shift = np.fft.fftshift(f)

    low_parts_gaussian = gaussian_low_pass_filter(f_shift.copy(), D0=D0)
    high_parts_gaussian = gaussian_high_pass_filter(f_shift.copy(), D0=D0)
    low_parts_img = i_fft(low_parts_gaussian)
    high_parts_img = i_fft(high_parts_gaussian)

    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high


def get_low_high_f_by_circle_filter(img, radius_ratio):

    f = np.fft.fftn(img)
    f_shift = np.fft.fftshift(f)

    low_parts_circle = circle_low_pass_filter(f_shift.copy(), radius_ratio=radius_ratio)
    high_parts_circle = circle_high_pass_filter(f_shift.copy(), radius_ratio=radius_ratio)

    low_parts_img = i_fft(low_parts_circle)
    high_parts_img = i_fft(high_parts_circle)

    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
            np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high


def get_low_high_f_by_square_filter(img, length):

    f = np.fft.fftn(img)
    f_shift = np.fft.fftshift(f)

    low_parts_circle = square_low_pass_filter(f_shift.copy(), length=length)
    high_parts_circle = square_high_pass_filter(f_shift.copy(), length=length)
    low_parts_img = i_fft(low_parts_circle)
    high_parts_img = i_fft(high_parts_circle)

    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high


if __name__ == '__main__':
    radius_ratio = 0.5
    length = 50
    D0 = 50

    img = cv2.imread('C:/Users/u/Pictures/scene.jpg', cv2.IMREAD_GRAYSCALE)
    # low_freq_parts_img, high_freq_parts_img = get_low_high_f_by_gaussian_filter(img, D0=D0)
    low_freq_parts_img, high_freq_parts_img = get_low_high_f_by_circle_filter(img, radius_ratio=radius_ratio)
    # low_freq_parts_img, high_freq_parts_img = get_low_high_f_by_square_filter(img, length=length)

    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(low_freq_parts_img, 'gray'), plt.title('low_freq_img')
    plt.axis('off')
    plt.subplot(133), plt.imshow(high_freq_parts_img, 'gray'), plt.title('high_freq_img')
    plt.axis('off')
    plt.show()
