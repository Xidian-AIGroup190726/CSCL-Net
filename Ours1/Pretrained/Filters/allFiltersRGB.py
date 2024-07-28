# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from allFilters import gaussian_low_pass_filter
from allFilters import gaussian_high_pass_filter
from allFilters import circle_low_pass_filter
from allFilters import circle_high_pass_filter
from allFilters import square_high_pass_filter
from allFilters import square_low_pass_filter


def gaussian_low_pass_filter_RGB(f_shift, D0):

    low_parts_gaussian = np.zeros_like(f_shift, dtype=complex)
    for i in range(3):
        low_parts_gaussian[:, :, i] = gaussian_low_pass_filter(f_shift[:, :, i], D0)

    return low_parts_gaussian


def gaussian_high_pass_filter_RGB(f_shift, D0):

    high_parts_gaussian = np.zeros_like(f_shift, dtype=complex)
    for i in range(3):
        high_parts_gaussian[:, :, i] = gaussian_high_pass_filter(f_shift[:, :, i], D0)

    return high_parts_gaussian


def circle_low_pass_filter_RGB(f_shift, radius_ratio):

    low_parts_circle = np.zeros_like(f_shift, dtype=complex)
    for i in range(3):
        low_parts_circle[:, :, i] = circle_low_pass_filter(f_shift[:, :, i], radius_ratio)

    return low_parts_circle


def circle_high_pass_filter_RGB(f_shift, radius_ratio):

    high_parts_circle = np.zeros_like(f_shift, dtype=complex)
    for i in range(3):
        high_parts_circle[:, :, i] = circle_high_pass_filter(f_shift[:, :, i], radius_ratio)

    return high_parts_circle


def square_high_pass_filter_RGB(f_shift, length):

    high_parts_square = np.zeros_like(f_shift, dtype=complex)
    for i in range(3):
        high_parts_square[:, :, i] = square_high_pass_filter(f_shift[:, :, i], length)

    return high_parts_square


def square_low_pass_filter_RGB(f_shift, length):

    low_parts_square = np.zeros_like(f_shift, dtype=complex)
    for i in range(3):
        low_parts_square[:, :, i] = square_low_pass_filter(f_shift[:, :, i], length)

    return low_parts_square


def i_fft(f_shift):

    i_shift = np.fft.ifftshift(f_shift)  # 把低频部分shift回左上角
    i_img = np.fft.ifftn(i_shift)  # 出来的是复数，无法显示
    i_img = np.abs(i_img)  # 返回复数的模
    return i_img


def get_low_high_f_by_gaussian_filter(img, D0):

    f = np.zeros_like(img, dtype=complex)
    f_shift = np.zeros_like(img, dtype=complex)
    low_parts_img = np.zeros_like(img, dtype=float)
    high_parts_img = np.zeros_like(img, dtype=float)
    img_new_low = np.zeros_like(img)
    img_new_high = np.zeros_like(img)

    for i in range(3):
        f[:, :, i] = np.fft.fftn(img[:, :, i])
        f_shift[:, :, i] = np.fft.fftshift(f[:, :, i])

    low_parts_circle = gaussian_low_pass_filter_RGB(f_shift, D0)
    high_parts_circle = gaussian_high_pass_filter_RGB(f_shift, D0)

    for i in range(3):
        low_parts_img[:, :, i] = i_fft(low_parts_circle[:, :, i])
        high_parts_img[:, :, i] = i_fft(high_parts_circle[:, :, i])

    for i in range(3):
        low_parts_img[:, :, i] = (low_parts_img[:, :, i] - np.amin(low_parts_img[:, :, i])) / (
                    np.amax(low_parts_img[:, :, i]) - np.amin(low_parts_img[:, :, i]) + 0.00001)
        high_parts_img[:, :, i] = (high_parts_img[:, :, i] - np.amin(high_parts_img[:, :, i]) + 0.00001) / (
                    np.amax(high_parts_img[:, :, i]) - np.amin(high_parts_img[:, :, i]) + 0.00001)

    for i in range(3):
        img_new_low = np.array(low_parts_img * 255, np.uint8)
        img_new_high = np.array(high_parts_img * 255, np.uint8)
    return img_new_low, img_new_high


def get_low_high_f_by_circle_filter(img, radius_ratio):

    f = np.zeros_like(img, dtype=complex)
    f_shift = np.zeros_like(img, dtype=complex)
    low_parts_img = np.zeros_like(img, dtype=float)
    high_parts_img = np.zeros_like(img, dtype=float)
    img_new_low = np.zeros_like(img)
    img_new_high = np.zeros_like(img)

    for i in range(3):
        f[:, :, i] = np.fft.fftn(img[:, :, i])
        f_shift[:, :, i] = np.fft.fftshift(f[:, :, i])

    low_parts_circle = circle_low_pass_filter_RGB(f_shift, radius_ratio=radius_ratio)
    high_parts_circle = circle_high_pass_filter_RGB(f_shift, radius_ratio=radius_ratio)

    for i in range(3):
        low_parts_img[:, :, i] = i_fft(low_parts_circle[:, :, i])
        high_parts_img[:, :, i] = i_fft(high_parts_circle[:, :, i])

    for i in range(3):
        low_parts_img[:, :, i] = (low_parts_img[:, :, i] - np.amin(low_parts_img[:, :, i])) / (
                np.amax(low_parts_img[:, :, i]) - np.amin(low_parts_img[:, :, i]) + 0.00001)
        high_parts_img[:, :, i] = (high_parts_img[:, :, i] - np.amin(high_parts_img[:, :, i]) + 0.00001) / (
                np.amax(high_parts_img[:, :, i]) - np.amin(high_parts_img[:, :, i]) + 0.00001)

    for i in range(3):
        img_new_low[:, :, i] = np.array(low_parts_img[:, :, i] * 255, np.uint8)
        img_new_high[:, :, i] = np.array(high_parts_img[:, :, i] * 255, np.uint8)
    return img_new_low, img_new_high


def get_low_high_f_by_square_filter(img, length):

    f = np.zeros_like(img, dtype=complex)
    f_shift = np.zeros_like(img, dtype=complex)
    low_parts_img = np.zeros_like(img, dtype=float)
    high_parts_img = np.zeros_like(img, dtype=float)
    img_new_low = np.zeros_like(img)
    img_new_high = np.zeros_like(img)

    for i in range(3):
        f[:, :, i] = np.fft.fftn(img[:, :, i])
        f_shift[:, :, i] = np.fft.fftshift(f[:, :, i])

    low_parts_circle = square_low_pass_filter_RGB(f_shift, length)
    high_parts_circle = square_high_pass_filter_RGB(f_shift, length)

    for i in range(3):
        low_parts_img[:, :, i] = i_fft(low_parts_circle[:, :, i])
        high_parts_img[:, :, i] = i_fft(high_parts_circle[:, :, i])

    for i in range(3):
        low_parts_img[:, :, i] = (low_parts_img[:, :, i] - np.amin(low_parts_img[:, :, i])) / (
                    np.amax(low_parts_img[:, :, i]) - np.amin(low_parts_img[:, :, i]) + 0.00001)
        high_parts_img[:, :, i] = (high_parts_img[:, :, i] - np.amin(high_parts_img[:, :, i]) + 0.00001) / (
                    np.amax(high_parts_img[:, :, i]) - np.amin(high_parts_img[:, :, i]) + 0.00001)

    for i in range(3):
        img_new_low[:, :, i] = np.array(low_parts_img[:, :, i] * 255, np.uint8)
        img_new_high[:, :, i] = np.array(high_parts_img[:, :, i] * 255, np.uint8)
    return img_new_low, img_new_high


if __name__ == '__main__':
    radius_ratio = 0.5
    length = 50
    D0 = 50

    img = cv2.imread('C:/Users/u/Pictures/scene.jpg', cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # b, g, r = cv2.split(img)
    # img_rgb = cv2.merge([r, g, b])

    # print(img_rgb.shape)
    low_freq_parts_img, high_freq_parts_img = get_low_high_f_by_gaussian_filter(img_rgb, D0=D0)
    # low_freq_parts_img, high_freq_parts_img = get_low_high_f_by_circle_filter(img_rgb, radius_ratio=radius_ratio)
    # low_freq_parts_img, high_freq_parts_img = get_low_high_f_by_square_filter(img_rgb, length=length)
    # print(low_freq_parts_img)
    # print(high_freq_parts_img)

    plt.subplot(131), plt.imshow(img_rgb), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(low_freq_parts_img), plt.title('low_freq_img')
    plt.axis('off')
    plt.subplot(133), plt.imshow(high_freq_parts_img), plt.title('high_freq_img')
    plt.axis('off')
    plt.show()
