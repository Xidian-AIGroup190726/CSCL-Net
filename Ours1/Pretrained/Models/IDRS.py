# -*- coding: utf-8 -*-
# import cv2
import numpy as np
# from matplotlib import pyplot as plt

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
    # print('x is:{}'.format(x))
    # print('y is:{}'.format(y))
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


def gaussian_low_pass_filter_RGB(f_shift, D0, num):

    low_parts_gaussian = np.zeros_like(f_shift, dtype=complex)
    for i in range(num):
        low_parts_gaussian[i, :, :] = gaussian_low_pass_filter(f_shift[i, :, :], D0)
    return low_parts_gaussian


def gaussian_high_pass_filter_RGB(f_shift, D0, num):

    high_parts_gaussian = np.zeros_like(f_shift, dtype=complex)
    for i in range(num):
        high_parts_gaussian[i, :, :] = gaussian_high_pass_filter(f_shift[i, :, :], D0)
    return high_parts_gaussian


def circle_low_pass_filter_RGB(f_shift, radius_ratio, num):

    low_parts_circle = np.zeros_like(f_shift, dtype=complex)
    for i in range(num):
        low_parts_circle[i, :, :] = circle_low_pass_filter(f_shift[i, :, :], radius_ratio)
    return low_parts_circle


def circle_high_pass_filter_RGB(f_shift, radius_ratio, num):

    high_parts_circle = np.zeros_like(f_shift, dtype=complex)
    for i in range(num):
        high_parts_circle[i, :, :] = circle_high_pass_filter(f_shift[i, :, :], radius_ratio)
    return high_parts_circle


def square_high_pass_filter_RGB(f_shift, length, num):

    high_parts_square = np.zeros_like(f_shift, dtype=complex)
    for i in range(num):
        high_parts_square[i, :, :] = square_high_pass_filter(f_shift[i, :, :], length)
    return high_parts_square


def square_low_pass_filter_RGB(f_shift, length, num):

    low_parts_square = np.zeros_like(f_shift, dtype=complex)
    for i in range(num):
        low_parts_square[i, :, :] = square_low_pass_filter(f_shift[i, :, :], length)
    return low_parts_square


def i_fft(f_shift):

    i_shift = np.fft.ifftshift(f_shift)
    i_img = np.fft.ifftn(i_shift)
    i_img = np.abs(i_img)
    return i_img


def get_low_high_f_by_gaussian_filter(img, D0, num):

    # print(num)
    f = np.zeros_like(img, dtype=complex)
    f_shift = np.zeros_like(img, dtype=complex)
    low_parts_img = np.zeros_like(img, dtype=float)
    high_parts_img = np.zeros_like(img, dtype=float)
    img_new_low = np.zeros_like(img)
    img_new_high = np.zeros_like(img)
    for i in range(num):
        f[i, :, :] = np.fft.fftn(img[i, :, :])
        f_shift[i, :, :] = np.fft.fftshift(f[i, :, :])

    low_parts_circle = gaussian_low_pass_filter_RGB(f_shift, D0, num)
    high_parts_circle = gaussian_high_pass_filter_RGB(f_shift, D0, num)

    for i in range(num):
        low_parts_img[i, :, :] = i_fft(low_parts_circle[i, :, :])
        high_parts_img[i, :, :] = i_fft(high_parts_circle[i, :, :])

    for i in range(num):
        low_parts_img[i, :, :] = (low_parts_img[i, :, :] - np.amin(low_parts_img[i, :, :])) / (
                    np.amax(low_parts_img[i, :, :]) - np.amin(low_parts_img[i, :, :]) + 0.00001)
        high_parts_img[i, :, :] = (high_parts_img[i, :, :] - np.amin(high_parts_img[i, :, :]) + 0.00001) / (
                    np.amax(high_parts_img[i, :, :]) - np.amin(high_parts_img[i, :, :]) + 0.00001)

    for i in range(num):
        img_new_low[i, :, :] = np.array(low_parts_img[i, :, :] * 255, np.uint8)
        img_new_high[i, :, :] = np.array(high_parts_img[i, :, :] * 255, np.uint8)
    return img_new_low, img_new_high


def get_low_high_f_by_circle_filter(img, radius_ratio, num):
    f = np.zeros_like(img, dtype=complex)
    f_shift = np.zeros_like(img, dtype=complex)
    low_parts_img = np.zeros_like(img, dtype=float)
    high_parts_img = np.zeros_like(img, dtype=float)
    img_new_low = np.zeros_like(img)
    img_new_high = np.zeros_like(img)

    for i in range(num):
        f[i, :, :] = np.fft.fftn(img[i, :, :])
        f_shift[i, :, :] = np.fft.fftshift(f[i, :, :])

    low_parts_circle = circle_low_pass_filter_RGB(f_shift, radius_ratio, num)
    high_parts_circle = circle_high_pass_filter_RGB(f_shift, radius_ratio, num)

    for i in range(num):
        low_parts_img[i, :, :] = i_fft(low_parts_circle[i, :, :])
        high_parts_img[i, :, :] = i_fft(high_parts_circle[i, :, :])

    for i in range(num):
        low_parts_img[i, :, :] = (low_parts_img[i, :, :] - np.amin(low_parts_img[i, :, :])) / (
                np.amax(low_parts_img[i, :, :]) - np.amin(low_parts_img[i, :, :]) + 0.00001)
        high_parts_img[i, :, :] = (high_parts_img[i, :, :] - np.amin(high_parts_img[i, :, :]) + 0.00001) / (
                np.amax(high_parts_img[i, :, :]) - np.amin(high_parts_img[i, :, :]) + 0.00001)

    for i in range(num):
        img_new_low[i, :, :] = np.array(low_parts_img[i, :, :] * 255, np.uint8)
        img_new_high[i, :, :] = np.array(high_parts_img[i, :, :] * 255, np.uint8)
    return img_new_low, img_new_high


def get_low_high_f_by_square_filter(img, length, num):
    f = np.zeros_like(img, dtype=complex)
    f_shift = np.zeros_like(img, dtype=complex)
    low_parts_img = np.zeros_like(img, dtype=float)
    high_parts_img = np.zeros_like(img, dtype=float)
    img_new_low = np.zeros_like(img)
    img_new_high = np.zeros_like(img)

    for i in range(num):
        f[i, :, :] = np.fft.fftn(img[i, :, :])
        f_shift[i, :, :] = np.fft.fftshift(f[i, :, :])

    low_parts_circle = square_low_pass_filter_RGB(f_shift, length, num)
    high_parts_circle = square_high_pass_filter_RGB(f_shift, length, num)

    for i in range(num):
        low_parts_img[i, :, :] = i_fft(low_parts_circle[i, :, :])
        high_parts_img[i, :, :] = i_fft(high_parts_circle[i, :, :])

    for i in range(num):
        low_parts_img[i, :, :] = (low_parts_img[i, :, :] - np.amin(low_parts_img[i, :, :])) / (
                    np.amax(low_parts_img[i, :, :]) - np.amin(low_parts_img[i, :, :]) + 0.00001)
        high_parts_img[i, :, :] = (high_parts_img[i, :, :] - np.amin(high_parts_img[i, :, :]) + 0.00001) / (
                    np.amax(high_parts_img[i, :, :]) - np.amin(high_parts_img[i, :, :]) + 0.00001)

    for i in range(num):
        img_new_low[i, :, :] = np.array(low_parts_img[i, :, :] * 255, np.uint8)
        img_new_high[i, :, :] = np.array(high_parts_img[i, :, :] * 255, np.uint8)
    return img_new_low, img_new_high


class IDM:
    def __init__(self, name='gaussian', D0=50, radius_ratio=0.5, length=50):
        self.name = name
        self.D0 = D0
        self.radius_ratio = radius_ratio
        self.length = length
    def select(self, img, num):
        if self.name == 'gaussian':
            return get_low_high_f_by_gaussian_filter(img=img, D0=self.D0, num=num)
        elif self.name == 'circle':
            return get_low_high_f_by_circle_filter(img=img, radius_ratio=self.radius_ratio, num=num)
        elif self.name == 'square':
            return get_low_high_f_by_square_filter(img=img, length=self.length, num=num)
