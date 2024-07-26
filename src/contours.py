import cv2
import numpy as np

import scipy
import scipy.spatial
from scipy.signal import savgol_filter

import heapq

def approximte_contour(cnt: np.array, epsilon=0.04):
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    return approx



def n_max_contours(mask, n=1):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return  sorted(contours, key=cv2.contourArea)[:n]

def bounding_rectangle(cnt):
    bb = cv2.minAreaRect(cnt)
    return np.int0(cv2.boxPoints(bb))

def plate_width_line(rect1, rect2):
    rect1, rect2 = np.squeeze(rect1), np.squeeze(rect2)
    d = scipy.spatial.distance.cdist(rect1, rect2)
    (p11_idx, p21_idx), (p12_idx, p22_idx) = heapq.nsmallest(2, np.ndindex(d.shape), key=d.__getitem__)
    
    line1 = rect1[p11_idx], rect1[p12_idx]
    line2 = rect2[p21_idx], rect2[p22_idx]
    return line1, line2

def calculate_projection(line, p):
    a = line[1] - line[0]
    a = a / np.linalg.norm(a)
    p1 = p - line[0]
    proj = np.dot(p1, a) * a + line[0]
    return proj

def calculate_bias(line1, line2):
    p11, p12 = line1
    p21, p22 = line2
    
    proj = calculate_projection(line2, p11)
    bias = np.linalg.norm(proj - p21)
    return bias

def line_intersection_contur(lines, contour):
    contour = np.squeeze(contour)
    m = contour.shape[0]
    results = []
    for line in lines:
        d = scipy.spatial.distance.cdist(contour, line)
        i1, i2 = np.argmin(d, axis=0)
        c1 = np.take(contour, np.arange(i1, i2 + m * (i1 > i2) + 1), mode='wrap', axis=0)
        c2 = np.take(contour, np.arange(i2, i1 + m * (i2 > i1) + 1), mode='wrap', axis=0)

        c1 = np.expand_dims(c1, axis=1)
        c2 = np.expand_dims(c2, axis=1)

        results.append(min([c1, c2], key=lambda x: cv2.arcLength(x, closed=False)))
    return  results


def contour_mass_center(cnt):
    m = cv2.moments(cnt)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])

    cx, cy

def find_deviation_peaks(line, cnt, threshold):
    cnt = np.squeeze(cnt)
    p1, p2 = line
    a = p2 - p1
    a = a / np.linalg.norm(a)
    projections = np.stack([np.dot(cnt - p1, a)] * 2, axis=-1) * np.stack([a] * cnt.shape[0])
    d = np.linalg.norm(cnt - p1 - projections, axis=-1)
    dets  = -np.linalg.det(np.stack([np.stack([a] * cnt.shape[0], axis=0), cnt - p1 - projections], axis=-1))
    dets = np.sign(dets)
    # window_size = max(d.shape[0] // 5, 1)
    # d = savgol_filter(d, window_size, window_size - 1)
    d = np.multiply(dets, d)
    idx = [np.argmax(d), np.argmin(d)]
    dist = d[idx]
    cnts = cnt[idx]
    projs = (projections[idx] + p1).astype(np.uint32)
    return dist, idx, cnts, projs


def calculate_projection_line_width(line1, line2):
    p1 = calculate_projection(line1, line2[0])
    p2 = calculate_projection(line1, line2[1])

    return np.linalg.norm(p1 - p2)
