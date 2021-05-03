import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import shutil

'''
分两步 第一步 find_cnts 先将小图从大图中抠出来
第二步 final_koutu 将小图背景的其他不完整jingyuan去除

'''

def find_cnts(path):
    for image_file in tqdm(os.listdir(path)):
        # bar.update(1)
        if image_file.endswith('.bmp'):
            img_name = os.path.basename(image_file)
            # cv2.imshow('1', image)
            img_name = img_name.split('.')[0]
            image = cv2.imread(os.path.join(path, image_file))
            # cv2.imshow('img', image)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([11, 43, 43])

            upper_blue = np.array([34, 255, 255])

            mask = cv2.inRange(image_gray, lower_blue, upper_blue)
            res = cv2.bitwise_and(image, image, mask, mask)
            res = cv2.erode(res, np.ones((3, 3)))
            # cv2.imshow('1', res)
            bin = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            # kernel = np.ones((3, 3))
            _, bin = cv2.threshold(bin, 50, 255, cv2.THRESH_BINARY)
            cnts, h = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            i = 1
            for cnt in cnts:
                rect = cv2.minAreaRect(cnt)  # 根据cnt获取最小最小框 (中心(x,y), (宽,高), 旋转角度)
                if rect[1][0] * rect[1][1] > 52000 and rect[1][0] * rect[1][1] < 60000:
                    box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
                    box = np.int0(box)
                    x0 = np.min(box[:, 0])
                    y0 = np.max(box[:, 1])
                    x1 = np.max(box[:, 0])
                    y1 = np.min(box[:, 1])
                    # pad =
                    pts = np.array(((x0, y0), (x1, y1)), np.int32)
                    box_pts = np.zeros_like(box)

                    # box_pts[:, 0] = box[:, 0] - (pts[0, 0] - pad)
                    # box_pts[:, 1] = box[:, 1] - (pts[0, 1] - pad)
                    # print('box_pts', box_pts)
                    # print('box:', box)
                    # roi = image[y1:y0, x0:x1+pad]

                    # if roi.any() == True:
                    cv2.drawContours(image, [box], -1, (0, 64, 255), 10)

            cv2.imshow('d', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
                        # if roi.any() == True:
                    #     cv2.imshow('r', roi)
                    #     cv2.waitKey()
                    #     cv2.destroyAllWindows()
                    #     new_folder = './' + 'koutu_3'
                    #     if not os.path.exists(new_folder):
                    #         os.mkdir(new_folder)
                    #     # print(new_folder+os.sep+'x')
                    #     i = i + 1
                    #     cv2.imwrite(new_folder + os.sep + str(i) + '.png', roi)


def final_koutu(path):
    j = 0
    for file in os.listdir(path):
        if file.endswith('.png'):
            # j = 0
            image = cv2.imread(os.path.join(path, file))
            # print(file)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([11, 43, 43])
            upper_blue = np.array([34, 255, 255])
            mask = cv2.inRange(image_gray, lower_blue, upper_blue)
            ret = cv2.bitwise_and(image, image, mask, mask)
            ret = cv2.erode(ret, np.ones((5, 5)))
            # cv2.imshow('r', ret)

            bin = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
            # bin = cv2.blur(bin, (3, 3))
            # cv2.imshow('b', image_blur)
            # edge = cv2.Canny(bin, 50, 120)
            # kernel = np.ones((0, 0))
            # bin = cv2.dilate(edge, kernel)
            # cv2.imshow('e', bin)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            _, th_bin = cv2.threshold(bin, 50, 255, cv2.THRESH_BINARY)
            # cv2.imshow('t', th_bin)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(bin, connectivity=8)
            # j = 0
            cnts, h = cv2.findContours(th_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in cnts:
                rect = cv2.minAreaRect(cnt)
                # m = rect[0][1]*rect[1][1]
                box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
                box = np.int0(box)
                x0 = np.min(box[:, 0])
                y0 = np.max(box[:, 1])
                x1 = np.max(box[:, 0])
                y1 = np.min(box[:, 1])
                h = y0 - y1
                w = x1 - x0
                m_1 = h * w

                if m_1 > 40000:
                    pad = 20

                    # cv2.drawContours(image, [box], -1, (0, 64, 255), 3)

                    roi = image[y1:y0, x0:x1 + pad]
                    angle = rect[2]
                    if rect[1][0] > rect[1][1]:  # 如果宽比高大
                        angle = angle + 90
                    rh, rw = roi.shape[:2]
                    center = (rh/2, rw/2)
                    M = cv2.getRotationMatrix2D(center, angle, 1)
                    rotated = cv2.warpAffine(roi, M, (rw, rh))

                    r_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)

                    lower_blue = np.array([11, 43, 43])
                    upper_blue = np.array([34, 255, 255])

                    mask = cv2.inRange(r_gray, lower_blue, upper_blue)
                    ret_r = cv2.bitwise_and(rotated, rotated, mask, mask)
                    ret_r = cv2.erode(ret_r, np.ones((5, 5)))
                    rotated_gray = cv2.cvtColor(ret_r, cv2.COLOR_BGR2GRAY)
                    _, rotated_th = cv2.threshold(rotated_gray, 50, 255, cv2.THRESH_BINARY)
                    cnts, _ = cv2.findContours(rotated_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    m = []
                    # j = 0
                    for cnt in cnts:
                        rect = cv2.minAreaRect(cnt)
                        # m = rect[0][1]*rect[1][1]
                        box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
                        box = np.int0(box)
                        x0 = np.min(box[:, 0])
                        y0 = np.max(box[:, 1])
                        x1 = np.max(box[:, 0])
                        y1 = np.min(box[:, 1])
                        h = y0 - y1
                        w = x1 - x0
                        m_1 = h * w
                        if m_1 > 20000 and m_1 < 60000:
                            # m.append(m_1)
                            # print(m)
                            pad = 2

                            # cv2.drawContours(rotated, [box], -1, (0, 64, 255), 3)

                            rotated_roi = rotated[y1:y0, x0:x1+pad]
                            new_folder = './' + 'final_koutu'
                            if not os.path.exists(new_folder):
                                os.mkdir(new_folder)
                            # print(new_folder+os.sep+'x')
                            j = j + 1
                            # print(j)
                            cv2.imwrite(new_folder + os.sep + str(j) + '.png', rotated_roi)
                    #         cv2.imshow('e', rotated_roi)
                    #
                    # # cv2.imshow('r', rotated)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
        # m = rect[0][1]*rect[1][1]
        # if m > 40000:
        #     cv2.imshow('d', image)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        #     print(m)


path_final_koutu = '/Users/pengzhang/Desktop/imag_modify/koutu1'
final_koutu(path_final_koutu)
print('DONE!!!')
