import os
import cv2
import numpy as np
import json
from tqdm import tqdm


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
            kernel = np.ones((3, 3))
            _, bin = cv2.threshold(bin, 50, 255, cv2.THRESH_BINARY)
            # bin = cv2.erode(bin, kernel)
            # cv2.imshow('bing', bin)
            # img_blur = cv2.blur(bin, (3, 3))
            # edges = cv2.Canny(img_blur, 30, 120)

            # edges = cv2.erode(edges, kernel)
            # edges = cv2.dilate(edges, kernel)
            # cv2.imshow('e', edges)
            # bin = cv2.dilate(bin, np.ones((5, 5), dtype=np.int8))
            # bin = cv2.erode(bin, np.ones((5, 5), dtype=np.int8))
            # cv2.imshow('bing2', bin)
            cnts, h = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            box_list = []
            obj_dict = {}
            i = 1
            for cnt in cnts:
                rect = cv2.minAreaRect(cnt)  # 根据cnt获取最小最小框 (中心(x,y), (宽,高), 旋转角度)
                if rect[1][0] * rect[1][1] > 52000 and rect[1][0] * rect[1][1] < 60000:
                    # print(rect[1][0] * rect[1][1])
                    # print(rect[1][0]*rect[1][1])
                    box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
                    box = np.int0(box)
                    # print(box)
                    x0 = np.min(box[:, 0])
                    y0 = np.max(box[:, 1])
                    x1 = np.max(box[:, 0])
                    y1 = np.min(box[:, 1])
                    h_ori = int(rect[1][1])
                    w_ori = int(rect[1][0])
                    # print(x0)
                    # print(y0)
                    h_roi = int(rect[1][0])
                    w_roi = int(rect[1][1])

                    # obj = ([x0, y0], [x1, y1], rect[1])
                    # obj_list.append(obj)
                    obj = (box, h_ori, w_ori, h_roi, w_roi)
                    box_list.append(obj)
                    # for i in range(len(box_list)):
                    #     obj_dict[i] = {'box': box_list[i][0], 'h_ori': box_list[i][1], 'w_ori': box_list[i][2],
                    #                    'h_roi': box_list[i][3], 'w_roi': box_list[i][4]}
                    # np.save('1.npy', obj_dict)

                    # pad = 20
                    # roi = image[y0-h:y0, x0:x0+w]
                    # cv2.drawContours(image, [box], -1, (0, 64, 255), 3)
                    # roi = image[(y0 - h_roi):y0 + pad, x0:(x0 + w_roi + pad)]
                    roi = image[y1:y0, x0:x1]

                    cv2.imshow('i', roi)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    # if roi.any() == True:
                    #     new_folder = './' + 'koutu'
                    #     if not os.path.exists(new_folder):
                    #         os.mkdir(new_folder)
                    #     # print(new_folder+os.sep+'x')
                    #     i = i + 1
                    #     cv2.imwrite(new_folder + os.sep + str(i) + '.png', roi)


path = '/users/pengzhang/desktop/snap2'
find_cnts(path)
print('DONE!!!')

