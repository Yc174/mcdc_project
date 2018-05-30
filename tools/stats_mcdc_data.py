#coding=utf-8
from __future__ import print_function
import time
import argparse
from glob import glob
import os, cv2
import json

def show(image_path, bbox):
    print(image_path, bbox)

    im = cv2.imread(image_path)
    x, y, w, h = bbox

    # left = int(x - w / 2)
    # right = int(x + w / 2)
    # top = int(y - h / 2)
    # bottom = int(y + h / 2)

    left = int(x)
    top = int(y)
    right = int(x + w)
    bottom = int(y + h)

    cv2.rectangle(im, (left, top), (right, bottom), color=[0, 255, 0], thickness=3)

    im = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))
    cv2.imshow('image', im)
    # draw_bbox_with_center(arr, r)
    k = cv2.waitKey(0)

    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('messigray.png', im)
        cv2.destroyAllWindows()

def show_with_center(image_path, bbox):
    print(image_path, bbox)

    im = cv2.imread(image_path)
    x, y, w, h = bbox

    left = int(x - w / 2)
    right = int(x + w / 2)
    top = int(y - h / 2)
    bottom = int(y + h / 2)

    cv2.rectangle(im, (left, top), (right, bottom), color=[0, 255, 0], thickness=3)

    im = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))
    cv2.imshow('image', im)
    # draw_bbox_with_center(arr, r)
    k = cv2.waitKey(0)

    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('messigray.png', im)
        cv2.destroyAllWindows()


if __name__ == '__main__':
#     data_dir  = '/home/hzshuai/mcdc/mcdc_data'
    data_dir = '/data/mcdc_data'
    train_dir = data_dir + '/train/train_images'
    label_dir = '/home/m12/mcdc_data/train/train_labels'
    ann_file  = data_dir + '/train/MCDC_train_100000.coco.json'

    with open(ann_file) as fin:
        ann = json.loads(fin.read())
#     with open(label_dir + '/train_format.json', 'w') as fout:
#         json.dump(ann, fout, indent=4, ensure_ascii=False)

    ann_map = {}
    cls = {}
    for im in ann['images']:
        ann_map[im['id']] = im

    for a in ann['annotations']:
        if 'car_rear' in a and 'rear_box' in a['car_rear'] and a['image_id'] in ann_map:

            if 'ann' not in ann_map[a['image_id']]:
                ann_map[a['image_id']]['ann'] = []

            ann_map[a['image_id']]['ann'].append(a)

            if a['type'] not in cls:
                cls[a['type']] = 0
            cls[a['type']] += 1
            
# {u'xiaoxingche': 189955, u'gongchengche': 305, u'huoche': 12975, u'unknown': 63462, u'sanlunche': 6684, u'others': 228, u'gongjiaokeche': 20610}
# 96104

#{u'xiaoxingche': 18813, u'gongchengche': 26, u'huoche': 1267, u'unknown': 6244, u'sanlunche': 642, u'others': 19, u'gongjiaokeche': 1912}

            # if a['type'] == 'unknown' and cls[a['type']] % 23 == 0:
            # if a['image_id'] == 0:
            #     image_path = train_dir + '/' + ann_map[a['image_id']]['file_name']
            #     show(image_path, a['car_rear']['rear_box'])

    print(cls)

    im_list = []
    cls = ['xiaoxingche', 'gongchengche', 'huoche', 'unknown', 'sanlunche', 'others', 'gongjiaokeche']
    for k, image in ann_map.iteritems():
        if 'ann' in image:
#             print(k)
            # print(k, image)

            image_path = train_dir + '/' + image['file_name']
            im_list.append(image_path)
            txt_path = label_dir + '/' + image['file_name'][:-4] + '.txt'
            
            dirname = os.path.dirname(txt_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            
            dw, dh = 1./image['width'], 1./image['height']
#             print(txt_path, 1./dw, 1./dh)
            with open(txt_path, 'w') as fout:
                for a in image['ann']:
                    # print(a)

                    x, y, w, h = a['car_rear']['rear_box']
                    # show(image_path, (x, y, w, h))

                    x = x + w / 2.
                    y = y + h / 2.

                    # show_with_center(image_path, (x, y, w, h))

                    x *= dw
                    y *= dh
                    w *= dw
                    h *= dh

                    bb = [x, y, w, h]
                    cls_id = cls.index(a['type'])
                    fout.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

            # break

    print(txt_path)
    print(len(im_list))
    with open(label_dir + '/train.txt', 'w') as fout:
        for e in im_list:
            fout.write(e + '\n')

    with open(label_dir + '/valid.txt', 'w') as fout:
        for i, e in enumerate(im_list):
            if i % 10 == 0:
                fout.write(e + '\n')

