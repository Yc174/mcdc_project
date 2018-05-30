from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
from collections import OrderedDict, defaultdict
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

categories = {'xiaoxingche':1, 'huoche':2, 'gongjiaokeche':3, 'gongchengche':4, 'sanlunche':5, 'others':6, 'unknown':7, 'none':8}


class MCDC(COCO):
  def createIndex(self):
    # create index
    print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
      for ann in self.dataset['annotations']:
        imgToAnns[ann['image_id']].append(ann)
        anns[ann['id']] = ann

    if 'images' in self.dataset:
      for img in self.dataset['images']:
        imgs[img['id']] = img

    if 'categories' in self.dataset:

        cats = self.get_categories()

    if 'annotations' in self.dataset and 'categories' in self.dataset:
      for ann in self.dataset['annotations']:
        catToImgs[categories[ann['type']]].append(ann['image_id'])

    print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats
    self.dataset['categories'] = [cat for id, cat in cats.items()]

  def get_categories(self,):
    cats = {}
    id_to_cats = {}
    for name, id in categories.items():
      mcdc_cat = {'supercategory': 'MCDC', 'id': id, 'name': name}
      cats[id] = mcdc_cat

    return cats

class mcdc(imdb):
  def __init__(self, number, image_set):
    imdb.__init__(self, 'mcdc_' + image_set + '_' + number)
    # COCO specific config options
    self.config = {'use_salt': True,
                   'cleanup': True}
    # name, paths
    self._numbuer = number
    self._image_set = image_set
    self._data_path = osp.join(cfg.DATA_DIR, 'mcdc')
    # load COCO API, classes, class <-> id mappings
    # self._COCO = COCO(self._get_ann_file())
    self._COCO = MCDC(self._get_ann_file())

    # self._COCO.dataset['categories'] = self.adpt_categories()
    # self._COCO.cats = self._get_cats()
    # self._COCO.catToImgs = self._cat_to_Imgs()

    cats = self._COCO.loadCats(self._COCO.getCatIds())
    self._classes = tuple(['__background__'] + [c['name'] for c in cats])
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats],
                                               self._COCO.getCatIds())))
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    roi = self.gt_roidb()
    self.set_proposal_method('gt')
    self.competition_mode(False)

    # Some image sets are "views" (i.e. subsets) into others.
    # For example, minival2014 is a random 5000 image subset of val2014.
    # This mapping tells us where the view's images and proposals come from.
    self._view_map = {
      'minival2014': 'val2014',  # 5k val2014 subset
      'valminusminival2014': 'val2014',  # val2014 \setminus minival2014
      'test-dev2015': 'test2015',
    }
    coco_name = image_set + number  # e.g., "val2014"
    self._data_name = (self._view_map[coco_name]
                       if coco_name in self._view_map
                       else coco_name)
    # Dataset splits that have ground-truth annotations (test splits
    # do not have gt annotations)
    self._gt_splits = ('train', 'val', 'minival')

  def adpt_categories(self):

      cats=[]
      id_to_cats = {}
      for name, i in categories.items():
        id_to_cats[i] = name
      sorted(id_to_cats)
      for ind, name in id_to_cats.items():
        mcdc_cat = {'supercategory': 'MCDC', 'id': ind, 'name': name}
        cats.append(mcdc_cat)

      return cats

  def _get_cats(self):
    cats = {}
    for cat in self._COCO.dataset['categories']:
      cats[cat['id']] = cat

    return cats

  def _cat_to_Imgs(self):
    catToImgs = defaultdict(list)
    for ann in self._COCO.dataset['annotations']:
      cat_id = categories[ann['type']]
      catToImgs[cat_id].append(ann['image_id'])
    return catToImgs

  def _get_ann_file(self):
    prefix = 'instances' if self._image_set.find('test') == -1 \
      else 'image_info'
    path =  osp.join(self._data_path,
                     'MCDC_'+ self._image_set + '_' + self._numbuer + '.coco.json')
    return path

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    image_ids = self._COCO.getImgIds()
    return image_ids

  def _get_widths(self):
    anns = self._COCO.loadImgs(self._image_index)
    widths = [ann['width'] for ann in anns]
    return widths

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    # file_name = ('COCO_' + self._data_name + '_' +
    #              str(index).zfill(12) + '.jpg')
    # image_path = osp.join(self._data_path, 'images',
    #                       self._data_name, file_name)

    file_name = self._COCO.imgs[index]['file_name']
    image_path = osp.join(self._data_path, 'train_images', file_name)

    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    print('cache file:', cache_file)
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_coco_annotation(index)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_coco_annotation(self, index):
    """
    Loads COCO bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    im_ann = self._COCO.loadImgs(index)[0]
    width = im_ann['width']
    height = im_ann['height']

    annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
    objs = self._COCO.loadAnns(annIds)
    # Sanitize bboxes -- some are invalid
    valid_objs = []
    for obj in objs:
      if isinstance(obj['car_rear'], basestring):
        continue
      # print('obj', obj)
      rear_box = obj['car_rear']['rear_box']
      x1 = np.max((0, rear_box[0]))
      y1 = np.max((0, rear_box[1]))
      # x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
      # y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
      # x1 = obj['bbox'][0]
      # y1 = obj['bbox'][1]
      x2 = np.min((width - 1, rear_box[0] + np.max((0, rear_box[2] - 1))))
      y2 = np.min((height - 1, rear_box[1] + np.max((0, rear_box[3] - 1))))
      if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
        obj['clean_bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)
    objs = valid_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Lookup table to map from COCO category ids to our internal class
    # indices
    coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                      self._class_to_ind[cls])
                                     for cls in self._classes[1:]])

    for ix, obj in enumerate(objs):
      cls = coco_cat_id_to_class_ind[self._class_to_ind[obj['type']]]
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      seg_areas[ix] = obj['area']
      if obj['iscrowd']:
        # Set overlap to -1 for all classes for crowd objects
        # so they will be excluded during training
        overlaps[ix, :] = -1.0
      else:
        overlaps[ix, cls] = 1.0

    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _load_coco_annotation_old(self, index):
    """
    Loads COCO bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    im_ann = self._COCO.loadImgs(index)[0]
    width = im_ann['width']
    height = im_ann['height']

    annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
    objs = self._COCO.loadAnns(annIds)
    # Sanitize bboxes -- some are invalid
    valid_objs = []
    for obj in objs:
      x1 = np.max((0, obj['bbox'][0]))
      y1 = np.max((0, obj['bbox'][1]))
      # x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
      # y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
      # x1 = obj['bbox'][0]
      # y1 = obj['bbox'][1]
      x2 = np.min((width - 1, obj['bbox'][0] + np.max((0, obj['bbox'][2] - 1))))
      y2 = np.min((height - 1, obj['bbox'][1] + np.max((0, obj['bbox'][3] - 1))))
      if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
        obj['clean_bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)
    objs = valid_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Lookup table to map from COCO category ids to our internal class
    # indices
    coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                      self._class_to_ind[cls])
                                     for cls in self._classes[1:]])

    for ix, obj in enumerate(objs):
      cls = coco_cat_id_to_class_ind[self._class_to_ind[obj['type']]]
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      seg_areas[ix] = obj['area']
      if obj['iscrowd']:
        # Set overlap to -1 for all classes for crowd objects
        # so they will be excluded during training
        overlaps[ix, :] = -1.0
      else:
        overlaps[ix, cls] = 1.0

    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def _get_box_file(self, index):
    # first 14 chars / first 22 chars / all chars + .mat
    # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
    file_name = ('COCO_' + self._data_name +
                 '_' + str(index).zfill(12) + '.mat')
    return osp.join(file_name[:14], file_name[:22], file_name)

  def _print_detection_eval_metrics(self, coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
      ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                     (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
      iou_thr = coco_eval.params.iouThrs[ind]
      assert np.isclose(iou_thr, thr)
      return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
      coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      # minus 1 because of __background__
      precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
      ap = np.mean(precision[precision > -1])
      print('{:.1f}'.format(100 * ap))

    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()

  def _do_detection_eval(self, res_file, output_dir):
    ann_type = 'bbox'
    coco_dt = self._COCO.loadRes(res_file)
    coco_eval = COCOeval(self._COCO, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    self._print_detection_eval_metrics(coco_eval)
    eval_file = osp.join(output_dir, 'detection_results.pkl')
    with open(eval_file, 'wb') as fid:
      pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
    print('Wrote COCO eval results to: {}'.format(eval_file))

  def _coco_results_one_category(self, boxes, cat_id):
    results = []
    for im_ind, index in enumerate(self.image_index):
      dets = boxes[im_ind].astype(np.float)
      if dets == []:
        continue
      scores = dets[:, -1]
      xs = dets[:, 0]
      ys = dets[:, 1]
      ws = dets[:, 2] - xs + 1
      hs = dets[:, 3] - ys + 1
      results.extend(
        [{'image_id': index,
          'category_id': cat_id,
          'bbox': [xs[k], ys[k], ws[k], hs[k]],
          'score': scores[k]} for k in range(dets.shape[0])])
    return results

  def _write_coco_results_file(self, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                       self.num_classes - 1))
      coco_cat_id = self._class_to_coco_cat_id[cls]
      results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                     coco_cat_id))
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
      json.dump(results, fid)

  def evaluate_detections(self, all_boxes, output_dir):
    res_file = osp.join(output_dir, ('detections_' +
                                     self._image_set +
                                     self._numbuer +
                                     '_results'))
    if self.config['use_salt']:
      res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    self._write_coco_results_file(all_boxes, res_file)
    # Only do evaluation on non-test sets
    if self._image_set.find('test') == -1:
      self._do_detection_eval(res_file, output_dir)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


def show_with_center(image_path, bboxes, gt_classes, classes_name):
    import cv2
    print(image_path, bboxes)

    im = cv2.imread(image_path)
    for bbox, bbox_class in zip(bboxes,gt_classes):

      # x, y, w, h = bbox
      #
      # left = int(x - w / 2)
      # right = int(x + w / 2)
      # top = int(y - h / 2)
      # bottom = int(y + h / 2)
      #
      # cv2.rectangle(im, (left, top), (right, bottom), color=[0, 255, 0], thickness=3)
      cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=[0, bbox_class*30, 0], thickness=3)
      font = cv2.FONT_HERSHEY_SIMPLEX
      im = cv2.putText(im, classes_name[bbox_class], (bbox[0], bbox[1]), font, 1.2, (255, 255, 255), 2)


    im = cv2.resize(im, (int(im.shape[1]/2), int(im.shape[0]/2)))
    cv2.imshow('image', im)
    # draw_bbox_with_center(arr, r)
    # k = cv2.waitKey(0)

    # if k == 27:  # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    # elif k == ord('s'):  # wait for 's' key to save and exit
    #     cv2.imwrite('messigray.png', im)
    #     cv2.destroyAllWindows()

if __name__ == '__main__':
    print('ok')

    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    # img = np.zeros((512, 512, 3), np.uint8)
    # cv2.rectangle(img, (20, 50), (411, 411), (55, 255, 155), 5)
    # plt.imshow(img, 'brg')

    dataset = mcdc(str(1000), 'train')
    classes = dataset.classes
    for i in dataset.image_index:
      img_name = dataset._COCO.imgs[i]['file_name']
      print(i, img_name)
      img_path = osp.join(dataset._data_path, 'train_images', img_name)
      bboxes = dataset.roidb[i]['boxes']
      gt_classes = dataset.roidb[i]['gt_classes']
      # bboxes = np.array(bboxes)[:,:4]
      show_with_center(img_path, bboxes, gt_classes, classes)

    print('done!')