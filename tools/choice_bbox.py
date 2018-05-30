import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def find_middle_car_use_iou(boxes, rang=None):
    if rang.all() == None:
        rang = [500, 500, 1000, 1000]
        rang = np.array(rang)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # rang_x1=rang[0,0]
    # rang_y1=rang[0,1]
    # rang_x2=rang[0,2]
    # rang_y2=rang[0,3]
    rang_x1=rang[0]
    rang_y1=rang[1]
    rang_x2=rang[2]
    rang_y2=rang[3]

    rang_area=(rang_y2-rang_y1+1)*(rang_x2-rang_x1+1)

    xx1 = np.maximum(rang_x1, boxes[:, 0])
    yy1 = np.maximum(rang_y1, boxes[:, 1])
    xx2 = np.minimum(rang_x2, boxes[:, 2])
    yy2 = np.minimum(rang_y2, boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    ovr=inter/(areas+rang_area-inter)
    inds = ovr.argsort()[::-1]

    return boxes[inds[0]]



def vis_detections(im, class_name, dets, thresh=0.5, video= None,fid=0):
    """Draw detected bounding boxes."""
    dirname = os.path.dirname(__file__)
    show_dir = os.path.join(dirname, '..', 'show/%s' % os.path.basename(video))
    # print(show_dir)
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('%s/all_bboxes_%d.jpg' % (show_dir, fid))
    # plt.show()