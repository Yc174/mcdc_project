#from __future__ import print_function
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import sys
from cython_bbox import bbox_overlaps
from projection import ground_point_to_bird_view_proj
from projection import bird_view_proj_to_ground_point as bv2gp


    #cv2.imwrite('output/'+str(tot)+".jpg", image)
    for x in range(5, 50, 3):
	for y in range(-3, 4):
		u, v = bv2gp(x, y, cam_param)
		cv2.putText(image, str(x) + "," + str(y), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
		cv2.circle(image, (u, v), 3, (0, (x + y + 103) % 2 * 255, (x + y + 102) % 2 * 255), -1) 
    cv2.imwrite('output_lyc/'+ str(nb) + '_' + str(tot)+".jpg", image)
