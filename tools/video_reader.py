#coding=utf-8
import cv2

# import pylab
# import imageio
# import skimage
# import numpy as np

class VideoReader:
    def __init__(self, video):

        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            print('Error: Can not read video', video)

    def next(self):
#         frame = cv2.imread('./test/car.jpg')

        ret, frame = self.cap.read()
        if ret == False:
            self.cap.release()
            cv2.destroyAllWindows()
            print("End of video")

        return frame
#
# # 可以选择解码工具
# vid = imageio.get_reader(video, 'ffmpeg')
# for num, im in enumerate(vid):
#     # image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary
#     image = skimage.img_as_float(im).astype(np.float32)
#     fig = pylab.figure()
#     fig.suptitle('image #{}'.format(num), fontsize=20)
#     pylab.imshow(image)
# pylab.show()

if __name__ == '__main__':
    # 可以选择解码工具
    video = './test/valid_video_00.avi'
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print('Error: Can not read video', video)
    else:
        while True:
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('video', frame)
                k = cv2.waitKey(20)
                # q to exit
                if (k & 0xff == ord('q')):
                    break
            else:
                cap.release()
                cv2.destroyAllWindows()
                print("End of video or can not read the video")