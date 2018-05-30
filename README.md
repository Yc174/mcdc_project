# This project for the first MCDC competition
> The task is to detect the vehicle and select the car ahead, to predict the relative distance and speed.

## Detector
> We try [faster rcnn](https://github.com/endernewton/tf-faster-rcnn) and [yolo](https://pjreddie.com/darknet/yolo/) as our detector

## Predictor
> We try projection method, area, LSTM and [unsupervised depth estimation](https://github.com/mrharicot/monodepth) to predict the distance.