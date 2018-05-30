#!bin/sh

cd mcdc_project

for GPU_ID in {2..7}
do
    echo ${GPU_ID}
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py --gpu ${GPU_ID} --input-dir=/data/mcdc_data/test/ --output-dir=/home/m12/test_pre --cam-calib=/data/mcdc_data/test/camera_parameter.json --detector='tf-faster-rcnn' &
done

