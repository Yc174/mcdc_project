#!bin/sh

for GPU_ID in {2..7}
do
    echo ${GPU_ID}
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py --gpu ${GPU_ID} --input-dir=/data/mcdc_data/valid/ --output-dir=valid_pre --cam-calib=/data/mcdc_data/valid/camera_parameter.json --detector='yolo-v3' --data-type=val &
    #sleep 10
done


