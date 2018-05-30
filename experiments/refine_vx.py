
#coding=utf-8
from __future__ import print_function
import glob
import json
from tools.filter import Exponentially_weighted_averages

def do_window_average(pre_file, time_file):
    with open(pre_file) as fin:
        pre = json.loads(fin.read())['frame_data']
    times = []
    with open(time_file, 'r') as fin:
        for line in fin.readlines():
            times.append(float(line))
            # if len(times) > 1:
            #     print('diff = ', times[-1] - times[-2])

    new_pre = []

    for i in range(len(times)):
        if i < 3:
            new_pre.append(pre[i])
            pre_vx = pre[i]['vx']
        else:
            t_diff = (times[i] - times[i - 3]) / 3
            d_diff = ((pre[i]['x'] - pre[i-2]['x']) + (pre[i-1]['x'] - pre[i-3]['x'])) / 2
            vx = d_diff / (t_diff * 2)

            vx = Exponentially_weighted_averages(pre_vx, vx, i, theta=0.9)
            pre_vx = vx
            new_pre.append(pre[i])
            new_pre[i]['vx'] = vx


    filename = pre_file[:-8] + 'pre_w.json'
    print(' >> save to ', filename)
    with open(filename, 'w') as fout:
        data = {'frame_data': new_pre}
        json.dump(data, fout, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    valid_pre_flies = glob.glob('./valid_pre/*pre.json')
    for file in valid_pre_flies:
        print(file)
        time_file = file[:-8] + 'time.txt'
        do_window_average(file, time_file)
