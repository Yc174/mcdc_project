#write by yanchao
import matplotlib.pyplot as plt
import numpy as np
import error_estimation
import glob
import json
import os

def draw_subplot(x,y,y_label='vx',line='truth',):
    if line=='truth':
        #draw real line
        plt.plot(x,y,color='red',label='truth')
        # plt.xlabel('time-axis label')
        plt.ylabel(y_label)
        # plt.title(y_label+' line plot')
        # plt.show()
    elif line=='predict':
        #draw ground truth
        plt.plot(x,y,color='blue',label='predict')
        # plt.xlabel('time-axis label')
        plt.ylabel(y_label)
        # plt.title(y_label+' line plot')
        # plt.show()
    else:
        pass
    plt.legend(loc='upper right')
    
def draw_x_vx(time, x, vx, name):
    plt.figure(name)
    plt.plot(time,x,'b',time,vx,'b-')
    dir_path = os.path.dirname(__file__)
    show_path = os.path.join(dir_path, '..', 'show/curve_both_detector_predictor/test/')
    plt.savefig(show_path+name.split('.')[0]+'.png')
    print(show_path+name.split('.')[0]+'_pre.png')
    plt.show()

def draw_one_time(time,x,vx,gt_x,gt_vx,name):
    plt.figure(name)
    plt.subplot(211)
    draw_subplot(time, x, y_label='x', line='predict' )
    draw_subplot(time, gt_x, y_label='x', line='truth')
    plt.title(name)
    plt.subplot(212)
    draw_subplot(time,vx,y_label='vx',line='predict')
    draw_subplot(time,gt_vx,y_label='vx',line='truth')
    dir_path = os.path.dirname(__file__)
    show_path = os.path.join(dir_path, '..', 'show/curve_both_detector_predictor/valid/')
    plt.savefig(show_path+name.split('.')[0]+'.png')
    print(show_path+name.split('.')[0]+'.png')
    plt.show()
    
def err_estimation(result_file, gt_file, before, name = None):
    with open(result_file) as f:
        result = json.load(f)
        fids = [r['fid'] for r in result["frame_data"]]
        # vx = [r['vx'] * 5.5 for r in result["frame_data"]]
        # x = [r['x'] * 5.5 + 4.  for r in result["frame_data"]]
        vx = [r['vx'] for r in result["frame_data"]]
        x = [r['x'] for r in result["frame_data"]]

    with open(gt_file) as f:
        ground_truth = json.load(f)
        gt_x = [gt['x'] for gt in ground_truth["frame_data"]]
        gt_vx = [gt['vx'] for gt in ground_truth["frame_data"]]
    # before = 10
    # print(len(fids), len(x), len(vx), len(gt_x), len(gt_vx))
    draw_one_time(fids[:before], x[:before], vx[:before], gt_x[:before], gt_vx[:before], name)

    err_x = error_estimation.x_err_esti(x[:before], gt_x[:before])
    err_vx = error_estimation.vx_err_esti(vx[:before], gt_vx[:before])
    high = error_estimation.high_esti(x[:before], gt_x[:before])
    print('Error for x :', err_x)
    print('Error for vx:', err_vx)
    print('high        :', high)
    return err_x, err_vx, high

if __name__ == '__main__':
    #test draw subplot
    # x=np.arange(-20,20,1)
    # y1=(3*x*x-12)/2.
    # y2 = (3 * x * x - 12) / 2.+np.random.randn(len(x))*10
    # draw_line(x,y1,line='real')
    # draw_line(x,y2)
    # plt.show()

    #test draw_one_tome
    # time = np.arange(-20,20,1)
    # x = (3*time**2-12)/2.
    # vx = np.sin(time)
    # gt_x = (3*time**2-12)/2.+np.random.randn(len(time))*10
    # gt_vx = np.sin(time)+np.random.randn(len(time))
    # draw_one_time(time,x,vx,gt_x,gt_vx,name='the first time')
    # plt.show()

    valid_gt_flies = glob.glob('./valid_pre/*gt.json')

    for file in valid_gt_flies:
        print(file)

    valid_result_files = [v[:-7] + 'pre.json' for v in valid_gt_flies]
    for file in valid_result_files:
        print(file)

    err_x_list = []
    vx_list = []
    for result_file, gt_file in zip(valid_result_files, valid_gt_flies):
        if os.path.exists(result_file):
            (err_x, vx, high) = err_estimation(result_file, gt_file, 100, gt_file.split('/')[-1])
            err_x_list.append(err_x)
            vx_list.append(vx)

    print('Average err_x : ', np.mean(np.array(err_x_list)))
    print('Average err_vx: ', np.mean(np.array(vx_list)))


