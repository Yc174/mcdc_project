import numpy as np
import os, json
import tensorflow as tf
import matplotlib.pyplot as plt


from .base_predictor import BasePredictor
from tools.filter import Exponentially_weighted_averages
from tools.bird_view_projection import read_cam_param, bird_view_proj

class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, LR, x_vx_mode):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.learning_rate = LR
        self.x_vx_mode = x_vx_mode
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [batch_size, None, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [batch_size, None, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [self.batch_size, -1, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)

        # lstm_cell = tf.contrib.rnn.MultiRNNCell(
        #     [lstm_cell() for _ in range(3)], state_is_tuple=True)

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)


    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            outputs = tf.matmul(l_out_x, Ws_out) + bs_out
            self.pred = tf.reshape(outputs, [self.batch_size, -1, self.output_size])
    def compute_cost(self):
        # if self.x_vx_mode == 'x' or self.x_vx_mode == 'vx':
        #     losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #         [tf.reshape(self.pred, [-1], name='reshape_pred')],
        #         [tf.reshape(self.ys, [-1], name='reshape_target')],
        #         [tf.ones([self.batch_size * self.n_steps * self.output_size], dtype=tf.float32)],
        #         average_across_timesteps=True,
        #         softmax_loss_function=self.ms_error,
        #         name='losses'
        #     )
        # elif self.x_vx_mode == 'x_vx':
        #     pass
        #     losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #                 [tf.reshape(self.pred[:,:,0], [-1], name='reshape_pred')],
        #                 [tf.reshape(self.ys[:,:,0], [-1], name='reshape_target')],
        #                 [tf.ones([self.batch_size * self.n_steps ], dtype=tf.float32)],
        #                 average_across_timesteps=True,
        #                 softmax_loss_function=self.ms_error,
        #                 name='losses'
        #             )+tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #                 [tf.reshape(self.pred[:,:,1], [-1], name='reshape_pred')],
        #                 [tf.reshape(self.ys[:,:,1], [-1], name='reshape_target')],
        #                 [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
        #                 average_across_timesteps=True,
        #                 softmax_loss_function=self.ms_error,
        #                 name='losses'
        #             )
        # else:
        #     pass

        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps * self.output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )

        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        # return tf.reduce_sum(tf.square(tf.subtract(labels, logits)))
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

class BirdProj():
    def __init__(self, cam_calib):
        # read camera parameters
        with open(cam_calib) as f:
            cam_param = json.load(f)
            self.K, self.dist_coeff, self.R, self.T = read_cam_param(cam_param)

    def proj(self, u, v):
        x, y = bird_view_proj(u, v,
                       self.K, self.dist_coeff, self.R, self.T)
        return x, y

bird_proj = BirdProj('/home/yanchao/dataset/mcdc_data/valid/camera_parameter.json')
HIGHT = 500.
WIDTH = 500.
class LSTMPredictor(BasePredictor):
    def __init__(self, args):
        BasePredictor.__init__(self, args)
        self.x_vx_mode = args.x_vx_mode
        self.lstm_predictor_model = args.lstm_predictor_model
        self.Tx = 50  #args.n_steps
        self.M =  100 #args.batch_size
        self.n_a = 16
        self.lr = args.learning_rate
        self.WIDTH = 500.
        self.HIGHT = 500.
        self.bboxes = []
        self.samples = []
        self.x_window = np.zeros(20, dtype = np.float32)
        self.pre_vx = 0.
        self.pre_x = 0.
        self.ave_x_pre = 0.

        if self.x_vx_mode == 'x':
            self.n_x = 6
            self.n_y = 1

        elif self.x_vx_mode == 'vx':
            self.n_x = 6
            self.n_y = 1

        elif self.x_vx_mode == 'x_vx':
            self.n_x = 6
            self.n_y = 1 #

        else:
            pass
        self.build_model_for_x()
#         self.build_model_for_vx()

    def get_test_batch(self, X_test, batch_size):
        X_test = np.asarray(X_test)
        X_test_batch = np.tile(X_test, (batch_size, 1, 1))
#         print('X_test_batch shape:',X_test_batch.shape)

        return X_test_batch

    def extract_bbox(self, b):
        '''
        "top": 597.832580566406,
        "right": 880.870239257812,
        "bot": 739.836364746094,
        "left": 686.165344238281
        '''

        h = (b[3] - b[1]+1.) / WIDTH
        w = (b[2]- b[0]+1.) / HIGHT
        area = h * w
        x, y = bird_proj.proj((b[0] + b[2])/2, b[3])

        # return [h, w, area, x]
        # return [h, w, area, area_dao]
        return [h, w, area, 1./h, 1./w, 1./area]

    def extract_sample(self, filename, time_file):
        with open(filename) as fin:
            gts = json.loads(fin.read())['frame_data']
        # t_samples = [extract_bbox(e['ref_bbox'])] for e in gts]
        t_samples = [self.extract_bbox(e['ref_bbox']) for e in gts]
        # t_targets = [e['vx'] for e in gts] # e['x'] for dis, e['vx'] for relative v
        t_targets = [[e['x'], e['vx']] for e in gts]  # e['x'] for dis, e['vx'] for relative v

        times = []
        with open(time_file, 'r') as fin:
            for line in fin.readlines():
                times.append(float(line))

        # add vx as new feature
        tvx = []
        for i in range(len(times)):
            if i == 0:
                tvx.append(0)
            else:
                tvx.append((t_samples[i][-1] - t_samples[i - 1][-1]) / (times[i] - times[i - 1]))

        # tvx[0] = tvx[1]
        # for i in range(len(times)):
        #     t_samples[i].append(tvx[i])

        return t_samples, t_targets

    def build_model_for_x(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        g2 = tf.Graph()
        self.sess2 = tf.Session(config=config, graph=g2)

        with self.sess2.as_default():
            with g2.as_default():
                LR = tf.Variable(self.lr, trainable=False)
                self.model_x = LSTMRNN(self.Tx, self.n_x, self.n_y, self.n_a, self.M, LR, self.x_vx_mode)
                saver = tf.train.Saver()
                saved_path = os.path.join(self.lstm_predictor_model, 'x','x'+'_lstm')
                saver.restore(self.sess2, saved_path+'_30000')

    def build_model_for_vx(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        g3 = tf.Graph()
        self.sess3 = tf.Session(config=config, graph=g3)

        with self.sess3.as_default():
            with g3.as_default():
                LR = tf.Variable(self.lr, trainable=False)
                self.model_vx = LSTMRNN(self.Tx, self.n_x, self.n_y, self.n_a, self.M, LR, self.x_vx_mode)
                saver = tf.train.Saver()
                saved_path = os.path.join(self.lstm_predictor_model, 'vx', 'vx'+'_lstm')
                saver.restore(self.sess3, saved_path+'_20000')


    def predict(self, bbox, time, fid):
        bbox = bbox[:4]
        self.bboxes.append(bbox)
        self.samples.append(self.extract_bbox(bbox))
        test_batch = self.get_test_batch(self.samples, self.M)
        feed_dict = {
            self.model_x.xs: test_batch,
        }
        test_pred_x = self.sess2.run(self.model_x.pred, feed_dict=feed_dict).astype(np.float32)

#         feed_dict = {
#             self.model_vx.xs: test_batch,
#         }
#         test_pred_vx = self.sess3.run(self.model_vx.pred, feed_dict=feed_dict).astype(np.float32)

        # print('test_pred_x shpe: ', test_pred_x.shape)
        # print('test_pred_vx shape: ', test_pred_vx.shape)
        x = test_pred_x[0,:,0][-1]
        x = Exponentially_weighted_averages(self.pre_x, x, fid, theta=0.8)     
        
#         vx= test_pred_vx[0,:,0][-1]
        if fid == 0:
            self.x_window = np.ones(20, dtype = np.float32) * x
            self.pre_x = x
        self.x_window = np.append(self.x_window, x)
        self.x_window = np.delete(self.x_window, 0)
        ave_x0 = self.x_window[:10].mean() 
        ave_x1 = self.x_window[10:].mean()
#         vx = (ave_x1 - ave_x0) / (0.04 * 10)
#         vx = (ave_x - self.ave_x_pre) / 0.04
#         self.ave_x_pre = ave_x
        
        vx = (x -self.pre_x) / 0.04
        if vx > 4. :
            vx = 4.
        elif vx < -4. :
            vx = -4.
        vx = Exponentially_weighted_averages(self.pre_vx, vx, fid, theta=0.92)
#         # x, vx = self.lstm_process(bbox,fid)
        self.pre_x = x
        self.pre_vx = vx
        cur_pred = {
            'fid': fid,
            'vx': float(vx),
            'x': float(x),
            'ref_bbox': {
                'left': float(bbox[0]), 'top': float(bbox[1]),
                'right': float(bbox[2]), 'bot': float(bbox[3])
            }
        }

        self.result.append(cur_pred)
#         if fid == 499:
#             for i, x in enumerate(test_pred_x[0,:,0]):
#                 self.result[i]['x'] = x
                
#             for i, vx in enumerate(test_pred[0,:,1]):
#                 self.result[i]['vx'] = vx

        return cur_pred

    def to_json(self, filename):
        print('Save prediction to ', filename)
        if len(self.result) > 1:
            self.result[0]['vx'] = self.result[1]['vx']
#         for i in range(10):
#             self.result[i]['vx'] = self.result[10]['vx']
        with open(filename, 'w') as fout:
            data = {'frame_data': self.result}
            # print('data:', data)
            # print("Ready save to json...")
            # json.dump(data, fout, indent=4, ensure_ascii=False)
            json.dump(data, fout)
            # print("Saved to json.")

    def lstm_process(self, bbox, fid):

        return x,vx