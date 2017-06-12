import tensorflow as tf
import numpy as np

class RecursiveNet(object):

    # Extract features for a single time step
    def extract_features(self, input_data=None, reuse=True):
        if input_data is None:
            input_data = self.data
        x_r = input_data
        
        layers = []
        with tf.variable_scope('features', reuse=reuse):
            with tf.variable_scope('dense_1'):
                x_r = tf.layers.dense(x_r, 32, activation=tf.nn.relu)
                layers.append(x_r)
            return layers

    def __init__(self, batch_size, R_HISTORY=1, N_HIDDEN=1024, INPUT_DIM=9, OUTPUT_DIM=1, save_path=None, sess=None):
        self.batch_size = tf.constant(batch_size,dtype=tf.int32)
        # r_depth, batch, input_dim
        self.data   = tf.placeholder('float',shape=[R_HISTORY*batch_size,INPUT_DIM],name='input_data_seq')
        self.imgdata = tf.placeholder('float', shape=[R_HISTORY*batch_size, 3072],name='image_feats') # Super fragile...
        # batch, output_dim
        self.labels  = tf.placeholder('float',shape=[R_HISTORY*batch_size],name='output_error')
        self.lr = tf.placeholder('float',shape=[1],name='lr')
        self.state = tf.placeholder('float', shape=[N_HIDDEN],name='cell_state')

        self.layers = []
        self.features = []
        
        sub_batches = R_HISTORY*np.ones(batch_size)

        for k in range(R_HISTORY):
            data = tf.concat([self.data[batch_size*k:batch_size*(k+1),:], self.imgdata[batch_size*k:batch_size*(k+1),:]],axis=1)
            r = (k!=0)
            self.features.append(self.extract_features(data, reuse=r)[-1])
        
        with tf.variable_scope('core_lstm'):
            with tf.variable_scope('core_0'):
                x_r = self.features
                self.layers.append(x_r)
            with tf.variable_scope('core_1'):
#                lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(N_HIDDEN),tf.contrib.rnn.BasicLSTMCell(N_HIDDEN),tf.contrib.rnn.BasicLSTMCell(N_HIDDEN)])
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN)
                x_r, states = tf.contrib.rnn.static_rnn(lstm_cell, x_r, dtype=tf.float32, sequence_length=sub_batches)
                self.layers.append(x_r[-1])
                x_r = tf.concat(x_r, axis=0)                
            with tf.variable_scope('core_end'):
                W = tf.get_variable("weights", shape=[N_HIDDEN,OUTPUT_DIM])
                b = tf.get_variable("bias", shape=[OUTPUT_DIM])
                x_r = tf.matmul(x_r, W) + b
                self.layers.append(x_r)
        self.states = states
        self.prediction = tf.reshape(x_r, [-1, OUTPUT_DIM])

        if save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
        # Needs a bit of work still
        self.loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.labels[-1], self.prediction[-1])))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr[0]).minimize(self.loss)