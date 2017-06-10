import tensorflow as tf


def fire_module(x,inp,sp,e11p,e33p):
    with tf.variable_scope("fire"):
        with tf.variable_scope("squeeze"):
            W = tf.get_variable("weights",shape=[1,1,inp,sp])
            b = tf.get_variable("bias",shape=[sp])
            s = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")+b
            s = tf.nn.relu(s)
        with tf.variable_scope("e11"):
            W = tf.get_variable("weights",shape=[1,1,sp,e11p])
            b = tf.get_variable("bias",shape=[e11p])
            e11 = tf.nn.conv2d(s,W,[1,1,1,1],"VALID")+b
            e11 = tf.nn.relu(e11)
        with tf.variable_scope("e33"):
            W = tf.get_variable("weights",shape=[3,3,sp,e33p])
            b = tf.get_variable("bias",shape=[e33p])
            e33 = tf.nn.conv2d(s,W,[1,1,1,1],"SAME")+b
            e33 = tf.nn.relu(e33)
        return tf.concat([e11,e33],3)


class SqueezeNet(object):
    def extract_features(self, input=None, reuse=True):
        if input is None:
            input = self.image
        x = input
        layers = []
        with tf.variable_scope('features', reuse=reuse):
            with tf.variable_scope('layer0'):
                W = tf.get_variable("weights",shape=[3,3,self.NUM_CHANNELS,64])
                b = tf.get_variable("bias",shape=[64])
                x = tf.nn.conv2d(x,W,[1,2,2,1],"VALID")
                x = tf.nn.bias_add(x,b)
                layers.append(x)
            with tf.variable_scope('layer1'):
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('layer2'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer3'):
                x = fire_module(x,64,16,64,64)
                layers.append(x)
            with tf.variable_scope('layer4'):
                x = fire_module(x,128,16,64,64)
                layers.append(x)
            with tf.variable_scope('layer5'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer6'):
                x = fire_module(x,128,32,128,128)
                layers.append(x)
            with tf.variable_scope('layer7'):
                x = fire_module(x,256,32,128,128)
                layers.append(x)
            with tf.variable_scope('layer8'):
                x = tf.nn.max_pool(x,[1,3,3,1],strides=[1,2,2,1],padding='VALID')
                layers.append(x)
            with tf.variable_scope('layer9'):
                x = fire_module(x,256,48,192,192)
                layers.append(x)
            with tf.variable_scope('layer10'):
                x = fire_module(x,384,48,192,192)
                layers.append(x)
            with tf.variable_scope('layer11'):
                x = fire_module(x,384,64,256,256)
                layers.append(x)
            with tf.variable_scope('layer12'):
                x = fire_module(x,512,64,256,256)
                layers.append(x)
        return layers

    def __init__(self, NUM_CLASSES = 1000, NUM_CHANNELS=3, img_dim=192, bin_weights=None, save_path=None, sess=None):
        """Create a SqueezeNet model.
        Inputs:
        - save_path: path to TensorFlow checkpoint
        - sess: TensorFlow session
        """
        self.image = tf.placeholder('float',shape=[None,None,None,NUM_CHANNELS],name='input_image')
        self.labels = tf.placeholder('int32', shape=[None], name='labels')
        self.layers = []
        self.lr = tf.placeholder('float',shape=[1],name='lr')
        if bin_weights is not None:
            self.bin_weights = tf.constant(bin_weights);
        x = self.image
        self.NUM_CHANNELS = int(NUM_CHANNELS)
        self.layers = self.extract_features(x, reuse=False)
        self.features = self.layers[-1]
        with tf.variable_scope('classifier'):
            with tf.variable_scope('layer0'):
                x = self.features
                self.layers.append(x)
            with tf.variable_scope('layer1'):
                if(save_path is not None):
                    W = tf.get_variable("weights",shape=[1,1,512,1000])
                    b = tf.get_variable("bias",shape=[1000])
                else:
                    W = tf.get_variable("weights",shape=[1,1,512,NUM_CLASSES])
                    b = tf.get_variable("bias",shape=[NUM_CLASSES])
                x = tf.nn.conv2d(x,W,[1,1,1,1],"VALID")
                x = tf.nn.bias_add(x,b)
                self.layers.append(x)
            with tf.variable_scope('layer2'):
                x = tf.nn.relu(x)
                self.layers.append(x)
            with tf.variable_scope('layer3'):
                if(img_dim == 192):   
                    x = tf.nn.avg_pool(x,[1,11,11,1],strides=[1,11,11,1],padding='VALID')
                elif(img_dim == 150):
                    x = tf.nn.avg_pool(x,[1,6,5,1],strides=[1,4,4,1],padding='VALID')
                elif(img_dim == 48):
                    x = tf.nn.avg_pool(x,[1,2,3,1],strides=[1,2,1,1],padding='VALID')
                else:
                    print('Untested image dim ',img_dim, 'If dimensions don\'t work, change layer3 pooling')
                    x = tf.nn.avg_pool(x,[1,13,13,1],strides=[1,13,13,1],padding='VALID')
                    
                self.layers.append(x)
            with tf.variable_scope('output_layer'):

                if(save_path is not None and NUM_CLASSES is not 1000):
                    x = tf.reshape(x,[-1,1000])
                    self.classifier = tf.layers.dense(x, NUM_CLASSES)
                    self.layers.append(x)
                else:
                    self.classifier = tf.reshape(x,[-1,NUM_CLASSES])
        if save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)

        self.prediction = tf.cast(tf.argmax(self.classifier,1),'int32')
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.labels),tf.float32))

        # Softmax loss:
        if bin_weights is None:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, NUM_CLASSES), logits=self.classifier)) + tf.reduce_mean(tf.abs(tf.squared_difference(tf.cast(self.labels,'float'), tf.cast(self.prediction,'float'))))
        else:
            y_a = tf.gather(self.bin_weights,self.labels)
            y_p = tf.gather(self.bin_weights, self.prediction)
            self.loss = tf.reduce_mean( tf.abs(tf.squared_difference(y_p, y_a) ) ) 
            #+ tf.reduce_mean( tf.sqared_difference(bin_weights[self.prediction], bin_weights[self.labels] ) ) # L2 loss
            
        # L1 loss:
#        self.loss = tf.reduce_mean((tf.abs(tf.cast(tf.one_hot(self.labels, NUM_CLASSES),'float32')-self.classifier)))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr[0]).minimize(self.loss)

