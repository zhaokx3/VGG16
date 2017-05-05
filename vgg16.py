########################################################################################
#zhaokx3, 2017                                                                  #
# VGG16 implementation in TensorFlow                                                   #

# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
# from scipy.misc import imread, imresize
# from imagenet_classes import class_names


class VGG16:
    def __init__(self, config):
        self.global_step = tf.get_variable("global_step", initializer=0,
                    dtype=tf.int32, trainable=False)
        self.wd = config.wd
        self.stddev = config.stddev
        self.img_height = config.img_height
        self.img_width = config.img_width
        self.num_channel = config.num_channel
        self.num_classes = config.num_classes
        self.moving_average_decay = config.moving_average_decay
        self.params_dir = config.params_dir

        self.imgs = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.num_channel])
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes])
        self.parameters = []
        # self.probs = tf.nn.softmax(self.fc3l)
        # if weights is not None and sess is not None:
        #     self.load_weights(weights, sess)

    def building(self, is_Train):
        # images = pre_process()
        out_fc = self.cnn_fc(self.imgs, self.num_classes, is_Train, 'fc')
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(out_fc, self.labels)
        return out_fc

    def pre_process(self):
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean
        return images

    def cnn_fc(self, input_, num_classes, is_Train, name):
        trainable = is_Train

        with tf.variable_scope(name) as scope:
            
            # assume the input image shape is 224 x 224 x 3
            
            conv1_1, kernel1_1, bias1_1  = self.conv_layer('conv1_1', input_, 64, 3, trainable)
            self.parameters += [kernel1_1, bias1_1]
            
            conv1_2, kernel1_2, bias1_2  = self.conv_layer('conv1_2', conv1_1, 64, 3, trainable)
            self.parameters += [kernel1_2, bias1_2]
            
            pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
            # pool1 shape: 112 x 112 x 64

            conv2_1, kernel2_1, bias2_1  = self.conv_layer('conv2_1', pool1, 128, 3, trainable)
            self.parameters += [kernel2_1, bias2_1]
            
            conv2_2, kernel2_2, bias2_2  = self.conv_layer('conv2_2', conv2_1, 128, 3, trainable)
            self.parameters += [kernel2_2, bias2_2]
            
            pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
            # pool2 shape: 56 x 56 x 128

            conv3_1, kernel3_1, bias3_1  = self.conv_layer('conv3_1', pool2, 256, 3, trainable)
            self.parameters += [kernel3_1, bias3_1]
            
            conv3_2, kernel3_2, bias3_2  = self.conv_layer('conv3_2', conv3_1, 256, 3, trainable)
            self.parameters += [kernel3_2, bias3_2]
            
            conv3_3, kernel3_3, bias3_3  = self.conv_layer('conv3_3', conv3_2, 256, 3, trainable)
            self.parameters += [kernel3_3, bias3_3]
            
            pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')
            # pool3 shape: 28 x 28 x 256

            conv4_1, kernel4_1, bias4_1  = self.conv_layer('conv4_1', pool3, 512, 3, trainable)
            self.parameters += [kernel4_1, bias4_1]
            
            conv4_2, kernel4_2, bias4_2  = self.conv_layer('conv4_2', conv4_1, 512, 3, trainable)
            self.parameters += [kernel4_2, bias4_2]
            
            conv4_3, kernel4_3, bias4_3  = self.conv_layer('conv4_3', conv4_2, 512, 3, trainable)
            self.parameters += [kernel4_3, bias4_3]
            
            pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
            # pool4 shape: 14 x 14 x 512

            conv5_1, kernel5_1, bias5_1  = self.conv_layer('conv5_1', pool4, 512, 3, trainable)
            self.parameters += [kernel5_1, bias5_1]
            
            conv5_2, kernel5_2, bias5_2  = self.conv_layer('conv5_2', conv5_1, 512, 3, trainable)
            self.parameters += [kernel5_2, bias5_2]
            
            conv5_3, kernel5_3, bias5_3  = self.conv_layer('conv5_3', conv5_2, 512, 3, trainable)
            self.parameters += [kernel5_3, bias5_3]
            
            pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')
            if (is_Train):
                pool5 = tf.nn.dropout(pool5, keep_prob = 0.5)
            # pool5 shape: 7 x 7 x 512

            fc1, fc1W, fc1b = self.fc_layer('fc1', pool5, 4096, trainable)
            self.parameters += [fc1W, fc1b]
            
            fc2, fc2W, fc2b = self.fc_layer('fc2', fc1, 4096, trainable)
            self.parameters += [fc2W, fc2b]
            
            fc3, fc3W, fc3b = self.fc_layer('fc3', fc2, num_classes, trainable)
            self.parameters += [fc3W, fc3b]

        return fc3

    def conv_layer(self, name, input_, output_channel, kernel_size, trainable):
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope + "Weights",
                shape = [kernel_size, kernel_size, input_.get_shape()[-1].value, output_channel],
                dtype = tf.float32,
                initializer = tf.contrib.layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_, kernel, [1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(tf.constant(0.0, shape=[output_channel], dtype=tf.float32), trainable, name='bias')
            output = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)
            self._activation_summary(output)
        return output, kernel, bias       

    def fc_layer(self, name, input_, output_num, trainable):
        shape = input_.get_shape()
        if len(shape) == 4:
            size = shape[1].value * shape[2].value * shape[3].value
        else:
            size = shape[-1].value
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope+'Weights',
                shape=[size, output_num],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.Variable(tf.constant(0.1, shape=[output_num], dtype=tf.float32), trainable, name='bias')
            flat = tf.reshape(input_, [-1, size])
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat, kernel), bias))
            self._activation_summary(output)
        return output, kernel, bias

    def _activation_summary(self, x):
        name = x.op.name
        tf.summary.histogram(name + '/activations', x)
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))
        tf.summary.scalar(name + '/max', tf.reduce_max(x))
        tf.summary.scalar(name + '/min', tf.reduce_min(x))
        tf.summary.scalar(name + '/mean', tf.reduce_mean(x))

    def loss_summary(self, loss):
        tf.summary.scalar(loss.op.name, loss)

    def save(self, sess, saver, filename, global_step):
        path = saver.save(sess, self.params_dir+filename, global_step=global_step)
        print "Save params at " + path

    def restore(self, sess, saver, filename):
        print "Restore from previous model: ", self.params_dir+filename
        saver.restore(sess, self.params_dir+filename)