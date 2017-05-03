from datetime import datetime
import os
import random
import sys
import time

import tensorflow as tf
import numpy as np

import vgg16

class Config():
	batch_size = 1
	img_height = 224
	img_width = 224
	num_channel = 3
	num_classes = 20
	num_images = 10000
	wd = 5e-4
	stddev = 5e-2
	moving_average_decay = 0.999
	initialize = True
	gpu = '/gpu:0'

	# checkpoint path and filename
	logdir = "./log/train_log/"
	params_dir = "./params/"
	load_filename = "cpm" + '-' + steps
	save_filename = "cpm"

	checkpoint_iters = 2000
  	summary_iters = 100
  	validate_iters = 2000

def training(learn_rate = 0.01, num_epochs = 1, save_model = False, debug = False):
	# assert len(train_x.shape) == 4
	# [num_images, img_height, img_width, num_channel] = train_x.shape
    num_classes = labels.shape[-1]

    config = Config()
    config.num_classes = num_classes

    num_steps = int(np.ceil(config.num_images / float(config.batch_size)))

    with tf.Graph().as_default():

    	model = vgg16.VGG16(config)

    	predicts = model.predict_(True)

    	# loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
    	loss = tf.reduce_mean(cross_entropy)

    	# optimizer with decayed learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learn_rate, global_step, num_steps*num_epochs, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # prediction for the training data
        predicts_result = tf.nn.softmax(predicts)

        # Initializing operation
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep = 100)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
        	# initialize parameters or restore from previous model
            if not os.path.exists(config.params_dir):
                os.makedirs(config.params_dir)
            if os.listdir(config.params_dir) == [] or config.initialize:
                print "Initializing Network"
                sess.run(init_op)
            else:
                sess.run(init_op)
                model.restore(sess, saver, config.load_filename)

            merged = tf.summary.merge_all()
            logdir = os.path.join(config.logdir,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

            writer = tf.summary.FileWriter(logdir, sess.graph)

        	for epoch in range(num_epochs):
        		for step in range(num_steps):
        			
        			#########
        			with tf.device("/cpu:0"):
        				imgs, labels = get_batch()
        			##########
        			
        			feed_dict = {
        				model.imgs = imgs
        				model.labels = labels
        			}
        			with tf.device(config.gpu):
        				_, l, predictions = sess.run([optimizer, loss, predicts_result], feed_dict = feed_dict)

        			with tf.device('/cpu:0'):
                  		# write summary
                  		if (idx + 1) % config.summary_iters == 0:
                      		tmp_global_step = model.global_step.eval()
                      		summary = sess.run(merged, feed_dict=feed_dict)
                      		writer.add_summary(summary, tmp_global_step)
                  		# save checkpoint
                  		if (idx + 1) % config.checkpoint_iters == 0:
                      		tmp_global_step = model.global_step.eval()
                      		model.save(sess, saver, config.save_filename, tmp_global_step)
# predictions/labels is a 2-D matrix [num_images, num_classes]

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

if __name__ == "__main__":
	training()