from datetime import datetime
import os
import random
import sys
import time

import tensorflow as tf
import numpy as np

import vgg16
from VOC2012 import *

class Config():
	batch_size = 64
	test_size = 5000
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
	load_filename = "vgg16" + '-' + "-1"
	save_filename = "vgg16"

	checkpoint_iters = 100
	summary_iters = 10
	validate_iters = 20

def one_hot(batch_y, num_classes):
	y_ = np.zeros((batch_y.shape[0], num_classes))
	y_[np.arange(batch_y.shape[0]), batch_y] = 1
	return y_

def training(learn_rate = 0.01, num_epochs = 1000, save_model = False, debug = False):
	# assert len(train_x.shape) == 4
	# [num_images, img_height, img_width, num_channel] = train_x.shape
	# num_classes = labels.shape[-1]

	config = Config()
	# config.num_classes = num_classes

	num_steps = int(np.ceil(config.num_images / float(config.batch_size)))


	with tf.Graph().as_default():

		model = vgg16.VGG16(config)

		voc2012 = VOC2012('../data', config.batch_size, config.test_size)

		predicts = model.building(True)

		# loss function
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = model.labels, logits = predicts)
		loss = tf.reduce_mean(cross_entropy)
		model.loss_summary(loss)

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

			epoch_loss = 0.0

			for epoch in range(num_epochs):
				for step in range(num_steps):
					with tf.device("/cpu:0"):
						imgs, labels = voc2012.train.next_batch(config.batch_size)
					labels = one_hot(labels, 20)

					feed_dict = {
						model.imgs: imgs,
						model.labels: labels
						}
					with tf.device(config.gpu):
						_, l, predictions = sess.run([optimizer, loss, predicts_result], feed_dict = feed_dict)
					print "batch loss: " , l

					epoch_loss += l
				
				print "each epoch Loss: %0.6f" % (epoch_loss / num_steps)

				with tf.device("/cpu:0"):
					# write summary
					if epoch % config.summary_iters == 0:
						tmp_global_step = model.global_step.eval()
						summary = sess.run(merged, feed_dict=feed_dict)
						writer.add_summary(summary, tmp_global_step)
					# save checkpoint
					if epoch % config.checkpoint_iters == 0:
						tmp_global_step = model.global_step.eval()
						model.save(sess, saver, config.save_filename, tmp_global_step)

			test_loss = 0.0
			test_accuracy = 0.0
			for i in range(4):
				
				with tf.device("/cpu:0"):
					valid_x, valid_y = voc2012.test.next_batch(config.test_size)
				valid_y = one_hot(test_labels, 20)
				
				if valid_x is not None and valid_y is not None:
					feed_dict = {
						model.imgs: valid_x,
						model.labels: valid_y
						}

					l, predictions = sess.run([loss, predicts_result], feed_dict = feed_dict)
					test_loss += l
					test_accuracy += predictions
			print('Valid Loss = %.6f\t Accuracy = %.6f%%' % (test_loss/4, accuracy(predictions, valid_y)/4))

# predictions/labels is a 2-D matrix [num_images, num_classes]
def accuracy(predictions, labels):
	return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

if __name__ == "__main__":
	# batch = 64
	# voc2012 = VOC2012('../data', batch, batch)
	
	# # imgs shape  (128, 224, 224, 3)
	# # labels shape (128, 20)
	# imgs, labels = voc2012.train.next_batch(batch)
	# labels = one_hot(labels, 20)
	
	# test_imgs, test_labels = voc2012.test.next_batch(batch)
	# test_labels = one_hot(test_labels, 20)

	# training(imgs, labels, test_imgs, test_labels)
	training()