import tensorflow as tf
import numpy as np
import os
import tarfile
import sys
from six.moves import urllib

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 300
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 300
LABEL_BYTES = 1
HEIGHT = 224
WIDTH = 224
CHANNEL = 3

def read_img(filename_queue):

    class ImgRecord(object):
        pass
    result = ImgRecord()

    # Dimensions of the images in the CIFAR-10 dataset.
    # input format.
    label_bytes = LABEL_BYTES
    result.height = HEIGHT
    result.width = WIDTH
    result.depth = CHANNEL
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast( \
            tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    result.uint8image = tf.reshape( \
            tf.strided_slice(record_bytes, [label_bytes], \
                    [label_bytes + image_bytes]), \
            [result.width, result.height, result.depth])
  
    return result

def _generate_image_and_label_batch( \
            image, label, min_queue_examples, \
            batch_size, shuffle):

    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
                [image, label], \
                batch_size=batch_size, \
                num_threads=num_preprocess_threads, \
                capacity=min_queue_examples + 3 * batch_size, \
                min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
                [image, label], \
                batch_size=batch_size, \
                num_threads=num_preprocess_threads, \
                capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])

class Data(object):
    def __init__(self, filenames, min_queue_examples, batch_size, shuffle):        
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        self.sess = tf.Session()

        filename_queue = tf.train.string_input_producer(filenames)
        self.read_input = read_img(filename_queue)

        self.read_input.uint8image.set_shape([HEIGHT, WIDTH, CHANNEL])
        self.read_input.label.set_shape([1])
        self.images, self.lables = _generate_image_and_label_batch(self.read_input.uint8image, \
                self.read_input.label, min_queue_examples, batch_size, shuffle)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

    def next_batch(self, batch_size):
        images, lables = self.sess.run([self.images, self.lables])
        images = images.astype(np.float64) / 256.0
        return images, lables


class VOC2012(object):
    def __init__(self, data_path, batch_size, test_batch_size):
        data_dir = os.path.join(data_path, 'processed_data')
        
        filenames_train = [os.path.join(data_dir, '%d_train.bin' % i) \
                for i in xrange(20)]
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples_train = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                min_fraction_of_examples_in_queue)
        self.train = Data(filenames_train, min_queue_examples_train, batch_size, True)

        filenames_test = [os.path.join(data_dir, '%d_val.bin' % i) \
                for i in xrange(20)]
        min_queue_examples_test = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        self.test = Data(filenames_test, min_queue_examples_test, test_batch_size, False)
