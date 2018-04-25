import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image

import models


class Predict:
    def __init__(self, model_data_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        height = 228
        width = 304
        channels = 3
        batch_size = 1

        self.input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
        self.net = models.ResNet50UpProj({'data': self.input_node}, batch_size, 1, False)
        self.sess = tf.Session()
        saver = tf.train.Saver()

        saver.restore(self.sess, model_data_path)
        print('Loading the model')
        # return sess

    def depth_predict(self, image):
        img = np.array(image).astype('float32')
        # print img.shape
        # img = np.expand_dims(np.asarray(img), axis = 0)

        # print img.shape
        pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
        # Plot result
        # fig = plt.figure()
        # ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()
        # print 'Running finished!'

        # result = Image.fromarray(((pred[0,:,:,0]*255).astype(np.uint8)))
        # result.save('res.png')
        return pred[0, :, :, 0].astype(np.float32)
