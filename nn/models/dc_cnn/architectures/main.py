import sys

import tensorflow as tf
import numpy as np
import collections


sys.path.insert(0, 'lib/')
from architecture import architecture_base

Results = collections.namedtuple('Results', ('output'))

sys.path.insert(0, 'models/lib')
import variables

class architecture(architecture_base):
    def __init__(self):
        self.hparams = tf.contrib.training.HParams(
          decay_rate=0.9,
          decay_steps=10000.,
          learning_rate=1.e-5, # 0.001
          maximum_learning_rate = 1.e-8, # 1.e-7
        )

    def __test_config__(self):
        pass

    def build(self, input_images):
        self.__test_config__()
        #input images = [batch, height,width] BUT COMPLEX TYPE!
        its = input_images.get_shape().as_list() # [batch, features]
        input_images = tf.squeeze(tf.reshape(input_images, [its[0], its[1]*its[2], 1]), axis=2)
        print("input size")
        print(its)
        layer1 = tf.layers.dense(input_images, 1024, activation=tf.nn.relu,name='0',reuse=tf.AUTO_REUSE)
        print("layer")
        print(layer1.get_shape().as_list())
        layer1 = tf.layers.dense(layer1, 768, activation=tf.nn.relu,name='1',reuse=tf.AUTO_REUSE)
        layer1 = tf.layers.dense(layer1, 512, activation=tf.nn.relu,name='2',reuse=tf.AUTO_REUSE)
        layer1 = tf.layers.dense(layer1, 256, activation=tf.nn.relu,name='3',reuse=tf.AUTO_REUSE)
        layer1 = tf.layers.dense(layer1, 256, activation=tf.nn.relu,name='4',reuse=tf.AUTO_REUSE)
        layer1 = tf.layers.dense(layer1, 256, activation=tf.nn.relu,name='5',reuse=tf.AUTO_REUSE)
        layer1 = tf.layers.dense(layer1, 128, activation=tf.nn.relu,name='6',reuse=tf.AUTO_REUSE)
        layer1 = tf.transpose(tf.expand_dims(layer1, axis=2), [0,2,1])
        W_1 = variables.weight_variable([1, 128, 2])
        W_1 = tf.tile(W_1, [its[0], 1, 1])
        b_1 = variables.bias_variable([1])
        layer2 = tf.add(tf.matmul(layer1,W_1), b_1)
        print("final")
        print(layer2.get_shape().as_list())
        layer2 = tf.reshape(layer2, [its[0], 1, 2])

        result = Results(layer2)
        print(">>>>> Graph Built!")

        with tf.device('/cpu:0'):
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False) # [] [unfinished]
            global_step = tf.add(global_step, 1)
        return result



    def loss_func(self, input_images, ground_truth, validation_input_images, validation_ground_truth, extra_data, validation_extra_data):
        input_images = tf.expand_dims(input_images, axis=3)
        #input_images = tf.expand_dims(extra_data["duplicate"], axis=3)
        ground_truth = tf.expand_dims(ground_truth, axis=3)
        print(">>>>>>Shape")
        print(input_images.get_shape().as_list())
        validation_input_images = tf.expand_dims(validation_input_images, axis=3)
        #validation_input_images = tf.expand_dims(validation_extra_data["duplicate"], axis=3)
        validation_ground_truth = tf.expand_dims(validation_ground_truth, axis=3)
        mini_batch_size = input_images.get_shape().as_list()[0]
        validation_mini_batch_size = validation_input_images.get_shape().as_list()[0]


        input_images = tf.squeeze(input_images, axis=3)
        ground_truth = tf.squeeze(ground_truth, axis=3)
        validation_input_images = tf.squeeze(validation_input_images, axis=3)
        validation_ground_truth = tf.squeeze(validation_ground_truth, axis=3)


        print(">>>Start Building Architecture.")
        res = self.build(input_images)
        print(">>>Finished Building Architecture.")
        output = res.output

        print(">>> Run on validation set")
        validation_res = self.build(validation_input_images)
        validation_output = validation_res.output
        print(">>> Find MSE for the validation set")
        v_diff = tf.subtract(tf.cast(validation_ground_truth, tf.float32), validation_output)
        v_MSE_loss = tf.norm(v_diff)
        with tf.name_scope('validation'):
            tf.summary.scalar("validation_total_loss", tf.reduce_sum(v_MSE_loss))
        print(">>>Some Maths on result")
        print(">>>> Find Difference")
        difference = tf.subtract(tf.cast(ground_truth, tf.float32), tf.cast(res.output, tf.float32))
        print(">>>> Find Norm")
        #L2_norm = tf.norm(difference, axis=[1,2])
        #L1_norm = tf.abs(difference)
        #L2_norm = L1_norm

        accuracy_all = tf.abs(100.*tf.divide(tf.subtract(tf.cast(ground_truth, tf.float32), tf.cast(res.output, tf.float32)), tf.cast(ground_truth, tf.float32)))
        a_s = accuracy_all.get_shape().as_list()
        accuracy_1 = tf.squeeze(tf.squeeze(tf.slice(accuracy_all, [0,0,0], [a_s[0], a_s[1], 1]), axis=2), axis=1)
        accuracy_2 = tf.squeeze(tf.squeeze(tf.slice(accuracy_all, [0,0,1], [a_s[0], a_s[1], 1]), axis=2), axis=1)

        print(">>>> Find Mean of Norm")

        L2_norm = tf.norm(difference)
        batch_loss = tf.reduce_sum(L2_norm)
        difference=tf.real(difference)
        print(">>>> Find + and - loss")
        positive_loss =  tf.reduce_sum(tf.boolean_mask(difference, tf.greater(difference, 0.)))
        negative_loss =  tf.reduce_sum(tf.boolean_mask(difference, tf.less(difference, 0.)))

        print(">>>> Find Mean Loss")
        with tf.name_scope('total'):
            print(">>>>>> Add to collection")
            tf.add_to_collection('losses', batch_loss)
            print(">>>>>> Creating summary")
            tf.summary.scalar(name='batch_L2_reconstruction_cost', tensor=batch_loss)
            print(">>>> Add result to collection of loss results for this tower")
            all_losses = tf.get_collection('losses') # [] , this_tower_scope) # list of tensors returned
            total_loss = tf.add_n(all_losses) # element-wise addition of the list of tensors
            #print(total_loss.get_shape().as_list())
            tf.summary.scalar('total_loss', total_loss)
        print(">>>> Add results to output")
        with tf.name_scope('accuracy'):
            tf.summary.scalar('positive_loss', positive_loss)
            tf.summary.scalar('negative_loss', tf.multiply(negative_loss, -1.))
            #tf.summary.scalar('estimate_accuracy', accuracy_1)
            #tf.summary.scalar('moe_accuracy', accuracy_2)


        diagnostics = {'positive_loss':positive_loss, 'negative_loss':negative_loss, 'total_loss':total_loss, 'mse': L2_norm, 'accuracy': accuracy_1, 'moe_acc': accuracy_2}
        return output, batch_loss, diagnostics, [] #, [tf.get_variable('ConvCaps1/squash/weights')]
