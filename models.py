"""
Copyright (c) 2018 Shane Steinert-Threlkeld

    *****
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    *****
"""
import tensorflow as tf


def ffnn_model_fn(features, labels, mode, params):

    # -- features: [batch_size, width, height, channels]
    # -- images: [batch_size, width*height*channels]
    # input_layer reshapes behind the scenes
    images = tf.feature_column.input_layer(features, params['feature_columns'])

    net = images
    training = mode == tf.estimator.ModeKeys.TRAIN
    for layer in params['layers']:
        net = tf.layers.dense(net,
                              units=layer['units'],
                              activation=layer['activation'])
        if layer['dropout']:
            net = tf.layers.dropout(net,
                                    rate=layer['dropout'],
                                    training=training)
    # -- net: [batch_size, params['layers'][-1]['units']]

    # -- logits: [batch_size, num_classes]
    logits = tf.layers.dense(net, units=params['num_classes'], activation=None)

    # prediction
    # -- predicted_classes: [batch_size]
    predicted_classes = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # loss and training
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    # TODO: parameterize optimizer?
    optimizer = tf.train.RMSPropOptimizer(0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes)
    metrics = {'total_accuracy': accuracy}
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics)


def cnn_model_fn(features, labels, mode, params):

    images = features[params['img_feature_name']]
    net = images

    # TODO: Lewis will implement CNN here

    return


################################################################
# Recurrent Attention Model
# From Mnih et al 2014, Recurrent Models of Visual Attention
# TODO: document!
################################################################


class Retina(object):

    def __init__(self, img_size, patch_size, num_patches=1):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(self, images, locs):
        # TODO: support multiple patches
        return tf.image.extract_glimpse(
            images,
            [self.patch_size, self.patch_size],
            locs)


class GlimpseNetwork(object):

    def __init__(self, img_size, patch_size, g_size, l_size, out_size):
        self.retina = Retina(img_size, patch_size)
        self.g_size = g_size
        self.l_size = l_size
        self.out_size = out_size

    def __call__(self, images, locs):

        # -- patches: [batch_size, patch_size, patch_size, 3]
        patches = self.retina(images, locs)

        # TODO: Lewis implements Glimpse Network here

        return tf.Variable(tf.zeros([tf.shape(images)[0], self.out_size]),
                           name="dummy_glimpse")


class LocationNetwork(object):

    def __init__(self, loc_dim, std=0.2, sampling=False):
        self.loc_dim = loc_dim
        self.std = std
        self.sampling = sampling

    def __call__(self, rnn_output):

        # TODO: Lewis implements Location Network here
        # tanh to force [-1, 1] values
        means = tf.tanh(
            tf.Variable(tf.zeros([tf.shape(rnn_output)[0], self.loc_dim]),
                        name="dummy_loc_mean"))

        if self.sampling:
            # tanh to force [-1, 1] values
            locs = tf.tanh(
                means + tf.random_normal(
                    [tf.shape(rnn_output)[0], self.loc_dim],
                    std=self.std))
        else:
            locs = means

        return locs, means


class GlimpseDecoder(tf.contrib.seq2seq.Decoder):

    def __init__(self, glimpse_network, location_network, rnn_cell, images,
                 num_glimpses):
        self.glimpse_net = glimpse_network
        self.loc_net = location_network
        self.rnn_cell = rnn_cell
        self.images = images
        self.batch_size = tf.shape(images)[0]
        self.num_glimpses = num_glimpses
        self.locs = []
        self.loc_means = []

    def initialize(self):
        # TODO: context network for initial locs?
        # a la Multiple Object Recognition paper
        init_locs = tf.random_uniform(
            [self.batch_size, self.loc_net.loc_dim],
            minval=-1., maxval=1.)
        self.locs.append(init_locs)
        # -- init_glimpse: [batch_size, glimpse_out_size]
        init_glimpses = self.glimpse_net(self.images, init_locs)
        finished = [False]*self.batch_size
        # -- init_state: [batch_size, rnn_cell_size]
        init_state = self.rnn_cell.zero_state(self.batch_size)
        return finished, init_glimpses, init_state

    def step(self, time, inputs, state):
        # run rnn_cell
        outputs, new_state = self.rnn_cell(inputs, state)

        # get next input (glimpses), based on previous rnn state
        locs, loc_means = self.loc_net(state)
        self.locs.append(locs)
        self.loc_means.append(loc_means)
        glimpses = self.glimpse_net(self.images, locs)

        # finished iff have taken the right number of glimpses
        done = time == self.num_glimpses - 1
        finished = [done]*self.batch_size

        return outputs, new_state, glimpses, finished


def ram_model_fn(features, labels, mode, params):

    images = features[params['img_feature_name']]

    with tf.variable_scope('glimpse_network'):
        glimpse_net = GlimpseNetwork(params['img_size'], params['patch_size'],
                                     params['g_size'], params['l_size'],
                                     params['glimpse_out_size'])

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('location_network'):
        location_net = LocationNetwork(params['loc_dim'], sampling=is_training)

    with tf.variable_scope('core_network'):
        # TODO: parameterize rnn_cell, i.e. implement split cell from paper
        rnn_cell = tf.nn.rnn_cell.LSTMCell(params['core_size'])
        core_decoder = GlimpseDecoder(glimpse_net, location_net, rnn_cell,
                                      images)

    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(core_decoder)

    # classification
    last_outputs = outputs[-1]
    with tf.variable_scope('action_network'):
        logits = tf.layers.dense(last_outputs, params['num_classes'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        # collect outputs here
        outputs = {
            'logits': logits,
            'locs': core_decoder.locs
        }
        return tf.estimator.EstimatorSpec(mode, predictions=outputs)
