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
        self.patch_size = patch_size
        self.g_size = g_size
        self.l_size = l_size
        self.out_size = out_size

    def __call__(self, images, locs):

        # -- patches: [batch_size, patch_size, patch_size, 3]
        patches = self.retina(images, locs)
        patches = tf.reshape(patches,
                             [tf.shape(patches)[0],
                              self.patch_size**2*3])

        # TODO: Lewis implements Glimpse Network here

        return tf.layers.dense(patches, self.out_size)


class LocationNetwork(object):

    def __init__(self, loc_dim, std=0.2, sampling=False):
        self.loc_dim = loc_dim
        self.std = std
        self.sampling = sampling

    def __call__(self, rnn_output):

        # TODO: Lewis implements Location Network here
        # tanh to force [-1, 1] values
        means = tf.layers.dense(rnn_output, self.loc_dim,
                                activation=tf.tanh)

        if self.sampling:
            # clip to force [-1, 1] values
            locs = tf.clip_by_value(
                means + tf.random_normal(
                    [tf.shape(rnn_output)[0], self.loc_dim],
                    stddev=self.std),
                -1., 1.)
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
        self.num_glimpses = num_glimpses
        self.loc_means = tf.constant(0.0, shape=[1, self.loc_net.loc_dim])

    @property
    def batch_size(self):
        return tf.shape(self.images)[0]

    @property
    def output_size(self):
        return self.rnn_cell.output_size

    @property
    def output_dtype(self):
        return tf.float32

    def initialize(self):
        # TODO: context network for initial locs?
        # a la Multiple Object Recognition paper
        init_locs = tf.random_uniform(
            [self.batch_size, self.loc_net.loc_dim],
            minval=-1., maxval=1.)
        self.locs = init_locs
        # -- init_glimpse: [batch_size, glimpse_out_size]
        init_glimpses = self.glimpse_net(self.images, init_locs)
        finished = tf.cast([False]*self.batch_size, tf.bool)
        # -- init_state: [batch_size, rnn_cell_size]
        init_state = self.rnn_cell.zero_state(self.batch_size, tf.float32)
        return finished, init_glimpses, init_state

    def step(self, time, inputs, state):
        # run rnn_cell
        with tf.variable_scope('core_network'):
            outputs, new_state = self.rnn_cell(inputs, state)

        # get next input (glimpses), based on previous rnn state
        c, h = state  # c = state, h = output; see state_is_tuple
        locs, loc_means = self.loc_net(c)
        self.locs = tf.concat([self.locs, locs], 0)
        # self.loc_means.append(loc_means)
        glimpses = self.glimpse_net(self.images, locs)

        # finished iff have taken the right number of glimpses
        done = (time + 1 >= self.num_glimpses)
        finished = tf.reduce_all(done)

        return outputs, new_state, glimpses, finished
    """
    core_decoder = GlimpseDecoder(glimpse_net, location_net, rnn_cell,
                                      images, params['num_glimpses'])
    # -- outputs: [batch_size, num_glimpses, core_size]
    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
        core_decoder, scope='decoder')
    last_outputs = outputs[:, -1, :]
    """


def ram_model_fn(features, labels, mode, params):

    images = features[params['img_feature_name']]
    batch_size = tf.shape(images)[0]

    glimpse_net = GlimpseNetwork(params['img_size'], params['patch_size'],
                                 params['g_size'], params['l_size'],
                                 params['glimpse_out_size'])

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    location_net = LocationNetwork(params['loc_dim'], sampling=is_training)

    # TODO: convert the for loop below into tf.while_loop?
    # I had been using tf.contrib.seq2seq.dynamic_decode, but because of how it
    # hides the tf.while_loop, it makes it hard to extract information that's
    # computed during the while_loop, e.g. the sampled locations
    # See the code commented out below GlimpseDecoder above for how to use it

    # TODO: parameterize rnn_cell, i.e. implement split cell from paper
    rnn_cell = tf.nn.rnn_cell.LSTMCell(params['core_size'])

    # locs, loc_means, outputs = [], [], []
    # -- locs: [None, loc_dim]
    locs = tf.random_uniform(
        [batch_size, params['loc_dim']],
        minval=-1., maxval=1.)
    # -- loc_means: [None, loc_dim]
    loc_means = tf.zeros([batch_size, params['loc_dim']])

    with tf.variable_scope('glimpse_network', reuse=tf.AUTO_REUSE):
        glimpses = glimpse_net(images, locs)
    with tf.variable_scope('core_network', reuse=tf.AUTO_REUSE):
        state = rnn_cell.zero_state(batch_size, tf.float32)

    def cond(t, *args):
        return tf.less(t, params['num_glimpses'])

    def body(t, glimpses, state, locs, loc_means):
        with tf.variable_scope('core_network', reuse=tf.AUTO_REUSE):
            output, new_state = rnn_cell(glimpses, state)
        c, h = new_state
        with tf.variable_scope('location_network', reuse=tf.AUTO_REUSE):
            cur_locs, cur_loc_means = location_net(h)
            # store new values
            locs = tf.concat([locs, cur_locs], axis=0)
            loc_means = tf.concat([loc_means, cur_loc_means], axis=0)
        with tf.variable_scope('glimpse_network', reuse=tf.AUTO_REUSE):
            new_glimpse = glimpse_net(images, cur_locs)
        return t+1, new_glimpse, new_state, locs, loc_means

    time = tf.constant(0.0)
    times, glimpses, state, locs, loc_means = tf.while_loop(
        cond,
        body,
        [time, glimpses, state, locs, loc_means])

    """
    for _ in range(params['num_glimpses']):
        with tf.variable_scope('core_network', reuse=tf.AUTO_REUSE):
            output, state = rnn_cell(glimpses, state)
        outputs.append(output)
        c, h = state  # c = state, h = output; see state_is_tuple
        with tf.variable_scope('location_network', reuse=tf.AUTO_REUSE):
            cur_locs, cur_loc_means = location_net(h)
        locs.append(cur_locs)
        loc_means.append(cur_loc_means)
        with tf.variable_scope('glimpse_network', reuse=tf.AUTO_REUSE):
            glimpses = glimpse_net(images, cur_locs)
    """
    # classification
    # -- last_outputs: [batch_size, core_size]
    last_states, last_outputs = state.c, state.h
    # last_outputs = outputs[-1]
    last_outputs = tf.Print(last_outputs, [last_outputs[0]], summarize=64)
    with tf.variable_scope('action_network'):
        logits = tf.layers.dense(last_outputs, params['num_classes'])

    # `prediction` mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        # collect outputs here
        loc_shape = [batch_size, params['loc_dim']*(1+params['num_glimpses'])]
        outputs = {
            'logits': logits,
            'locs': tf.reshape(locs, loc_shape),
            'loc_means': tf.reshape(loc_means, loc_shape),
        }
        return tf.estimator.EstimatorSpec(mode, predictions=outputs)

    # training
    if mode == tf.estimator.ModeKeys.TRAIN:
        variables = tf.trainable_variables()
        loc_net_vars = [var for var in variables if 'location_network' in
                        var.name]
        core_net_vars = [var for var in variables if var not in loc_net_vars]

        # classification loss
        class_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=logits))
        # TODO: hybrid loss for core as well?
        core_gradients = tf.gradients(class_loss, core_net_vars)
        core_gradients, _ = tf.clip_by_global_norm(core_gradients,
                                                   params['max_grad_norm'])

        # TODO: reinforce loss for location

        # TODO: parameterize optimizer
        optimizer = tf.train.AdamOptimizer()
        grads_and_vars = []
        grads_and_vars.extend(zip(core_gradients, core_net_vars))
        train_op = optimizer.apply_gradients(
            grads_and_vars,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=class_loss,
                                          train_op=train_op)
