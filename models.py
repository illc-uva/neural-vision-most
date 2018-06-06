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
def ram_model_fn(features, labels, mode, params):

    images = features[params['img_feature_name']]
    batch_size = tf.shape(images)[0]
    tf.summary.image('images', images, max_outputs=8)

    # define network components 
    # NOTE: no actual tf Variables are initialized yet!

    # retina
    def retina(images, locs, scope='retina'):
        # TODO: more than one resolution!
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            return tf.image.extract_glimpse(
                images,
                [params['patch_size'], params['patch_size']],
                locs)

    # glimpse network
    def glimpse_network(images, locs, scope='glimpse_network'):
        # -- patches: [batch_size, patch_size, patch_size, 3]
        patches = retina(images, locs)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            patches = tf.reshape(patches,
                                 [tf.shape(patches)[0],
                                  params['patch_size']**2*3])

            # TODO: Lewis implements Glimpse Network here, using also
            # params['g_size'] and params['l_size']
            return tf.layers.dense(patches, params['glimpse_out_size'])

    # location_network
    # TODO: make std a param, or learnable?
    def location_network(rnn_output, sampling, std=params['std'],
                         scope='location_network'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # TODO: Lewis implements Location Network here
            # tanh to force [-1, 1] values
            means = tf.layers.dense(rnn_output, params['loc_dim'],
                                    activation=tf.tanh)

            if sampling:
                # clip to force [-1, 1] values
                locs = tf.clip_by_value(
                    means + tf.random_normal(
                        [tf.shape(rnn_output)[0], params['loc_dim']],
                        stddev=std),
                    -1., 1.)
            else:
                locs = means

            return locs, means

    # core network
    # TODO: parameterize rnn_cell, i.e. implement split cell from paper
    rnn_cell = tf.nn.rnn_cell.LSTMCell(params['core_size'])

    # get initial location, glimpses, and state

    # TODO: initial loc as a separate network?
    # -- locs: [None, loc_dim]
    locs = tf.random_uniform(
        [batch_size, params['loc_dim']],
        minval=-1., maxval=1.)
    # -- loc_means: [None, loc_dim]
    loc_means = tf.zeros([batch_size, params['loc_dim']])
    glimpse = glimpse_network(images, locs)
    state = rnn_cell.zero_state(batch_size, tf.float32)

    # set up the main loop, for sampling locations and extracting glimpses
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # NOTE: the use of a tf.while_loop allows extracting information by passing
    # through and concatenating to a tensor at each iteration.
    # Or, as I'm doing now, by writing to a TensorArray
    # I had been using tf.contrib.seq2seq.dynamic_decode, but because of how it
    # hides the tf.while_loop, it makes it hard to extract information that's
    # computed during the while_loop, e.g. the sampled locations
    def initialize_ta(N, init_tensor):
        ta = tf.TensorArray(tf.float32, N)
        ta = ta.write(0, init_tensor)
        return ta
    locs_ta = initialize_ta(1+params['num_glimpses'], locs)
    loc_means_ta = initialize_ta(1+params['num_glimpses'], loc_means)
    outputs_ta = tf.TensorArray(tf.float32, params['num_glimpses'])

    def cond(t, *args):
        return tf.less(t, params['num_glimpses'])

    def body(t, glimpse, state, outputs_ta, locs_ta, loc_means_ta):
        # run the core network
        with tf.variable_scope('core_network', reuse=tf.AUTO_REUSE):
            output, new_state = rnn_cell(glimpse, state)
            outputs_ta = outputs_ta.write(t, output)

        # get new location
        cur_locs, cur_loc_means = location_network(output, is_training)
        # store new values
        # TODO: can this be done better with TensorArrays?
        locs_ta = locs_ta.write(t+1, cur_locs)
        loc_means_ta = loc_means_ta.write(t+1, cur_loc_means)

        # get next glimpse
        new_glimpse = glimpse_network(images, cur_locs)
        return t+1, new_glimpse, new_state, outputs_ta, locs_ta, loc_means_ta

    # THE MAIN LOOP!
    time = tf.constant(0)
    time, glimpse, state, outputs_ta, locs_ta, loc_means_ta = (
        tf.while_loop(
            cond,
            body,
            [time, glimpse, state, outputs_ta, locs_ta, loc_means_ta]))

    def ta_to_batch_major(ta):
        # turns a TensorArray of time_steps length with [batch_size, dim]
        # entries at each step into a Tensor [batch_size, time_steps, dim]
        tensor = ta.stack()
        return tf.transpose(tensor, perm=[1, 0, 2])

    # [batch_size, 1+num_glimpses, loc_dim]
    locs = ta_to_batch_major(locs_ta)
    # [batch_size, 1+num_glimpses, loc_dim]
    loc_means = ta_to_batch_major(loc_means_ta)
    # [batch_size, num_glimpses, core_size]
    outputs = ta_to_batch_major(outputs_ta)

    # classification
    # -- last_outputs: [batch_size, core_size]
    _, last_outputs = state
    with tf.variable_scope('action_network', reuse=tf.AUTO_REUSE):
        logits = tf.layers.dense(last_outputs, params['num_classes'])
        predicted_classes = tf.argmax(logits, axis=1)
        predicted_classes = tf.cast(predicted_classes, tf.int32)

    # `prediction` mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        # collect outputs here
        outputs = {
            'logits': logits,
            'classes': predicted_classes,
            'locs': locs,
            'loc_means': loc_means,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=outputs)

    # losses
    with tf.variable_scope('losses'):
        # classification loss
        class_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))

        # reinforce loss for location
        # ignore first step, since that's a random loc choice
        # -- reshaped: [batch_size, num_glimpses, loc_dim]
        locs = locs[:, 1:, :]
        loc_means = loc_means[:, 1:, :]

        def log_likelihood(means, locs, std):
            dist = tf.distributions.Normal(means, params['std'])
            # -- logll: [batch_size, num_glimpses, loc_dim]
            logll = dist.log_prob(locs)
            # -- logll: [batch_size, num_glimpses]
            logll = tf.reduce_sum(logll, 2)
            return logll

        # log_prob(x_t | s_1:t-1)
        logll = log_likelihood(loc_means, locs,
                               params['std'])
        # reward: 1 for correct class, 0 for incorrect
        reward = tf.to_float(tf.equal(predicted_classes, labels))
        # reward: [batch_size, 1]
        reward = tf.expand_dims(reward, 1)
        # reward: [batch_size, num_glimpses]
        # NB: 1/0 at each time step is `cumulative' reward
        reward = tf.tile(reward, [1, params['num_glimpses']])
        # normalize reward
        # r_mean, r_std = tf.nn.moments(reward, axes=[0, 1])
        # reward = (reward - r_mean) / (r_std + 1e-10)
        # TODO: baseline for variance reduction
        # NB: requires getting all rnn_outputs from while_loop, not just the
        # final one
        adv = reward
        reinforce_loss = -tf.reduce_mean(logll * adv)

    # evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes)
        metrics = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=class_loss,
                                          eval_metric_ops=metrics)

    # training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        variables = tf.trainable_variables()
        loc_net_vars = [var for var in variables if 'location_network' in
                        var.name]
        core_net_vars = [var for var in variables if var not in loc_net_vars]
        # TODO: hybrid loss for core as well?
        core_gradients = tf.gradients(class_loss, core_net_vars)
        core_gradients, _ = tf.clip_by_global_norm(core_gradients,
                                                   params['max_grad_norm'])
        loc_gradients = tf.gradients(reinforce_loss, loc_net_vars)
        loc_gradients, _ = tf.clip_by_global_norm(loc_gradients,
                                                  params['max_grad_norm'])


        # TODO: parameterize optimizer
        optimizer = tf.train.AdamOptimizer()
        grads_and_vars = []
        grads_and_vars.extend(zip(core_gradients, core_net_vars))
        grads_and_vars.extend(zip(loc_gradients, loc_net_vars))
        train_op = optimizer.apply_gradients(
            grads_and_vars,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=class_loss,
                                          train_op=train_op)
