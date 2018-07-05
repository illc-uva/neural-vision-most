"""
Copyright (c) 2018 Shane Steinert-Threlkeld and Lewis O'Sullivan

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
import numpy as np


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
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes)
    metrics = {'total_accuracy': accuracy}
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics)


def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # images is the input layer
    images = features[params['img_feature_name']]
    batch_size = tf.shape(images)[0]
    # net carries forward the currrent output of the graph
    net = images

    training = mode == tf.estimator.ModeKeys.TRAIN
    # loop for adding a convolutional layer and max pooling layer pair
    for layer in params['layers']:
        for _ in range(layer['num_convs']):
            # convolutional layer
            net = tf.layers.conv2d(
                    inputs=net,
                    filters=layer['filters'],
                    kernel_size=layer['kernel_size'],
                    padding=layer['padding'],
                    activation=layer['activation'])

        # pooling layer
        net = tf.layers.max_pooling2d(
            inputs=net,
            pool_size=layer['pool_size'],
            strides=layer['pool_strides'])

    # flattening the output of last max pooling layer into a batch of vectors    
    net = tf.reshape(net, [batch_size, np.prod(net.shape[1:])])

    # Dense Layer
    for layer in params['dense']:
        net = tf.layers.dense(
                inputs=net,
                units=layer['units'],
                activation=layer['activation'])

    # Add dropout operation; 0.6 probability that element will be kept 
        net = tf.layers.dropout(
                inputs=net,
                rate=layer['rate'],
                training=training)

    # Logits layer
    logits = tf.layers.dense(inputs=net, units=params['num_classes'],
                             activation=None)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        'classes': tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


################################################################
# Recurrent Attention Model
# From Mnih et al 2014, Recurrent Models of Visual Attention
# TODO: document!
################################################################
def ram_model_fn(features, labels, mode, params):

    images = features[params['img_feature_name']]
    batch_size = tf.shape(images)[0]
    tf.summary.image('images', images, max_outputs=8)

    # set up the main loop, for sampling locations and extracting glimpses
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # define network components 
    # NOTE: no actual tf Variables are initialized yet!

    # retina
    def retina(images, locs, scope='retina'):
        # TODO: more than one resolution!
        patches = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for num in range(params['num_patches']):
                length = (params['patch_scale']**num)*params['patch_size']
                patches.append(
                    tf.image.resize_images(
                        tf.image.extract_glimpse(
                            images,
                            [length, length],
                            locs),
                        [params['patch_size'], params['patch_size']]
                    )
                )
        return tf.concat(patches, axis=1)

    # glimpse network
    if params['glimpse_type'] == 'CNN':
        def glimpse_network(images, locs, scope='glimpse_network'):
            # -- patches: [batch_size, patch_size * num_patches, patch_size, 3]
            patches = retina(images, locs)
            # TODO: dropout here?
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tf.layers.conv2d(patches, 64, 5)
                net = tf.layers.conv2d(net, 64, 3)
                net = tf.layers.conv2d(net, 128, 3)
                net = tf.reshape(net, [tf.shape(patches)[0],
                                       np.prod(net.shape[1:])])
                what = tf.layers.dense(net,
                                       units=params['glimpse_out_size'],
                                       activation=tf.nn.relu)
                where = tf.layers.dense(
                    inputs=locs,
                    units=params['glimpse_out_size'],
                    activation=tf.nn.relu)
                return what*where
    else:
        def glimpse_network(images, locs, scope='glimpse_network'):
            # -- patches: [batch_size, patch_size * num_patches, patch_size, 3]
            patches = retina(images, locs)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                patches = tf.reshape(
                    patches,
                    [tf.shape(patches)[0],
                     np.prod(patches.shape[1:])])

                hidden_sensor_layer = tf.layers.dense(
                        inputs=patches,
                        units=params['g_size'],
                        activation=tf.nn.relu)

                dense_sensor_layer = tf.layers.dense(
                        inputs=hidden_sensor_layer,
                        units=params['g_size'],
                        activation=None)

                hidden_location_layer = tf.layers.dense(
                        inputs=locs,
                        units=params['l_size'],
                        activation=tf.nn.relu)

                dense_location_layer = tf.layers.dense(
                        inputs=hidden_location_layer,
                        units=params['l_size'],
                        activation=None)

                """
                glimpse_out_layer = tf.layers.dense(
                        # TODO: is multiply better than add?
                        inputs=tf.multiply(
                                x=dense_sensor_layer,
                                y=dense_location_layer),
                        units=params['glimpse_out_size'],
                        activation=tf.nn.relu)
                """
                glimpse_out_layer = dense_sensor_layer * dense_location_layer

                return glimpse_out_layer

    # location_network
    def location_network(rnn_state, sampling, std=params['std'],
                         scope='location_network'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # tanh to force [-1, 1] values
            core_net_output = tf.stop_gradient(rnn_state)
            # core_net_output = rnn_state
            means = tf.clip_by_value(
                tf.layers.dense(
                    core_net_output, params['loc_dim'],
                    kernel_initializer=tf.glorot_normal_initializer(),
                    activation=None),
                -1., 1.)

            if sampling:
                # clip to force [-1, 1] values
                locs = tf.clip_by_value(
                    means + tf.random_normal(
                        [tf.shape(rnn_state)[0], params['loc_dim']],
                        stddev=std),
                    -1., 1.)
            else:
                locs = means

            locs = tf.stop_gradient(locs)
            return locs, means

    # core network
    with tf.variable_scope('core_network', reuse=tf.AUTO_REUSE):
        if params['core_type'] == 'LSTM':
            # TODO: dropout here?
            rnn_cell = tf.nn.rnn_cell.LSTMCell(
                params['core_size'],
                initializer=tf.glorot_normal_initializer())
            state = rnn_cell.zero_state(batch_size, tf.float32)
        else:
            def rnn_cell(glimpse, state):

                dense_state = tf.layers.dense(
                    inputs=state.c,
                    units=params['core_size'],
                    activation=None)

                dense_glimpse = tf.layers.dense(
                    inputs=glimpse,
                    units=params['core_size'],
                    activation=None)

                output = tf.layers.dense(
                    inputs=tf.add(
                        x=dense_state,
                        y=dense_glimpse),
                    units=params['core_size'],
                    activation=tf.nn.relu)

                return output, tf.nn.rnn_cell.LSTMStateTuple(output, output)

            state = tf.nn.rnn_cell.LSTMStateTuple(
                tf.zeros([batch_size, params['core_size']]),
                tf.zeros([batch_size, params['core_size']]))

    # get initial location, glimpses, and state

    # -- locs: [None, loc_dim]
    # zeros are like fixating at center of screen in experiments
    locs = tf.zeros(
        [batch_size, params['loc_dim']])

    # NOTE: the use of a tf.while_loop allows extracting information by passing
    # through and concatenating to a tensor at each iteration.
    # Or, as I'm doing now, by writing to a TensorArray
    # I had been using tf.contrib.seq2seq.dynamic_decode, but because of how it
    # hides the tf.while_loop, it makes it hard to extract information that's
    # computed during the while_loop, e.g. the sampled locations
    locs_ta = tf.TensorArray(tf.float32, params['num_glimpses'])
    loc_means_ta = tf.TensorArray(tf.float32, params['num_glimpses'])
    outputs_ta = tf.TensorArray(tf.float32, params['num_glimpses'])

    def cond(t, *args):
        return tf.less(t, params['num_glimpses'])

    def body(t, state, locs, outputs_ta, locs_ta, loc_means_ta):

        glimpse = glimpse_network(images, locs)
        if is_training:
            glimpse = tf.layers.dropout(glimpse,
                                        rate=params['core_drop'])

        # run the core network
        with tf.variable_scope('core_network', reuse=tf.AUTO_REUSE):
            output, new_state = rnn_cell(glimpse, state)

        loc_input = new_state.c
        if is_training:
            loc_input = tf.layers.dropout(loc_input,
                                          rate=params['core_drop'])
        # get new location
        cur_locs, cur_loc_means = location_network(loc_input, is_training)
        # store new values
        locs_ta = locs_ta.write(t, cur_locs)  # t+1 because of init_loc
        loc_means_ta = loc_means_ta.write(t, cur_loc_means)
        outputs_ta = outputs_ta.write(t, output)

        return t+1, new_state, cur_locs, outputs_ta, locs_ta, loc_means_ta

    # THE MAIN LOOP!
    time = tf.constant(0)
    time, state, _, outputs_ta, locs_ta, loc_means_ta = (
        tf.while_loop(
            cond,
            body,
            [time, state, locs, outputs_ta, locs_ta, loc_means_ta]))

    def ta_to_batch_major(ta):
        # turns a TensorArray of time_steps length with [batch_size, dim]
        # entries at each step into a Tensor [batch_size, time_steps, dim]
        tensor = ta.stack()
        return tf.transpose(tensor, perm=[1, 0, 2])
    # [batch_size, num_glimpses, loc_dim]
    locs = ta_to_batch_major(locs_ta)
    tf.summary.histogram('locs_hist', locs)
    # [batch_size, num_glimpses, loc_dim]
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
        class_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits)

        # reinforce loss for location
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
        with tf.variable_scope('baseline', reuse=tf.AUTO_REUSE):
            # [batch_size, num_glimpses, 1]
            baselines = tf.layers.dense(outputs, 1)
            # [batch_size, num_glimpses]
            baselines = tf.squeeze(baselines)
        # don't train baseline here; only with MSE later
        adv = reward - tf.stop_gradient(baselines)
        # final reinforce loss
        reinforce_loss = -tf.reduce_mean(logll * adv)
        tf.summary.scalar('reinforce_loss', reinforce_loss)

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
        # NOTE: the block below is a direct way of training location and core
        # networks separately; I'm fairly confident that the total loss with
        # stop_gradients as implemented further below also works
        """
        loc_net_vars = [var for var in variables if 'location_network' in
                        var.name]
        core_net_vars = [var for var in variables if var not in loc_net_vars]

        # gradients for core, glimpse, baseline network
        hybrid_loss = class_loss + tf.losses.mean_squared_error(
            baselines, tf.stop_gradient(reward))
        core_gradients = tf.gradients(hybrid_loss, core_net_vars)
        core_gradients, _ = tf.clip_by_global_norm(core_gradients,
                                                   params['max_grad_norm'])

        # gradients for location network
        loc_gradients = tf.gradients(reinforce_loss, loc_net_vars)
        loc_gradients, _ = tf.clip_by_global_norm(loc_gradients,
                                                  params['max_grad_norm'])

        grads_and_vars = []
        grads_and_vars.extend(zip(core_gradients, core_net_vars))
        grads_and_vars.extend(zip(loc_gradients, loc_net_vars))

        """

        # TODO: does this total_loss, with appropriate stop_gradient in
        # location_network, work the same as the above method of explicitly
        # training the two networks separately?
        total_loss = (class_loss + reinforce_loss +
                      # train baseline here, to approximate expected reward, of
                      # which reward is an unbiased estimator
                      tf.losses.mean_squared_error(baselines,
                                                   tf.stop_gradient(reward)))
        tf.summary.scalar('total loss', total_loss)

        gradients = tf.gradients(total_loss, variables)
        for grad, var in zip(gradients, variables):
            tf.summary.histogram('gradient/' + var.name, grad)
        # NOTE: try just back-propagating class loss, thanks to
        # re-parameterization of sampling
        # gradients = tf.gradients(class_loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients,
                                              params['max_grad_norm'])
        grads_and_vars = zip(gradients, variables)

        # TODO: parameterize optimizer
        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = optimizer.apply_gradients(
            grads_and_vars,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=class_loss,
                                          train_op=train_op)
