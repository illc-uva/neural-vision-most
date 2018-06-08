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
import data
import models
# TODO: command-line args!
# TODO: different model types

img_feature_name = 'image'
img_size = 256
patch_size = 12
batch_size = 16
num_classes = 2

tf.logging.set_verbosity(tf.logging.INFO)


def run():
    # TODO: figure out best way of having FFNN, CNN, RAM, each with parameters,
    # and called from the command-line

    # TODO: should input_fn's use one_shot_iterators and return get_next?
    def train_input_fn():
        return data.make_dataset('images/train/*.png', img_feature_name,
                                 shuffle=True, batch_size=batch_size,
                                 img_size=img_size,
                                 num_epochs=3)

    def test_input_fn():
        return data.make_dataset('images/test/*.png', img_feature_name,
                                 shuffle=False)

    save_runconfig = tf.estimator.RunConfig(
        save_checkpoints_secs=60,
        keep_checkpoint_max=3
    )
    
       # Create the Estimator
    model = tf.estimator.Estimator(
        models.cnn_model_fn,
        model_dir='/tmp/test_cnn',
        config=save_runconfig,
        params={
            'img_feature_name': img_feature_name,
            'layers': [
                {'filters': 32,
                 'kernel_size' : 4,
                 'padding' : "SAME",
                 'activation': tf.nn.relu,
                 'pool_size' : 2,
                 'strides' : 2},
                 {'filters': 64,
                 'kernel_size' : 4,
                 'padding' : "SAME",
                 'activation': tf.nn.relu,
                 'pool_size' : 2,
                 'strides' : 2}],
            'num_classes': 2})

    """
    model = tf.estimator.Estimator(
        models.ram_model_fn,
        model_dir='/tmp/ram_test',
        config=save_runconfig,
        params={
            'img_feature_name': img_feature_name,
            'img_size': img_size,
            'patch_size': patch_size,
            # TODO: get these from paper
            'g_size': 64,
            'l_size': 64,
            'glimpse_out_size': 128,
            'loc_dim': 2,  # x, y
            'std': 0.2,
            'core_size': 128,
            'num_glimpses': 4,
            'num_classes': 2,
            'max_grad_norm': 5.0
        })
    model.train(input_fn=train_input_fn)
    # print(list(model.predict(input_fn=test_input_fn)))
    print(model.evaluate(input_fn=test_input_fn))


    img_feature_columns = [tf.feature_column.numeric_column(
        img_feature_name, shape=[img_size, img_size, 3])]

    model = tf.estimator.Estimator(
        models.ffnn_model_fn,
        model_dir='/tmp/test',
        config=save_runconfig,
        params={
            'feature_columns': img_feature_columns,
            'layers': [
                {'units': 128,
                 'activation': tf.nn.elu,
                 'dropout': None}]*2,
            'num_classes': 2})
    """

    model.train(input_fn=train_input_fn)
    print(model.evaluate(input_fn=test_input_fn))
    


run()
