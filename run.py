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
batch_size = 64
num_classes = 2

tf.logging.set_verbosity(tf.logging.INFO)


def run():

    def train_input_fn():
        # return make_input_fn('images/train/*.png', batch_size=batch_size)
        return data.make_dataset('images/train/*.png', img_feature_name,
                                 shuffle=True, batch_size=batch_size,
                                 num_epochs=2)

    def test_input_fn():
        return data.make_dataset('images/test/*.png', img_feature_name,
                                 shuffle=False)

    img_feature_columns = [tf.feature_column.numeric_column(
        img_feature_name, shape=[img_size, img_size, 3])]

    ffnn_runconfig = tf.estimator.RunConfig(
        save_checkpoints_secs=60,
        keep_checkpoint_max=3
    )
    model = tf.estimator.Estimator(
        models.ffnn_model_fn,
        model_dir='/tmp/test',
        config=ffnn_runconfig,
        params={
            'feature_columns': img_feature_columns,
            'layers': [
                {'units': 128,
                 'activation': tf.nn.elu,
                 'dropout': None}]*2,
            'num_classes': 2})

    model.train(input_fn=train_input_fn)
    print(model.evaluate(input_fn=test_input_fn))


run()
