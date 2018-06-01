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
# TODO: command-line args!
# TODO: model types

img_feature_column_name = 'x'
img_size = 256
batch_size = 8
num_classes = 2


def make_input_fn(filename_pattern,
                  shuffle=True, batch_size=None, num_epochs=1):
    dataset = data.make_dataset(filename_pattern, shuffle,
                                batch_size, num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return {img_feature_column_name: features}, labels


def run():

    def train_input_fn():
        return make_input_fn('images/train/*.png', batch_size=batch_size,
                             num_epochs=3)

    def test_input_fn():
        return make_input_fn('images/test/*.png', shuffle=False)

    img_feature_columns = [tf.feature_column.numeric_column(
        img_feature_column_name, shape=[img_size, img_size, 3])]

    ffnn_runconfig = tf.estimator.RunConfig(
        save_checkpoints_secs=60,
        keep_checkpoint_max=3
    )
    model = tf.estimator.DNNClassifier(
        feature_columns=img_feature_columns,
        hidden_units=[256, 256],
        activation_fn=tf.nn.elu,
        n_classes=num_classes,
        model_dir='/tmp/test',
        config=ffnn_runconfig)

    model.train(input_fn=train_input_fn)
    print(model.evaluate(input_fn=test_input_fn))


run()
