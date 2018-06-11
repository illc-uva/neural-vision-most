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
import argparse
import tensorflow as tf
import data
import models

tf.logging.set_verbosity(tf.logging.INFO)


def ffnn(args, run_config):

    img_feature_columns = [tf.feature_column.numeric_column(
        args.img_feature_name, shape=[args.img_size, args.img_size,
                                      args.num_channels])]

    return tf.estimator.Estimator(
        models.ffnn_model_fn,
        model_dir=args.out_path,
        config=run_config,
        params={
            'feature_columns': img_feature_columns,
            'layers': [
                {'units': 128,
                 'activation': tf.nn.elu,
                 'dropout': None}]*2,
            'num_classes': args.num_classes})


def cnn(args, run_config):
    return tf.estimator.Estimator(
        models.cnn_model_fn,
        model_dir=args.out_path,
        config=run_config,
        params={
            'img_feature_name': args.img_feature_name,
            'layers': [
                {'filters': 32,
                 'kernel_size': 4,
                 'padding': "SAME",
                 'activation': tf.nn.relu,
                 'pool_size': 2,
                 'strides': 2},
                {'filters': 64,
                 'kernel_size': 4,
                 'padding': "SAME",
                 'activation': tf.nn.relu,
                 'pool_size': 2,
                 'strides': 2}],
            'num_classes': args.num_classes})


def ram(args, run_config):
    return tf.estimator.Estimator(
        models.ram_model_fn,
        model_dir=args.out_path,
        config=run_config,
        params={
            'img_feature_name': args.img_feature_name,
            'img_size': args.img_size,
            'patch_size': 12,
            # TODO: get these from paper
            'g_size': 64,
            'l_size': 64,
            'glimpse_out_size': 128,
            'loc_dim': 2,  # x, y
            'std': 0.2,
            'core_size': 128,
            'num_glimpses': 4,
            'num_classes': args.num_classes,
            'max_grad_norm': 5.0
        })


# TODO: eval logging hook?
def run(args):

    def train_input_fn():
        return data.make_dataset(args.train_images, args.img_feature_name,
                                 args.img_size, args.num_channels,
                                 shuffle=True, batch_size=args.batch_size,
                                 num_epochs=args.num_epochs)

    def test_input_fn():
        return data.make_dataset(args.test_images, args.img_feature_name,
                                 args.img_size, args.num_channels,
                                 shuffle=False)

    save_runconfig = tf.estimator.RunConfig(
        save_checkpoints_secs=60,
        keep_checkpoint_max=3
    )

    # Create the Estimator, using --model arg (default ram)
    model = globals()[args.model](args, save_runconfig)

    model.train(input_fn=train_input_fn)
    print(list(model.predict(input_fn=test_input_fn)))
    print(model.evaluate(input_fn=test_input_fn))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # file system arguments
    parser.add_argument('--out_path', help='path to outputs', type=str,
                        default='/tmp/ram')
    parser.add_argument('--train_images', help='regex to path of test images',
                        type=str, default='images/train/*.png')
    parser.add_argument('--test_images', help='regex to path of test images',
                        type=str, default='images/test/*.png')
    # experiment arguments
    parser.add_argument('--img_size', help='size (one int, width and height)',
                        type=int, default=256)
    parser.add_argument('--num_channels', help='number of channels', type=int,
                        default=3)
    parser.add_argument('--batch_size', help='batch size', type=int,
                        default=64)
    parser.add_argument('--num_epochs', help='number of epochs', type=int,
                        default=4)
    parser.add_argument('--img_feature_name', help='name of feature', type=str,
                        default='image')
    # model arguments
    # NOTE: for now, most parameters specifically for the models are defined in
    # the respective methods, not passed through the command-line
    parser.add_argument('--model', help='which model to use',
                        choices=['ffnn', 'cnn', 'ram'],
                        default='ram')
    parser.add_argument('--num_classes', help='how many classes', type=int,
                        default=2)
    # get all args
    args = parser.parse_args()

    run(args)
