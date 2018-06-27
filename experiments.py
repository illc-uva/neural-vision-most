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
import ray
import ray.tune as tune
import run


def ffnn(config):
    config['model'] = 'ffnn'
    config['learning_rate'] = tune.grid_search([1e-2, 1e-3, 1e-4])
    config['dropout'] = tune.grid_search([0.1, 0.25, 0.5])
    config['num_layers'] = tune.grid_search([2, 4, 8])
    config['units'] = tune.grid_search([1024, 2048])
    config['trial_name'] = lambda spec: '_'.join(
        [key + '-' + str(spec.config[key]) for key in ['learning_rate',
                                                       'dropout', 'num_layers',
                                                       'units']])
    config['layers'] = lambda spec: ([
        {'units': spec.config.units,
         # 'activation': tf.nn.relu,
         'dropout': spec.config.dropout}]*spec.config.num_layers)
    tune.run_experiments({
        'ffnn_experiment': {
            'run': 'run',
            'local_dir': config['ray_path'],
            'config': config
        }
    })


def cnn(config):
    config['model'] = 'cnn'
    config['learning_rate'] = tune.grid_search([1e-2, 1e-3, 1e-4])
    config['cnn_architecture'] = tune.grid_search(['vgg11', 'vgg13'])
    config['dropout'] = tune.grid_search([0.1, 0.25])
    config['trial_name'] = lambda spec: '_'.join(
        [key + '-' + str(spec.config[key]) for key in ['learning_rate',
                                                       'dropout',
                                                       'cnn_architecture']])
    tune.run_experiments({
        'cnn_experiment': {
            'run': 'run',
            'local_dir': config['ray_path'],
            'config': config
        }
    })


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--exp', help='name of exp to run', type=str,
                        choices=['ffnn', 'cnn', 'ram'], default='ram')
    parser.add_argument('--num_gpus', help='how many gpus for exp', type=int,
                        default=0)
    # file system
    parser.add_argument('--out_path', help='path to outputs', type=str,
                        default='/tmp')
    parser.add_argument('--ray_path', help='path to ray outputs', type=str,
                        default='/tmp')
    # NOTE: these should be absolute path names since ray moves working
    # directory under the hood.  Don't use relative path names!
    parser.add_argument('--train_images', help='regex to path of test images',
                        type=str, default='images/train/*.png')
    parser.add_argument('--test_images', help='regex to path of test images',
                        type=str, default='images/test/*.png')
    parser.add_argument('--val_images', help='regex to path of test images',
                        type=str, default='images/val/*.png')
    # experiment arguments
    parser.add_argument('--img_size', help='size (one int, width and height)',
                        type=int, default=256)
    parser.add_argument('--num_channels', help='number of channels', type=int,
                        default=3)
    parser.add_argument('--batch_size', help='batch size', type=int,
                        default=64)
    parser.add_argument('--num_epochs', help='number of epochs', type=int,
                        default=4)
    parser.add_argument('--epochs_per_eval', help='how often to evaluate',
                        type=int, default=1)
    parser.add_argument('--patience', help='how many epochs w/o improvement',
                        type=int, default=10)
    parser.add_argument('--img_feature_name', help='name of feature', type=str,
                        default='image')
    parser.add_argument('--num_classes', help='how many classes', type=int,
                        default=2)

    args = parser.parse_args()

    ray.init(num_gpus=args.num_gpus)
    tune.register_trainable('run', run.run)

    globals()[args.exp](args.__dict__)
