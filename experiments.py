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
import util
import multiprocessing
import run


# TODO: generalize this to allow for random search not just grid search
def run_experiment(model, config, **kwargs):

    config['model'] = model
    config['train'] = True
    config['eval'] = True
    # get trial variants
    trial_configs = list(util.product_dict(**kwargs))
    for trial_config in trial_configs:
        trial_config['trial_name'] = '_'.join(
            [key + '-' + str(trial_config[key]) for key in trial_config])
    # merge with fixed params
    trial_configs = [dict(trial_dict, **config)
                     for trial_dict in trial_configs]
    # run pool of trials
    pool = multiprocessing.Pool(config['num_cpus'])
    pool.map(run.run, trial_configs)


def ffnn(config):
    run_experiment('ffnn', config,
                   learning_rate=[1e-2, 1e-3, 1e-4],
                   dropout=[0.1, 0.25, 0.5],
                   num_layers=[2, 4, 8],
                   units=[1024, 2048],
                   activation=['relu'])


def cnn(config):
    run_experiment('cnn', config,
                   cnn_architecture=['vgg11', 'vgg13'],
                   learning_rate=[1e-2, 1e-3, 1e-4],
                   dropout=[0.1, 0.25])


def small_cnn(config):
    run_experiment('cnn', config,
                   cnn_architecture=['vgg5', 'vgg7', 'vgg9'],
                   learning_rate=[1e-4],
                   dropout=[0.25])


def ram(config):
    run_experiment('ram', config,
                   learning_rate=[1e-4, 1e-5],
                   patch_size=[12],
                   num_patches=[2, 4],
                   glimpse_size=[128, 256],
                   glimpse_out_size=[256],
                   core_size=[256, 512],
                   num_glimpses=[8, 16, 24],
                   core_type=['LSTM', 'RAMcell'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--exp', help='name of exp to run', type=str,
                        choices=['ffnn', 'cnn', 'small_cnn', 'ram'], default='ram')
    parser.add_argument('--num_cpus', help='how many gpus for exp', type=int,
                        default=12)
    # file system
    parser.add_argument('--out_path', help='path to outputs', type=str,
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

    globals()[args.exp](args.__dict__)
