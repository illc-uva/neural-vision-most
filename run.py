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
from distutils import dir_util
from collections import defaultdict
import tensorflow as tf
import data
import models
import util

tf.logging.set_verbosity(tf.logging.INFO)


def ffnn(config, run_config):

    img_feature_columns = [
        tf.feature_column.numeric_column(
            config['img_feature_name'],
            shape=[config['img_size'], config['img_size'],
                   config['num_channels']])]
    # build layers from specification
    if not config['layers']:
        config['layers'] = [
            {'units': config['units'],
             'dropout': config['dropout'],
             'activation': getattr(tf.nn, config['activation'])}
        ]*config['num_layers']
    return tf.estimator.Estimator(
        models.ffnn_model_fn,
        model_dir=config['model_dir'],
        config=run_config,
        params={
            'feature_columns': img_feature_columns,
            'layers': config['layers'] or [
                {'units': 128,
                 'activation': tf.nn.relu,
                 'dropout': None}]*2,
            'learning_rate': config['learning_rate'] or 0.01,
            'num_classes': config['num_classes']})


def cnn(config, run_config):
    architectures = {
        'vgg11': {
            'layers': [
                {'num_convs': 1,
                 'filters': 64,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
                {'num_convs': 1,
                 'filters': 128,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
                {'num_convs': 2,
                 'filters': 256,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
                {'num_convs': 2,
                 'filters': 512,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
                {'num_convs': 2,
                 'filters': 512,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
            ],
            'dense': [
                {'units': 4096,
                 'activation': tf.nn.relu,
                 'rate': config['dropout'] or 0.1}
            ]*2
        },
        'vgg13': {
            'layers': [
                {'num_convs': 2,
                 'filters': 64,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
                {'num_convs': 2,
                 'filters': 128,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
                {'num_convs': 2,
                 'filters': 256,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
                {'num_convs': 2,
                 'filters': 512,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
                {'num_convs': 2,
                 'filters': 512,
                 'kernel_size': 3,
                 'pool_strides': 2,
                 'pool_size': 2,
                 'padding': 'same',
                 'activation': tf.nn.relu},
            ],
            'dense': [
                {'units': 4096,
                 'activation': tf.nn.relu,
                 'rate': config['dropout'] or 0.1}
            ]*2
        }
    }
    architecture = architectures[config['cnn_architecture'] or 'vgg11']
    return tf.estimator.Estimator(
        models.cnn_model_fn,
        model_dir=config['model_dir'],
        config=run_config,
        params={
            'img_feature_name': config['img_feature_name'],
            'layers': architecture['layers'],
            'dense': architecture['dense'],
            'learning_rate': config['learning_rate'] or 1e-3,
            'num_classes': config['num_classes']})


def ram(config, run_config):
    return tf.estimator.Estimator(
        models.ram_model_fn,
        model_dir=config['model_dir'],
        config=run_config,
        params={
            'img_feature_name': config['img_feature_name'],
            'img_size': config['img_size'],
            'patch_size': config['patch_size'] or 12,  # TODO: random search?
            'patch_scale': 2,
            'num_patches': config['num_patches'] or 2,
            'g_size': config['glimpse_size'] or 256,
            'l_size': config['glimpse_size'] or 256,
            'glimpse_out_size': config['glimpse_out_size'] or 256,
            'loc_dim': 2,  # x, y
            'std': 0.03,  # TODO: random search?
            'core_size': config['core_size'] or 512,
            'num_glimpses': config['num_glimpses'] or 12,  # TODO: vary glimpse number by batch
            'learning_rate': config['learning_rate'] or 1e-5,
            'num_classes': config['num_classes'],
            'max_grad_norm': 5.0,
            'core_type': config['core_type'] or 'LSTM',
            'core_drop': 0.2,
            'glimpse_type': config['glimpse_type'] or 'CNN'
            })


def run(config):

    save_runconfig = tf.estimator.RunConfig(
        keep_checkpoint_max=1
    )
    # make config return default value None for all keys, so that methods can
    # set default arguments using `or`
    config = defaultdict(lambda: None, config)

    if config['train']:

        model_dir = (config['out_path'] + '/' + config['model'] + '/' +
                     (config['trial_name'] or ''))
        config['model_dir'] = model_dir
        # Create the Estimator, using --model arg (default ram)
        model = globals()[config['model']](config, save_runconfig)

        # prep eval storage
        eval_dicts = []
        lowest_val_loss = 1e10  # big initial number
        patience_steps = config['patience'] // config['epochs_per_eval']
        # make directory for saving the best model
        best_model_dir = model_dir + 'best'
        dir_util.mkpath(best_model_dir)

        for step in range(config['num_epochs'] // config['epochs_per_eval']):

            # TODO: can these input_fn's be outside the loop?
            def train_input_fn():
                return data.make_dataset(config['train_images'],
                                         config['img_feature_name'],
                                         config['img_size'],
                                         config['num_channels'],
                                         batch_size=config['batch_size'],
                                         num_epochs=config['epochs_per_eval'],
                                         shuffle=True)

            def eval_input_fn():
                return data.make_dataset(config['val_images'],
                                         config['img_feature_name'],
                                         config['img_size'],
                                         config['num_channels'],
                                         batch_size=config['batch_size'],
                                         shuffle=False)

            print('Training beginning.')
            model.train(input_fn=train_input_fn)

            print('Evaluation beginning.')
            eval_results = model.evaluate(input_fn=eval_input_fn)
            eval_results['num_epochs'] = (step+1) * config['epochs_per_eval']
            eval_dicts.append(eval_results)
            cur_loss = eval_results['loss']
            print('Loss after epoch {}: {}'.format(
                eval_results['num_epochs'], cur_loss))

            # save best model, report to ray tune, early stop, etc
            if cur_loss < lowest_val_loss:
                print('New best model, saving to ' + best_model_dir)
                dir_util.remove_tree(best_model_dir, verbose=True)
                dir_util.copy_tree(model_dir, best_model_dir)
                lowest_val_loss = cur_loss

            if patience_steps < step + 1:
                if cur_loss > eval_dicts[-(patience_steps+1)]['loss']:
                    print('No improvement over {} epochs; ending training.'.format(
                        config['patience']))
                    break

        util.dicts_to_csv(eval_dicts, model_dir + 'train_eval.csv')

        if config['eval']:
            print('Evaluating best model on the test set.')
            config['model_dir'] = best_model_dir

    def test_input_fn():
        return data.make_dataset(config['test_images'],
                                 config['img_feature_name'],
                                 config['img_size'],
                                 config['num_channels'],
                                 batch_size=config['batch_size'],
                                 shuffle=False)

    if config['eval']:
        model = globals()[config['model']](config, save_runconfig)
        util.dicts_to_csv([model.evaluate(input_fn=test_input_fn)],
                          model_dir + 'test_eval.csv')

    if config['predict']:
        config['model_dir'] = config['out_path']
        model = globals()[config['model']](config, save_runconfig)
        results = model.predict(input_fn=test_input_fn)
        output = util.process_predictions(results, include_locs=False,
                                          num_glimpses=config['num_glimpses'])
        print(output)
        output.to_csv(config['model_dir'] + 'test_predict.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # file system arguments
    parser.add_argument('--out_path', help='path to outputs', type=str,
                        default='/tmp')
    parser.add_argument('--train_images', help='regex to path of test images',
                        type=str, default='images/train/*.png')
    parser.add_argument('--test_images', help='regex to path of test images',
                        type=str, default='images/test/*.png')
    parser.add_argument('--val_images', help='regex to path of test images',
                        type=str, default='images/val/*.png')
    # what to do arguments
    parser.add_argument('--no_train', dest='train', action='store_false')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=True)
    parser.add_argument('--no_eval', dest='eval', action='store_false')
    parser.add_argument('--eval', dest='eval', action='store_true')
    parser.set_defaults(eval=True)
    parser.add_argument('--no_predict', dest='predict', action='store_false')
    parser.add_argument('--predict', dest='predict', action='store_true')
    parser.set_defaults(predict=False)
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
    parser.add_argument('--learning_rate', help='learning rate', type=float,
                        default=1e-5)
    # model arguments
    # NOTE: for now, most parameters specifically for the models are defined in
    # the respective methods, not passed through the command-line
    parser.add_argument('--model', help='which model to use',
                        choices=['ffnn', 'cnn', 'ram'],
                        default='ram')
    parser.add_argument('--img_feature_name', help='name of feature', type=str,
                        default='image')
    parser.add_argument('--num_classes', help='how many classes', type=int,
                        default=2)
    parser.add_argument('--core_type', help='type of RNN', type=str,
                        default='LSTM')
    parser.add_argument('--glimpse_type', help='type of glimpse net', type=str,
                        default='CNN')
    parser.add_argument('--num_glimpses', help='type of glimpse net', type=int,
                        default=4)
    parser.add_argument('--cnn_architecture', help='architecture for CNN',
                        type=str, default=None)
    # get all args
    args = parser.parse_args()

    # NOTE: using args.__dict__, because ray_tune passes a dictionary to the
    # method, not an object with attributes 
    run(args.__dict__)
