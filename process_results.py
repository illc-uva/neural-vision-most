"""
Copyright (c) 2019 Shane Steinert-Threlkeld and Lewis O'Sullivan

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
import pandas as pd
import numpy as np


def vgg_filename(depth):
    return 'results/vgg{}_results.csv'.format(depth)


def ram_filename(glimpses):
    return 'results/ram_cnn_g{}_big_relu_test_predict.csv'.format(glimpses)


def process_file(filename, model_name):
    data = pd.read_csv(filename)
    data['correct'] = (data['predicted_class'] ==
                       data['true_label']).astype(int)
    means = data.groupby(['trial_type', 'ratio'],
                         as_index=False)['correct'].mean()
    means.rename(columns={'correct': 'accuracy'}, inplace=True)
    means['model'] = model_name
    means['n1'], means['n2'] = means['ratio'].str.split(',').str
    means['n1'] = means['n1'].str.lstrip('(').astype(float)
    means['n2'] = means['n2'].str.rstrip(')').astype(float)
    means['ratio'] = means['n1'] / means['n2']
    return means


if __name__ == '__main__':

    vggs = [7, 9, 11, 13]
    rams = [4, 8, 16, 24]

    processed = []

    for vgg in vggs:
        processed.append(
            process_file(vgg_filename(vgg), 'VGG{}'.format(vgg)))

    for ram in rams:
        processed.append(
            process_file(ram_filename(ram), 'RAM{}'.format(ram)))

    data = pd.concat(processed, ignore_index=True)
    data.to_csv('results/mean_accuracies.csv')
