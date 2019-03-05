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
from __future__ import print_function
import argparse
import pandas as pd
import statsmodels.discrete.discrete_model as sm


def process_data(data):
    data['correct'] = (data['predicted_class'] == data['true_label']).astype(int)
    data['total'] = data['num_yellow'] + data['num_blue']
    data['difference'] = (data['num_yellow'] - data['num_blue']).abs()
    data['n1'], data['n2'] = data['ratio'].str.split(',').str
    data['n1'] = data['n1'].str.lstrip('(').astype(float)
    data['n2'] = data['n2'].str.rstrip(')').astype(float)
    data['ratio'] = data['n1'] / data['n2']
    # data['ratio'] = data['n2'] / data['n1']
    # data['ratio'] = data['n1']
    return data


def regression(data):
    model = sm.Logit.from_formula(
        'correct ~ ratio + total + difference + trial_type + model + ratio:model + ratio:trial_type',
        data)
    results = model.fit()
    print(results.summary())
    print(results.pvalues)


if __name__ == '__main__':

    vggs = [7, 9, 11]
    frames = []

    for vgg in vggs:
        cur = pd.read_csv('results/vgg{}_results.csv'.format(vgg))
        cur['model'] = 'VGG{}'.format(vgg)
        frames.append(cur)
    data = pd.concat(frames, ignore_index=True)
    data = process_data(data)

    sub_data = data[(data['trial_type'] == 'scattered_random') |
                    (data['trial_type'] == 'scattered_pairs')]
    # sub_data = sub_data[(sub_data['n1'] != 2) | (sub_data['n1'] != 3)]
    print('---\nVGG REGRESSION\n---')
    regression(sub_data)

    rams = [4, 8, 16, 24]
    frames = []
    for ram in rams:
        cur = pd.read_csv(
            'results/ram_cnn_g{}_big_relu_test_predict.csv'.format(ram))
        cur['model'] = 'RAM{}'.format(ram)
        frames.append(cur)
    data = pd.concat(frames, ignore_index=True)
    data = process_data(data)
    data['model'] = data['model'].astype('category', ordered=True,
                                         categories=['RAM24', 'RAM16', 'RAM8',
                                                     'RAM4'])
    data['trial_type'] = data['trial_type'].astype(
        'category', ordered=True,
        # is there a way to not hand-code this?
        categories=['column_pairs_sorted', 'column_pairs_mixed',
                    'scattered_pairs', 'scattered_random'])
    print('\n---\nRAM REGRESSION\n---')
    regression(data)
