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
import pandas as pd
from plotnine import *


def every_two(limits):
    xmin, xmax = limits
    xmin = int(xmin)
    xmax = int(xmax)
    return [x for x in range(xmin, xmax+1) if x % 2 == 0]


def learning_plot(data, out_file=None, scale_x=None):

    plot = (ggplot(data) +
            geom_line(aes(x='num_epochs', y='accuracy', colour='model')) +
            # scale_x_continuous(breaks=every_two) +
            ylab('val accuracy'))
    if scale_x:
        plot += scale_x_continuous(breaks=scale_x)
    if out_file:
        plot.save(out_file, height=6, width=8, dpi=300)
    else:
        print(plot)


def multi_training(model_roots, out_file, scale_x=None):
    frames = []
    for model in model_roots:
        train_data = pd.read_csv('results/{}_train_eval.csv'.format(model))
        train_data['model'] = model
        frames.append(train_data)
    all_training = pd.concat(frames, ignore_index=True)
    all_training['model'] = all_training['model'].astype(
        'category', ordered=True, categories=model_roots)
    learning_plot(all_training, out_file, scale_x)


if __name__ == '__main__':

    vggs = ['vgg{}'.format(vgg) for vgg in [7, 9, 11, 13]]
    multi_training(vggs, 'results/vgg_training.png', scale_x=every_two)

    rams = ['RAM{}'.format(ram) for ram in [4, 8, 16, 24]]
    multi_training(rams, 'results/ram_training.png')
