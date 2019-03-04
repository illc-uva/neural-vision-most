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
import argparse
import pandas as pd
import numpy as np
from plotnine import *


def by_ratio_plot(data):
    means = data.groupby(['ratio', 'model'], as_index=False).aggregate(np.mean)
    by_ratio_plot = (ggplot(means)
                     + geom_point(aes(x='n1', y='accuracy', colour='model'))
                     + geom_line(aes(x='n1', y='accuracy', colour='model'))
                     + scale_x_continuous(breaks=range(2, 11),
                                        labels=['{}/{}'.format(n1-1, n1) for n1 in
                                                range(2, 11)],
                                        limits=[2, 10],
                                        name='ratio')
                     + scale_color_brewer(type='div', palette='RdBu')
                     # + theme(legend_position='bottom')
                    )
    by_ratio_plot.save('results/by_ratio_plot.png', width=8, height=4, dpi=300)


def by_type_plot(data):
    means = data.groupby(['trial_type', 'model'], as_index=False).aggregate(np.mean)
    plot = (ggplot(means)
            + geom_col(aes(x='trial_type', y='accuracy', fill='trial_type'),
                       position='dodge')
            + facet_wrap('model', nrow=2)
            + theme(axis_text_x=element_blank(), axis_ticks=element_blank(),
                    axis_title_x=element_blank(), legend_position='bottom'))
    plot.save('results/trial_type_plot.png', width=8, height=5, dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--means', type=str,
                        help='file with mean accuracy data',
                        default='results/mean_accuracies.csv')
    parser.add_argument('--ratio_plot', dest='ratio_plot', action='store_true')
    parser.add_argument('--no_ratio_plot', dest='ratio_plot',
                        action='store_false')
    parser.set_defaults(ratio_plot=False)
    parser.add_argument('--trial_type_plot', dest='trial_type_plot',
                        action='store_true')
    parser.add_argument('--no_trial_type_plot', dest='trial_type_plot',
                        action='store_false')
    parser.set_defaults(trial_type_plot=False)
    args = parser.parse_args()

    data = pd.read_csv(args.means)
    # order models "correctly"
    data['model'] = data['model'].astype('category', ordered=True,
                                         categories=list(data['model'].unique()))

    if args.ratio_plot:
        by_ratio_plot(data)

    if args.trial_type_plot:
        by_type_plot(data)
