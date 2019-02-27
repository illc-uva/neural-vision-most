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
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import matplotlib.pyplot as plt


def weber_function(ns, w):
    """
    Args:
        ns: 2xM numpy array, rows are n1, n2: larger/smaller number of dots
        w: weber fraction
    """
    n1 = ns[0]
    n2 = ns[1]
    return 1 - (0.5 * scipy.special.erfc(
        (n1 - n2) / (np.sqrt(2) * w * np.sqrt(n1**2 + n2**2)))) # * 100


def fit_weber(data):
    # first, add 50% accuracy at 1/1 ratio
    data = data.append({'n1': 1, 'n2': 1, 'mean_accuracy': 0.5}, ignore_index=True)
    ns = data.as_matrix(columns=['n1', 'n2']).T
    w, cov = scipy.optimize.curve_fit(weber_function, ns, data['mean_accuracy'])
    print w
    print cov
    fitted_ys = weber_function(ns, w)
    data['ratio'] = data['n1'] / data['n2']
    plt.scatter(data['ratio'], data['mean_accuracy'])
    plt.plot(data['ratio'], fitted_ys)
    plt.show()


def fit_models(mean_file, model_prefix):
    data = pd.read_csv(mean_file)
    models = list(
        data[data['model'].str.startswith(model_prefix)]['model'].unique())
    for model in models:
        print model
        model_data = data[data['model'] == model]
        fit_weber(model_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--means', type=str,
                        help='file with mean accuracy data',
                        default='mean_accuracies_long.csv')
    parser.add_argument('--model_prefix', type=str,
                        help='prefix of model(s) to analyze',
                        default='RAM')
    args = parser.parse_args()
    fit_models(args.means, args.model_prefix)
