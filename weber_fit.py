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
from plotnine import *


def r2(y_obs, y_fit):
    mean_obs = np.mean(y_obs)
    ss_tot = np.sum((y_obs - mean_obs)**2)
    ss_res = np.sum((y_obs - y_fit)**2)
    return 1 - ss_res / ss_tot


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
    ns = data.as_matrix(columns=['n1', 'n2']).T
    w, cov = scipy.optimize.curve_fit(weber_function, ns, data['mean_accuracy'])
    print w
    print cov
    fitted_ys = weber_function(ns, w)
    rsq = r2(data['mean_accuracy'], fitted_ys)
    print rsq
    return w, cov, rsq, fitted_ys


def fit_models(mean_file, model_prefix):
    data = pd.read_csv(mean_file)
    models = list(
        data[data['model'].str.startswith(model_prefix)]['model'].unique())
    model_frames = []
    curve_fits = []
    for model in models:
        print model
        model_data = data[data['model'] == model]
        # first, add 50% accuracy at 1/1 ratio
        model_data = model_data.append(
            {'model': model, 'n1': 1, 'n2': 1, 'mean_accuracy': 0.5},
            ignore_index=True)
        w, cov, rsq, ys = fit_weber(model_data)
        curve_fits.append({
            'model': model,
            'w': w[0],
            'r_squared': rsq,
            'cov': cov[0][0]})
        model_data['fit_weber'] = ys
        model_frames.append(model_data)
    curve_fits = pd.DataFrame(curve_fits)
    curve_fits.to_csv('curve_fits_' + model_prefix + '.csv')
    models = pd.concat(model_frames)
    models['ratio'] = models['n1'] / models['n2']
    print models
    print (ggplot(models, aes(x='ratio'))
           + geom_point(aes(y='mean_accuracy', colour='model'))
           + geom_line(aes(y='fit_weber', colour='model')))


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
