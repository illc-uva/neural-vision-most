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
import itertools
import os
import pandas as pd
import data


# TODO: document this method
# see https://stackoverflow.com/a/5228294/9370349
def product_dict(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))


def dicts_to_csv(ls, filename):
    frame = pd.DataFrame(ls)
    frame.to_csv(filename)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def simplify(numer, denom):
    divisor = gcd(numer, denom)
    return numer / divisor, denom / divisor


def get_ratio(n1, n2):
    return simplify(max(n1, n2), min(n1, n2))


def process_predictions(predictions, include_locs=True, num_glimpses=None):
    results = pd.DataFrame(predictions)
    results['filename'] = results['filename'].astype(str)

    filenames = results['filename'].map(
        lambda filename: os.path.split(filename)[1])
    # TODO: make this more elegant!
    results['true_label'] = filenames.map(
        lambda filename: data.label_from_filename(filename))
    results['trial_type'] = filenames.map(
        lambda filename: filename[:filename.find('b')-1])
    results['num_blue'] = filenames.map(
        lambda filename: data.extract_color_dict(filename)['b'])
    results['num_yellow'] = filenames.map(
        lambda filename: data.extract_color_dict(filename)['y'])

    results['ratio'] = results.index.map(
        lambda idx: get_ratio(results['num_blue'][idx],
                              results['num_yellow'][idx]))

    if num_glimpses:
        results['num_glimpses'] = [num_glimpses]*len(results.index)
    if not include_locs:
        del results['locs']
    return results
