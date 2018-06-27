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
import pandas as pd


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
