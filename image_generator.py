"""
Copyright (c) 2018 Shane Steinert-Threlkeld

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

This module contains methods that generate images of colored dots
for use in psychological studies, especially those testing numerical abilities.
In particular, it contains Python implementations of image generators for
all four conditions of the study presented in the paper

    * Pietroski, P., Lidz, J., Hunter, T., & Halberda, J. (2009).
        The Meaning of `Most’: Semantics, Numerosity and Psychology.
        _Mind and Language_, 24(5), 554–585.

Usage as follows:
    * each trial type has a corresponding method, e.g. `scattered_pairs`
    * these methods take a `colors_dict` argument, a dictionary whose keys
        are strings corresponding to matplotlib colors and whose values are
        the number of dots of that color, e.g.
        {'y': 9, 'b': 10}
    * each method returns a list of `Dot`, where a `Dot` is an object with
        four fields: x, y, radius, and color
    * the method `make_image` takes such a list of `Dot`s and then
        generates and saves an image to disk
    * the method `make_batch` takes a list of trial types, a list of of
        `colors_dict`s, and a number N, and produces N images of each
        trial_type, color_dict pair
"""

from collections import namedtuple
import random
import math
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


Dot = namedtuple('Dot', ['x', 'y', 'radius', 'color'])


def clip(val, min_val, max_val):
    """Clips `val` to be in the range [min_val, max_val]. """
    return max(min(val, max_val), min_val)


def polar_to_cartesian(theta, r):
    """Converts polar coordinates to Cartesian. """
    return r*math.cos(theta), r*math.sin(theta)


def no_overlap(dots, x, y, radius):
    """Checks whether a new dot will have any overlap with an existing
    array of dots.

    Args:
        dots: an iterable of `Dot`s
        x: x-value of center of new dot
        y: y-value of center of new dot
        r: radius of new dot

    Returns:
        True if the new dot has no overlap with any of the dots in `dots',
        False otherwise
    """
    return all([(x - dot.x)**2 + (y - dot.y)**2 >= (radius + dot.radius)**2
                for dot in dots])


def get_random_radii(colors_dict, min_radius, max_radius, std=1):
    """Gets random radii of dots for a color_dict.  Radii are sampled from
    a Gaussian distribution with mean (max_r - min_r) / 2 and standard
    deviation std, then clipped.

    Args:
        color_dict: dictionary of colors, with integer values
        min_radius: smallest radius
        max_radius: biggest radius
        std: standard deviation

    Returns:
        a dictionary, with the same keys as colors_dict, and values a list of
        colors_dict[color] floating point numbers
    """
    mean = (max_radius - min_radius) / 2
    return {color: [clip(random.gauss(mean, std), min_radius, max_radius)
                    for _ in range(colors_dict[color])]
            for color in colors_dict}


def get_area_controlled_radii(colors_dict, min_radius, max_radius, std=0.5,
                              total_area=None):
    """Gets area controlled radii: the sum of the areas of circles of each
    color will be equal (either to total_area or to the total area taken by the
    largest number in colors_dict dots of mean radius).

    Args:
        colors_dict: as above
        min_radius: as above
        max_radius: as above
        std: as above
        total_area: a float, the total area to distribute to each color.  If
            not specified, this will be set to N*(max_radius - min_radius)/2^2,
            where N is the largest value in colors_dict

    Returns:
        a dictionary, as above
    """
    mean = (max_radius - min_radius) / 2
    if not total_area:
        total_area = math.pi*(mean**2)*max(colors_dict.values())
    radii = {color: [] for color in colors_dict}
    for color in colors_dict:
        num_remaining = colors_dict[color]
        area_remaining = total_area
        while num_remaining > 1:
            mean = math.sqrt(area_remaining / (num_remaining*math.pi))
            # get radius that is not too big to use up all remaining area!
            found_r = False
            while not found_r:
                r = clip(random.gauss(mean, std), min_radius, max_radius)
                if math.pi*r**2 < area_remaining:
                    found_r = True
            radii[color].append(r)
            area_remaining -= math.pi*r**2
            num_remaining -= 1
        radii[color].append(math.sqrt(area_remaining / math.pi))
    return radii


def scattered_random(colors_dict, area_control=False,
                     total_area=None,
                     num_pixels=(256, 256), padding=16,
                     min_radius=1, max_radius=5, std=1):
    """Generates ScatteredRandom images: the dots are scattered
    randomly through the image. """
    x_min, y_min = padding, padding
    x_max, y_max = num_pixels[0] - padding, num_pixels[1] - padding
    dots = []
    if area_control:
        radii = get_area_controlled_radii(colors_dict, min_radius, max_radius,
                                          std=std, total_area=total_area)
    else:
        radii = get_random_radii(colors_dict, min_radius, max_radius, std=std)
    # print({color: sum([math.pi*r**2 for r in radii[color]]) for color in radii})
    for color in colors_dict:
        for r in radii[color]:
            new_dot_added = False
            while not new_dot_added:
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                # avoid overlap with existing circles
                if no_overlap(dots, x, y, r):
                    dots.append(Dot(x=x, y=y, radius=r, color=color))
                    new_dot_added = True
    return dots


def scattered_split(colors_dict, area_control=False,
                    num_pixels=(512, 256), padding=24,
                    min_radius=1, max_radius=5, std=0.5,
                    color_order=None):
    """Generates ScatteredSplit images: the dots are scattered randomly through
    the image, but each color has its own region of the image, with different
    colors laid out horizontally. """
    width_per = num_pixels[0] / len(colors_dict)
    mean = (max_radius - min_radius) / 2
    total_area = math.pi*(mean**2)*max(colors_dict.values())
    color_dots = {color: scattered_random(
        {color: colors_dict[color]}, area_control=area_control,
        total_area=total_area,
        num_pixels=(width_per, num_pixels[1]), padding=padding,
        min_radius=min_radius, max_radius=max_radius, std=std)
        for color in colors_dict}
    dots = []
    if not color_order:
        colors = list(colors_dict.keys())
        random.shuffle(colors)
    else:
        colors = color_order
    for idx in range(len(colors)):
        dots.extend([dot._replace(x=dot.x + idx*width_per)
                     for dot in color_dots[colors[idx]]])
    return dots


# TODO: test the tuple num_pixels in the three following methods
# TODO: add area_control to these three methods a la scattered_random
def scattered_pairs(colors_dict, num_pixels=(256, 256), padding=16,
                    min_radius=1, max_radius=5):
    """Generates ScatteredPairs images: the dots are paired together, one of
    each type of color.  The remaining dots in the dominant color are randomly
    scattered.

    NOTE: `colors_dict` must have only two keys. """

    x_min, y_min = padding, padding
    x_max, y_max = num_pixels[0] - padding, num_pixels[1] - padding

    assert len(colors_dict) == 2
    sort_dict = sorted(colors_dict.items(), key=lambda pair: pair[1])
    num_leftover = sort_dict[1][1] - sort_dict[0][1]

    dots = []
    for _ in range(sort_dict[0][1]):
        new_pair_added = False
        while not new_pair_added:
            # one circle
            x1 = random.uniform(x_min, x_max)
            y1 = random.uniform(y_min, y_max)
            r1 = clip(random.gauss(3, 1), min_radius, max_radius)
            # get second dot coordinates
            r2 = clip(random.gauss(3, 1), min_radius, max_radius)
            theta = random.uniform(0, 2*math.pi)
            dist = r1 + r2
            tmp_x2, tmp_y2 = polar_to_cartesian(theta, dist)
            x2, y2 = tmp_x2 + x1, tmp_y2 + y1
            if no_overlap(dots, x1, y1, r1) and no_overlap(dots, x2, y2, r2):
                dots.append(Dot(x=x1, y=y1, radius=r1, color=sort_dict[0][0]))
                dots.append(Dot(x=x2, y=y2, radius=r2, color=sort_dict[1][0]))
                new_pair_added = True

    for _ in range(num_leftover):
        new_dot_added = False
        while not new_dot_added:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            r = clip(random.gauss(3, 1), min_radius, max_radius)
            # avoid overlap with existing circles
            if no_overlap(dots, x, y, r):
                dots.append(Dot(x=x, y=y, radius=r, color=sort_dict[1][0]))
                new_dot_added = True

    return dots


def column_pairs_mixed(colors_dict, num_pixels=(256, 256), pad=2.5,
                       min_radius=1, max_radius=5):
    """Generates ColumnPairsMixed images: the dots are paired in two columns,
    but which color is in which column depends on the row.

    NOTE: `colors_dict` must have only two keys. """
    assert len(colors_dict) == 2
    sort_dict = sorted(colors_dict.items(), key=lambda pair: pair[1])
    center = num_pixels[0] / 2
    x_vals = (center - max_radius - pad/2, center + max_radius + pad/2)
    y_step = max_radius + 2*pad
    y0 = center + int(sort_dict[1][1] / 2)*y_step
    num_lower = sort_dict[0][1]

    dots = []
    for row_num in range(sort_dict[1][1]):
        first_side = random.choice([0, 1])
        x1 = x_vals[first_side]
        y1 = y0 - row_num*y_step
        r1 = clip(random.gauss(3, 1), min_radius, max_radius)
        dots.append(Dot(x=x1, y=y1, radius=r1, color=sort_dict[1][0]))

        add_second = False
        if num_lower < sort_dict[1][1] - row_num:
            if random.random() < 0.5:
                add_second = True
        else:
            add_second = True

        if add_second:
            x2 = x_vals[0 if first_side == 1 else 1]
            y2 = y1
            r2 = clip(random.gauss(3, 1), min_radius, max_radius)
            dots.append(Dot(x=x2, y=y2, radius=r2, color=sort_dict[0][0]))
            num_lower -= 1

    return dots


def column_pairs_sorted(colors_dict, num_pixels=(256, 256), pad=2.5,
                        min_radius=1, max_radius=5):
    """Generates ColumnPairsSorted images: the dots are paired in two columns,
    with one color on the left and one on the right.

    NOTE: `colors_dict` must have only two keys. """
    assert len(colors_dict) == 2
    sort_dict = sorted(colors_dict.items(), key=lambda pair: pair[1])
    center = num_pixels[0] / 2
    x_vals = (center - max_radius - pad/2, center + max_radius + pad/2)
    y_step = max_radius + 2*pad
    y0 = center + int(sort_dict[1][1] / 2)*y_step
    num_lower = sort_dict[0][1]
    lower_color_side = random.choice([0, 1])
    higher_color_side = 1 if lower_color_side == 0 else 0

    dots = []
    for row_num in range(sort_dict[1][1]):
        x1 = x_vals[higher_color_side]
        y1 = y0 - row_num*y_step
        r1 = clip(random.gauss(3, 1), min_radius, max_radius)
        dots.append(Dot(x=x1, y=y1, radius=r1, color=sort_dict[1][0]))

        if row_num < num_lower:
            x2 = x_vals[lower_color_side]
            y2 = y1
            r2 = clip(random.gauss(3, 1), min_radius, max_radius)
            dots.append(Dot(x=x2, y=y2, radius=r2, color=sort_dict[0][0]))

    return dots


def make_image(file_name, dots, num_pixels=(256, 256)):
    """Make and save an image from a list of `Dot`s.

    Args:
        file_name: where to save the image
        dots: an iterable of `Dot`s to draw
        num_pixels: the saved image will be a square with num_pixels sides
    """
    # TODO: update doc-string for proper width-height understanding!

    plt.rcParams['axes.facecolor'] = 'grey'
    plt.rcParams['axes.linewidth'] = 0

    dpi = min(num_pixels)
    fig, ax = plt.subplots(dpi=dpi)
    fig.set_size_inches((max(num_pixels) / dpi, 1))

    ax.set_xlim((0, num_pixels[0]))
    ax.set_ylim((0, num_pixels[1]))
    for dot in dots:
        ax.add_patch(
            Circle((dot.x, dot.y), radius=dot.radius, fc=dot.color))

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0)
    # TODO: mkdir for file_name if not exists...
    fig.savefig(file_name, facecolor='grey', pad_inches=0)


def make_batch(trial_types, color_dicts, num_per_dict, out_dir='.',
               num_pixels=(256, 256), min_radius=1, max_radius=5, std=1,
               area_control=True):

    for trial_type in trial_types:
        image_method = globals()[trial_type]
        for dict_idx in range(len(color_dicts)):
            color_dict = color_dicts[dict_idx]
            for idx in range(num_per_dict):
                make_image(
                    '{}/{}_{}_{}_{}.png'.format(
                        out_dir, trial_type,
                        '_'.join([str(key) + str(color_dict[key])
                                  for key in color_dict]),
                        idx, dict_idx),
                    image_method(color_dict,
                                 num_pixels=num_pixels,
                                 min_radius=min_radius,
                                 max_radius=max_radius, std=std,
                                 area_control=area_control),
                    num_pixels=num_pixels)


def make_split_batch(color_dicts, num_per_dict, out_dir='.',
                     num_pixels=(800, 400), min_radius=1.5, max_radius=3.5,
                     std=0.5, area_control=True):
    for dict_idx in range(len(color_dicts)):
        color_dict = color_dicts[dict_idx]
        order = list(color_dict.keys())
        for idx in range(num_per_dict):
            # TODO: integrate this order parameter into make_batch instead of
            # having this as a separate method?
            # balance order by flipping every time
            order = list(reversed(order))
            make_image(
                '{}/{}_{}_{}_{}.png'.format(
                    out_dir, 'scattered_split',
                    '_'.join([str(key) + str(color_dict[key])
                              for key in color_dict]),
                    idx, dict_idx),
                scattered_split(color_dict,
                                num_pixels=num_pixels,
                                min_radius=min_radius,
                                max_radius=max_radius, std=std,
                                color_order=order, area_control=area_control),
                num_pixels=num_pixels)


def dicts_from_ratios(ratios, dicts_per_ratio,
                      colors=['b', 'y'], dot_range=(5, 25), multipliers=None):
    dicts = []
    assert multipliers is None or len(multipliers) == len(ratios)
    for idx in range(len(ratios)):
        ratio = ratios[idx]
        if not multipliers:
            # TODO: better method of generating ratios?
            mults = [n for n in range(1, 10)
                     if n*min(ratio) >= dot_range[0]
                     and n*max(ratio) <= dot_range[1]]
        else:
            mults = multipliers[idx]
        for idx in range(dicts_per_ratio):
            # reverse the ratio every trial to mix
            ratio = list(reversed(ratio))
            # mult = random.choice(multipliers)
            # TODO: make sure this logic works
            mult = mults[idx % len(mults)]
            dicts.append({colors[n]: mult*ratio[n] for n in range(len(colors))})
    return dicts


if __name__ == '__main__':

    ratios = [(n, n+1) for n in range(1, 10)]
    imgs_per_ratio = 100
    trial_types = ['scattered_random', 'scattered_pairs',
                   'column_pairs_mixed', 'column_pairs_sorted']
    color_dicts = dicts_from_ratios(ratios, imgs_per_ratio, dot_range=(5, 12))
    # TODO: refactor this to get proper # imgs per bin

    # make training set
    make_batch(trial_types, color_dicts, 5, 'images/small/train',
               num_pixels=128, min_radius=2, max_radius=4)
    # make val set
    make_batch(trial_types, color_dicts, 1, 'images/small/val',
               num_pixels=128, min_radius=2, max_radius=4)
    # make test set
    make_batch(trial_types, color_dicts, 1, 'images/small/test',
               num_pixels=128, min_radius=2, max_radius=4)
