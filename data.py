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
"""
# TODO: doc module
import glob
import tensorflow as tf

tf.enable_eager_execution()


def parse_file(filename, label):
    print(filename)
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    # label = label_from_filename(filename)
    return image, label


def most_blue_not_yellow(colors_dict):
    return int(colors_dict['b'] > colors_dict['y'])


def label_from_filename(filename, colors=['y', 'b'],
                        eval_fn=most_blue_not_yellow):
    strings = filename.split('_')
    colors_dict = {s[0]: int(s[1:]) for s in strings if s[0] in colors}
    return eval_fn(colors_dict)


filenames = glob.glob('*.png')
labels = [label_from_filename(filename) for filename in filenames]
print(labels)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(parse_file)

# TODO: shuffle, etc
iterator = dataset.make_one_shot_iterator()

print(iterator.get_next())
