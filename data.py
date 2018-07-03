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

# tf.enable_eager_execution()


def parse_file(filename, label, key, img_size, num_channels, grayscale=True):
    image_string = tf.read_file(filename)
    image = tf.to_float(
        tf.image.decode_png(image_string, channels=num_channels))
    if grayscale:
        image = tf.image.rgb_to_grayscale(image)
        num_channels = 1
    image.set_shape([img_size, img_size, num_channels])
    return {key: image, 'filename': filename}, label


def most(colors_dict, main_color):
    return int(colors_dict[main_color] >
               sum(colors_dict[color]
                   for color in colors_dict if color != main_color))


def most_blue(colors_dict):
    return most(colors_dict, 'b')


def extract_color_dict(filename, colors=['y', 'b']):
    strings = filename.split('_')
    return {s[0]: int(s[1:]) for s in strings if s[0] in colors}


def label_from_filename(filename, colors=['y', 'b'],
                        eval_fn=most_blue):
    colors_dict = extract_color_dict(filename, colors)
    return eval_fn(colors_dict)


def make_dataset(filename_pattern, img_feature_name, img_size, num_channels,
                 shuffle=True, batch_size=None, num_epochs=1):
    filenames = glob.glob(filename_pattern)
    labels = [label_from_filename(filename) for filename in filenames]
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # get image tensors
    dataset = dataset.map(lambda filename, label: parse_file(
        filename, label, img_feature_name, img_size, num_channels))
    # shuffle
    if shuffle:
        dataset = dataset.shuffle(len(filenames))
    # repeat for num epochs
    dataset = dataset.repeat(num_epochs)
    # batch
    if batch_size:
        dataset = dataset.batch(batch_size)
    else:
        # batch_size = None |--> one big batch
        dataset = dataset.batch(len(filenames))
    return dataset
