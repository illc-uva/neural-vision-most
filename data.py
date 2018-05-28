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
import tensorflow as tf

tf.enable_eager_execution()

filenames = tf.matching_files('*.png')
dataset = tf.data.Dataset.from_tensor_slices(filenames)
# TODO: include (filenames, labels) above


# TODO: label as argument here; or generate label based on filename?...
def parse_file(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    tf.to_float(image)
    return image


dataset = dataset.map(parse_file)

# TODO: shuffle, etc
iterator = dataset.make_one_shot_iterator()

print(iterator.get_next())
