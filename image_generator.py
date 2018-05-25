from collections import namedtuple
import random
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

PI = 3.1415
Dot = namedtuple('Dot', ['x', 'y', 'radius', 'color'])


# TODO: area vs size control, i.e. smart radius
# TODO: other trial types
# TODO: smart file names
def scattered_random(colors_dict, num_pixels=256, padding=16):

    x_min, y_min = padding, padding
    x_max, y_max = num_pixels - padding, num_pixels - padding
    # TODO: avoid overlap?
    dots = []
    for color in colors_dict:
        for _ in range(colors_dict[color]):
            new_dot_added = False
            while not new_dot_added:
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                radius = random.gauss(5, 1)
                add_this = True
                for dot in dots:
                    if ((x - dot.x)**2 + (y - dot.y)**2 <
                            (radius + dot.radius)**2):
                        add_this = False
                if add_this:
                    dots.append(Dot(x=x,
                                    y=y,
                                    radius=radius,
                                    color=color))
                    new_dot_added = True
    return dots


def make_image(file_name, dots,
               num_pixels=256):

    plt.rcParams['axes.facecolor'] = 'grey'
    plt.rcParams['axes.linewidth'] = 0

    fig, ax = plt.subplots(figsize=(1,1), dpi=num_pixels)

    ax.set_xlim((0, num_pixels))
    ax.set_ylim((0, num_pixels))
    for dot in dots:
        ax.add_patch(
            Circle((dot.x, dot.y), radius=dot.radius, fc=dot.color))

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0)
    fig.savefig(file_name, facecolor='grey', pad_inches=0)


if __name__ == '__main__':

    test_image = [
        Dot(x=0, y=0, radius=2, color='y'),
        Dot(x=20, y=53, radius=1.5, color='b'),
        Dot(x=67, y=22, radius=4, color='y')
    ]

    make_image('test.png', scattered_random({'y': 10, 'b': 9}))
