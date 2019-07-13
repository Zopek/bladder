import numpy as np
import scipy.misc
import scipy.ndimage as ndi

rotation_range = 45.0  # randomly rotate images in the range (degrees, 0 to 180)
width_shift_range = 0.1  # randomly shift images horizontally (fraction of total width)
height_shift_range = 0.1  # randomly shift images vertically (fraction of total height)
horizontal_flip = False  # randomly flip images
vertical_flip = True
row_index = 2
col_index = 3
channel_index = 1
shear_range = 0.1
zoom_range = 0.1
fill_mode = 'reflect'
cval = 0.

if np.isscalar(zoom_range):
    zoom_range = [1 - zoom_range, 1 + zoom_range]
elif len(zoom_range) == 2:
    zoom_range = [zoom_range[0], zoom_range[1]]
else:
    raise Exception('zoom_range should be a float or '
                    'a tuple or list of two floats. '
                    'Received arg: ', zoom_range)


def random_transform(x, label=None):
    # x is a single image, so it doesn't have image number at index 0
    img_row_index = row_index - 1
    img_col_index = col_index - 1
    img_channel_index = channel_index - 1

    # use composition of homographies to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_index]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if shear_range:
        shear = np.random.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    h, w = x.shape[img_row_index], x.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, img_channel_index,
                        fill_mode=fill_mode, cval=cval)
    if label is not None:
        label = apply_transform(label, transform_matrix, img_channel_index, fill_mode=fill_mode, cval=cval)

    if horizontal_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_index)
            if label is not None:
                label = flip_axis(label, img_col_index)

    if vertical_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_row_index)
            if label is not None:
                label = flip_axis(label, img_row_index)

    return x, label


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=1, mode=fill_mode, cval=cval) for x_channel
                      in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def zerosquare_func(x, sizeh=50, sizew=50, intensity=0.0, channel=0):
    # print(x.shape)
    up_bound = np.random.choice(224 - sizeh)  # zero out square
    right_bound = np.random.choice(224 - sizew)
    x[up_bound:(up_bound + sizeh), right_bound:(right_bound + sizew), channel] = intensity
    return x
