import numpy as np
# import skimage
from skimage import morphology, measure, filters
from scipy import ndimage
# import scipy


def crop(image, shape):
    input_shape = image.shape
    n_dim = len(shape)
    assert n_dim == len(input_shape)
    slices = []
    for i in range(n_dim):
        if input_shape[i] >= shape[i]:
            start = int(input_shape[i] - shape[i]) / 2
            stop = start + shape[i]
        else:
            start = 0
            stop = input_shape[i]
        slices.append(slice(start, stop))
    return image[slices]


def resize(image, new_shape):
    #print 'start ', image.shape
    resize_factor = (np.round(new_shape).astype(np.float) + 1e-3) / image.shape  # +1e-3 to suppress warning of scipy.ndimage.zoom
    new_image = ndimage.interpolation.zoom(image, resize_factor, order=2)
    #print 'end ', new_image.shape
    return new_image


def resample(image, spacing, new_spacing):
    # print 'start ', image.shape
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor) + 1e-3
    real_resize_factor = new_shape / image.shape
    real_new_spacing = spacing / real_resize_factor
    new_image = ndimage.interpolation.zoom(image, real_resize_factor, order=2)
    #print 'end ', new_image.shape
    return new_image, real_new_spacing


def denoise(image):
    image = image
    # image = scipy.ndimage.median_filter(image, 2)
    selem = morphology.ball(1)
    image = morphology.closing(image, morphology.ball(1))
    image = morphology.opening(image, selem)
    image = filters.gaussian(image, sigma=0.4, multichannel=False)
    return image


def get_dilated_mask(mask, pixels):
    return morphology.dilation(mask, morphology.ball(pixels))


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


# def get_bladder_mask_random_walker(image, threshold_low=0.0016, threshold_high=0.0020):
#     markers = np.zeros(image.shape)
#     markers[image > threshold_high] = 1
#     markers[image < threshold_low] = 2
#     mask = segmentation.random_walker(image, markers)
#     mask = (mask == 1)
#
#     selem = morphology.ball(2)
#     mask = morphology.closing(morphology.opening(mask, selem), selem)
#
#     labels = measure.label(mask, connectivity=1)
#     l_max = largest_label_volume(labels, bg=0)
#     if l_max is not None:
#         mask[labels != l_max] = 0
#     # for i in range(mask.shape[0]):
#     #     mask[i] = morphology.convex_hull_object(np.ascontiguousarray(mask[i]))
#     return mask.astype(int), markers


def get_bladder_threshold(image, voxel_volumn, min_bladder_volumn, scale):
    kth = - int(min_bladder_volumn / voxel_volumn)
    part = np.partition(image, kth, None)[kth:]
    part = np.sort(part)
    threshold = np.average(part[:len(part)/10]) * scale
    return threshold


def postprocess_bladder_mask(mask):
    selem = morphology.ball(3)
    mask = morphology.opening(mask, selem)
    labels = measure.label(mask)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:
        mask[labels != l_max] = 0

    mask = morphology.closing(mask, selem)
    return mask


def calculate_dwi(adc, ref_dwi, ref_b, target_b):
    # adc = [ln(dwi_1) - ln(dwi_0)] / (b0 - b1)
    # ln(dwi_1) = adc * (b0 - b1) + ln(dwi_0)
    # dwi_1 = exp(adc * (b0 - b1) + ln(dwi_0))
    target_dwi = np.exp(adc * (ref_b - target_b) + np.log(ref_dwi))
    return target_dwi


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def linear_regression_with_single_variable(x, y, axis=0):
    n = x.shape[axis]
    x_mean = x.mean(axis=axis)
    y_mean = y.mean(axis=axis)
    frac0 = np.sum(x * y, axis=axis) - n * x_mean * y_mean
    frac1 = np.sum(x * x, axis=axis) - n * x_mean * x_mean
    w = frac0 / frac1
    b = y_mean - x_mean * w
    return w, b


def dwi2adc(dwi_dict):
    # adc = [ln(dwi_1) - ln(dwi_0)] / (b0 - b1)
    assert len(dwi_dict) > 1
    [b_values, dwis] = zip(*dwi_dict.items())
    if len(dwi_dict) > 2:
        b_value_array = np.array(b_values).reshape((-1, 1, 1, 1))
        ln_dwi_array = np.log(np.stack(dwis))
        adc, _ = linear_regression_with_single_variable(b_value_array, ln_dwi_array)
        adc = -adc
    else:
        adc = np.log(dwis[1].astype(np.float) / dwis[0]) / (b_values[0] - b_values[1])
    return adc


def find_bounding_box(a):
    objects = list(ndimage.find_objects(a))
    assert len(objects) == 1
    return objects[0]
