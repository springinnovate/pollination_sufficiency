"""Map people fed equivalents back to ESA habitat."""
import argparse
import os
import logging
import sys

import scipy
import numpy
from osgeo import osr
from osgeo import gdal
import pygeoprocessing
import taskgraph
import ecoshard

WORKSPACE_DIR = 'workspace_realized_pollination'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
TARGET_NODATA = -1

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('pygeoprocessing').setLevel(logging.INFO)
logging.getLogger('ecoshard').setLevel(logging.INFO)


def _mask_op(base_array, mask_array):
    result = numpy.copy(base_array)
    # ensure mask that is close to 0 is 0.
    result[numpy.isclose(result, 0)] = 0
    result[mask_array != 1] = -1  # -1 is nodata
    return result


def _nodata_to_zero_op(base_array, base_nodata):
    """Convert nodata to zero."""
    result = numpy.copy(base_array)
    result[numpy.isclose(base_array, base_nodata)] = 0.0
    return result


def _make_mask_op(base_array, base_nodata):
    result = numpy.full(base_array.shape, False, dtype=bool)
    result[~numpy.isclose(base_array, base_nodata)] = True
    return result


def _div_op(num, denom):
    result = numpy.full(num.shape, -1, dtype=numpy.float32)
    valid_mask = (num > 0) & (denom > 0)
    result[valid_mask] = num[valid_mask] / denom[valid_mask]
    result[numpy.isclose(num, 0)] = 0
    return result


def compress_and_build_overviews(base_raster_path, target_raster_path):
    ecoshard.compress_raster(
        base_raster_path, target_raster_path)
    ecoshard.build_overviews(target_raster_path)


def _align_and_adjust_area(
        base_raster_path, value_raster_path, target_raster_path):
    """Target the same size as base with value in it adjusted for pixel."""
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    target_pixel_size = base_raster_info['pixel_size']
    pygeoprocessing.warp_raster(
        value_raster_path, target_pixel_size, target_raster_path,
        'average', target_bb=base_raster_info['bounding_box'])


def norm_by_hab_pixels(
        ppl_fed_raster_path,
        hab_mask_raster_path,
        kernel_raster_path,
        ppl_fed_div_hab_pixels_raster_path,
        norm_ppl_fed_within_2km_pixels_raster_path):
    # calculate count of hab pixels within 2km.
    hab_pixels_within_2km_raster_path = os.path.join(
        CHURN_DIR, 'hab_pixels_within_2km.tif')
    if not os.path.exists(hab_pixels_within_2km_raster_path):
        pygeoprocessing.convolve_2d(
            (hab_mask_raster_path, 1), (kernel_raster_path, 1),
            hab_pixels_within_2km_raster_path,
            working_dir=CHURN_DIR,
            mask_nodata=False,
            ignore_nodata_and_edges=False,
            normalize_kernel=False)
    pygeoprocessing.raster_calculator(
        [(ppl_fed_raster_path, 1), (hab_pixels_within_2km_raster_path, 1)],
        _div_op, ppl_fed_div_hab_pixels_raster_path, gdal.GDT_Float32, -1)

    pygeoprocessing.convolve_2d(
        (ppl_fed_div_hab_pixels_raster_path, 1), (kernel_raster_path, 1),
        norm_ppl_fed_within_2km_pixels_raster_path,
        working_dir=CHURN_DIR,
        mask_nodata=False,
        ignore_nodata_and_edges=False,
        normalize_kernel=False)


def main():
    """Entry point."""
    for dir_path in [WORKSPACE_DIR, CHURN_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    parser = argparse.ArgumentParser(description='Realized Pollination')
    parser.add_argument(
        '--ppl_fed_path', required=True, type=str,
        help='Path to people fed raster')
    parser.add_argument(
        '--hab_mask', required=True, type=str,
        help='Path to habitat mask')
    args = parser.parse_args()
    task_graph = taskgraph.TaskGraph(CHURN_DIR, 4, 5.0)
    kernel_raster_path = os.path.join(CHURN_DIR, 'radial_kernel.tif')
    kernel_task = task_graph.add_task(
        func=create_flat_radial_convolution_mask,
        args=(0.00277778, 2000., kernel_raster_path),
        target_path_list=[kernel_raster_path],
        task_name='make convolution kernel')

    aligned_ppl_fed_raster_path = (
        '%s_aligned%s' % os.path.splitext(args['ppl_fed_path']))
    align_ppl_fed_per_pixel_task = task_graph.add_task(
        func=_align_and_adjust_area,
        args=(
            args['hab_mask'],
            args['ppl_fed_path'],
            aligned_ppl_fed_raster_path),
        target_path_list=[aligned_ppl_fed_raster_path],
        task_name=f'align and adjust area for {aligned_ppl_fed_raster_path}')

    # calculate extent of ppl fed by 2km.
    ppl_fed_per_pixel_raster_path = os.path.join(
        CHURN_DIR, 'ppl_fed_per_pixel.tif')
    ppl_fed_per_pixel_task = task_graph.add_task(
        func=pygeoprocessing.convolve_2d,
        args=[
            (aligned_ppl_fed_raster_path, 1), (kernel_raster_path, 1),
            ppl_fed_per_pixel_raster_path],
        kwargs={
            'working_dir': CHURN_DIR,
            'mask_nodata': False,
            'ignore_nodata_and_edges': False,
            'normalize_kernel': False,
            },
        dependent_task_list=[kernel_task, align_ppl_fed_per_pixel_task],
        target_path_list=[ppl_fed_per_pixel_raster_path],
        task_name=(
            'calc people fed reach'
            f' {os.path.basename(ppl_fed_per_pixel_raster_path)}'))

    ppl_fed_div_hab_pixels_raster_path = os.path.join(
        CHURN_DIR, 'ppl_fed_div_hab_pixels_in_2km.tif')
    norm_ppl_fed_within_2km_pixels_raster_path = os.path.join(
        CHURN_DIR, 'norm_ppl_fed_within_2km_per_pixel.tif')
    norm_by_hab_pixel_task = task_graph.add_task(
        func=norm_by_hab_pixels,
        args=(
            aligned_ppl_fed_raster_path,
            args['hab_mask'],
            kernel_raster_path,
            ppl_fed_div_hab_pixels_raster_path,
            norm_ppl_fed_within_2km_pixels_raster_path),
        dependent_task_list=[
            align_ppl_fed_per_pixel_task, kernel_task],
        target_path_list=[
            ppl_fed_div_hab_pixels_raster_path,
            norm_ppl_fed_within_2km_pixels_raster_path],
        task_name='calc ppl fed div hab pixels')

    # mask to hab
    ppl_fed_coverage_mask_to_hab_raster_path = (
        '%s_mask_to_hab%s' % os.path.splitext(
            ppl_fed_per_pixel_raster_path))
    mask_ppl_fed_coverage_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(ppl_fed_per_pixel_raster_path, 1),
             (args['hab_mask'], 1)],
            _mask_op, ppl_fed_coverage_mask_to_hab_raster_path,
            gdal.GDT_Float32, -1),
        dependent_task_list=[ppl_fed_per_pixel_task],
        target_path_list=[ppl_fed_coverage_mask_to_hab_raster_path],
        task_name='mask ppl fed mask')

    norm_ppl_fed_coverage_mask_to_hab_raster_path = (
        '%s_mask_to_hab%s' % os.path.splitext(
            norm_ppl_fed_within_2km_pixels_raster_path))
    mask_normalized_ppl_fed_per_pixel_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(norm_ppl_fed_within_2km_pixels_raster_path, 1),
             (args['hab_mask'], 1)],
            _mask_op, norm_ppl_fed_coverage_mask_to_hab_raster_path,
            gdal.GDT_Float32, -1),
        dependent_task_list=[norm_by_hab_pixel_task],
        target_path_list=[norm_ppl_fed_coverage_mask_to_hab_raster_path],
        task_name='mask normalized ppl fed mask')

    # compress and ecoshard result
    compressed_ppl_fed_coverage_mask_to_hab_raster_path = os.path.join(
        WORKSPACE_DIR, os.path.basename('%s_compressed%s' % os.path.splitext(
            ppl_fed_coverage_mask_to_hab_raster_path)))
    compressed_norm_ppl_fed_coverage_mask_to_hab_raster_path = os.path.join(
        WORKSPACE_DIR, os.path.basename('%s_compressed%s' % os.path.splitext(
            norm_ppl_fed_coverage_mask_to_hab_raster_path)))

    task_graph.add_task(
        func=compress_and_build_overviews,
        args=(
            ppl_fed_coverage_mask_to_hab_raster_path,
            compressed_ppl_fed_coverage_mask_to_hab_raster_path),
        target_path_list=[compressed_ppl_fed_coverage_mask_to_hab_raster_path],
        dependent_task_list=[mask_ppl_fed_coverage_task],
        task_name=f'''compressing {
            compressed_ppl_fed_coverage_mask_to_hab_raster_path}''')
    task_graph.add_task(
        func=compress_and_build_overviews,
        args=(
            norm_ppl_fed_coverage_mask_to_hab_raster_path,
            compressed_norm_ppl_fed_coverage_mask_to_hab_raster_path),
        target_path_list=[
            compressed_norm_ppl_fed_coverage_mask_to_hab_raster_path],
        dependent_task_list=[mask_normalized_ppl_fed_per_pixel_task],
        task_name=f'''compressing {
            compressed_norm_ppl_fed_coverage_mask_to_hab_raster_path}''')

    task_graph.join()
    task_graph.close()


def create_flat_radial_convolution_mask(
        pixel_size_degree, radius_meters, kernel_filepath):
    """Create a radial mask to sample pixels in convolution filter.

    Parameters:
        pixel_size_degree (float): size of pixel in degrees.
        radius_meters (float): desired size of radial mask in meters.

    Returns:
        A 2D numpy array that can be used in a convolution to aggregate a
        raster while accounting for partial coverage of the circle on the
        edges of the pixel.

    """
    degree_len_0 = 110574  # length at 0 degrees
    degree_len_60 = 111412  # length at 60 degrees
    pixel_size_m = pixel_size_degree * (degree_len_0 + degree_len_60) / 2.0
    pixel_radius = numpy.ceil(radius_meters / pixel_size_m)
    n_pixels = (int(pixel_radius) * 2 + 1)
    sample_pixels = 200
    mask = numpy.ones((sample_pixels * n_pixels, sample_pixels * n_pixels))
    mask[mask.shape[0]//2, mask.shape[0]//2] = 0
    distance_transform = scipy.ndimage.morphology.distance_transform_edt(mask)
    mask = None
    stratified_distance = distance_transform * pixel_size_m / sample_pixels
    distance_transform = None
    in_circle = numpy.where(stratified_distance <= 2000.0, 1.0, 0.0)
    stratified_distance = None
    reshaped = in_circle.reshape(
        in_circle.shape[0] // sample_pixels, sample_pixels,
        in_circle.shape[1] // sample_pixels, sample_pixels)
    kernel_array = numpy.sum(reshaped, axis=(1, 3)) / sample_pixels**2
    normalized_kernel_array = kernel_array / numpy.max(kernel_array)
    LOGGER.debug(normalized_kernel_array)
    reshaped = None

    driver = gdal.GetDriverByName('GTiff')
    kernel_raster = driver.Create(
        kernel_filepath.encode('utf-8'), n_pixels, n_pixels, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_raster.SetGeoTransform([-180, 1, 0, 90, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    kernel_raster.SetProjection(srs.ExportToWkt())
    kernel_band = kernel_raster.GetRasterBand(1)
    kernel_band.SetNoDataValue(TARGET_NODATA)
    kernel_band.WriteArray(normalized_kernel_array)


if __name__ == '__main__':
    main()
