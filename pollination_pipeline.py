"""
This is a combination of make_poll_suff, calc_people_fed, and realized_pollination

Input:
    pollination dependent micronutrient production in units of equiv people fed per hectare (should be converted to "production" which is "people")
    biophyiscal table


New Calculations:
    'expression': 'raster1*raster2*raster3*(raster4>0)+(raster4<1)*-9999',
           'raster1': "monfreda_2008_yield_poll_dep_ppl_fed_5min.tif", #https://storage.googleapis.com/critical-natural-capital-ecoshards/monfreda_2008_yield_poll_dep_ppl_fed_5min.tif
           'raster2': r"workspace_poll_suff\churn\poll_suff_hab_ag_coverage_rasters\poll_suff_ag_coverage_prop_10s_forest_conversion_2050_md5_abda51.tif",
           'raster3': "esa_pixel_area_ha_md5_1dd3298a7c4d25c891a11e01868b5db6.tif", # https://storage.googleapis.com/ecoshard-root/esa_pixel_area_ha_md5_1dd3298a7c4d25c891a11e01868b5db6.tif
           'raster4': r"workspace_poll_suff\churn\ag_mask\forest_conversion_2050_md5_abda51_ag_mask.tif",

    Outputs:
        pollination_ppl_fed_on_ag_10s_forest_conversion_2050


#python realized_pollination.py --ppl_fed_path pollination_ppl_fed_on_ag_10s_forest_conversion_2050 --hab_mask_path "workspace_poll_suff\churn\hab_mask\forest_conversion_2050_md5_abda51_hab_mask.tif"


Expected outputs include:
    poll_suff_hab_ag_coverage_rasters

    [input landcover name]_ag_mask
    [input landcover name]_hab_mask

    pollination_ppl_fed_on_[input landcover name]

    "realized_pollination output" (this is the output of `realized_pollination.py`)
        * people fed
        * norm people fed


Pollination sufficiency analysis. This is based off the IPBES-Pollination
project so that we can run on any new LULC scenarios with ESA classification.
Used to be called dasgupta_agriculture.py but then we did it for more than just Dasgupta
"""
import argparse
import configparser
import glob
import logging
import multiprocessing
import os
import requests

from ecoshard import taskgraph
from osgeo import gdal
from osgeo import osr
import ecoshard
from ecoshard import geoprocessing
import numpy
import pandas
import scipy.ndimage.morphology

# set a limit for the cache
gdal.SetCacheMax(2**28)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger('pollination')
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)

_MULT_NODATA = -1
_MASK_NODATA = 2

WORKING_DIR = './workspace_poll_suff'
ECOSHARD_DIR = os.path.join(WORKING_DIR, 'ecoshard_dir')
CHURN_DIR = os.path.join(WORKING_DIR, 'churn')

for dir_path in [WORKING_DIR, ECOSHARD_DIR, CHURN_DIR]:
    os.makedirs(dir_path, exist_ok=True)

NODATA = -9999
N_WORKERS = max(1, multiprocessing.cpu_count())


def calculate_poll_suff(
        task_graph, landcover_path, ag_codes, natural_codes):
    """Calculate values for a given landcover.
    Parameters:
        task_graph (taskgraph.TaskGraph): taskgraph object used to schedule
            work.
        landcover_path (str): path to a landcover map with globio style
            landcover codes.
        ag_codes/natural_codes (list): list of single integer or tuple
            (min, max) ranges to treat as ag or natural codes.

    Returns:
        (path to pollination sufficiency ag coverage, path to ag mask)
    """
    landcover_key = os.path.splitext(os.path.basename(landcover_path))[0]
    output_dir = os.path.join(WORKING_DIR, landcover_key)

    # The proportional area of natural within 2 km was calculated for every
    #  pixel of agricultural land (GLOBIO land-cover classes 2, 230, 231, and
    #  232) at 10 arc seconds (~300 m) resolution. This 2 km scale represents
    #  the distance most commonly found to be predictive of pollination
    #  services (Kennedy et al. 2013).
    kernel_raster_path = os.path.join(CHURN_DIR, 'radial_kernel.tif')
    kernel_task = task_graph.add_task(
        func=create_radial_convolution_mask,
        args=(0.00277778, 2000., kernel_raster_path),
        target_path_list=[kernel_raster_path],
        task_name='make convolution kernel')

    # This loop is so we don't duplicate code for each mask type with the
    # only difference being the lulc codes and prefix
    mask_task_path_map = {}
    for mask_prefix, globio_codes in [
            ('ag', ag_codes), ('hab', natural_codes)]:
        mask_key = f'{landcover_key}_{mask_prefix}_mask'
        mask_target_path = os.path.join(
            CHURN_DIR, f'{mask_prefix}_mask',
            f'{mask_key}.tif')
        mask_task = task_graph.add_task(
            func=mask_raster,
            args=(landcover_path, globio_codes, mask_target_path),
            target_path_list=[mask_target_path],
            task_name=f'mask {mask_key}',)

        mask_task_path_map[mask_prefix] = (mask_task, mask_target_path)

    pollhab_2km_prop_path = os.path.join(
        CHURN_DIR, 'pollhab_2km_prop',
        f'pollhab_2km_prop_{landcover_key}.tif')
    pollhab_2km_prop_task = task_graph.add_task(
        func=geoprocessing.convolve_2d,
        args=[
            (mask_task_path_map['hab'][1], 1), (kernel_raster_path, 1),
            pollhab_2km_prop_path],
        kwargs={
            'working_dir': CHURN_DIR,
            'ignore_nodata_and_edges': True},
        dependent_task_list=[mask_task_path_map['hab'][0], kernel_task],
        target_path_list=[pollhab_2km_prop_path],
        task_name=(
            'calculate proportional'
            f' {os.path.basename(pollhab_2km_prop_path)}'))

    # calculate pollhab_2km_prop_on_ag_10s by multiplying pollhab_2km_prop
    # by the ag mask
    pollhab_2km_prop_on_ag_path = os.path.join(
        output_dir, f'''pollhab_2km_prop_on_ag_10s_{
            landcover_key}.tif''')
    pollhab_2km_prop_on_ag_task = task_graph.add_task(
        func=mult_rasters,
        args=(
            mask_task_path_map['ag'][1], pollhab_2km_prop_path,
            pollhab_2km_prop_on_ag_path),
        target_path_list=[pollhab_2km_prop_on_ag_path],
        dependent_task_list=[
            pollhab_2km_prop_task, mask_task_path_map['ag'][0]],
        task_name=(
            f'''pollhab 2km prop on ag {
                os.path.basename(pollhab_2km_prop_on_ag_path)}'''))

    #  1.1.4.  Sufficiency threshold A threshold of 0.3 was set to
    #  evaluate whether there was sufficient pollinator habitat in the 2
    #  km around farmland to provide pollination services, based on
    #  Kremen et al.'s (2005)  estimate of the area requirements for
    #  achieving full pollination. This produced a map of wild
    #  pollination sufficiency where every agricultural pixel was
    #  designated in a binary fashion: 0 if proportional area of habitat
    #  was less than 0.3; 1 if greater than 0.3. Maps of pollination
    #  sufficiency can be found at (permanent link to output), outputs
    #  "poll_suff_..." below.

    threshold_val = 0.3
    pollinator_suff_hab_path = os.path.join(
        CHURN_DIR, 'poll_suff_hab_ag_coverage_rasters',
        f'poll_suff_ag_coverage_prop_10s_{landcover_key}.tif')
    poll_suff_task = task_graph.add_task(
        func=threshold_select_raster,
        args=(
            pollhab_2km_prop_path,
            mask_task_path_map['ag'][1], threshold_val,
            pollinator_suff_hab_path),
        target_path_list=[pollinator_suff_hab_path],
        dependent_task_list=[
            pollhab_2km_prop_task, mask_task_path_map['ag'][0]],
        task_name=f"""poll_suff_ag_coverage_prop {
            os.path.basename(pollinator_suff_hab_path)}""")

    task_graph.join()
    return pollinator_suff_hab_path, mask_task_path_map['ag'][1]


def create_radial_convolution_mask(
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
    normalized_kernel_array = kernel_array / numpy.sum(kernel_array)
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
    kernel_band.SetNoDataValue(NODATA)
    kernel_band.WriteArray(normalized_kernel_array)


def threshold_select_raster(
        base_raster_path, select_raster_path, threshold_val, target_path):
    """Select `select` if `base` >= `threshold_val`.
    Parameters:
        base_raster_path (string): path to single band raster that will be
            used to determine the threshold mask to select from
            `select_raster_path`.
        select_raster_path (string): path to single band raster to pass
            through to target if aligned `base` pixel is >= `threshold_val`
            0 otherwise, or nodata if base == nodata. Must be the same
            shape as `base_raster_path`.
        threshold_val (numeric): value to use as threshold cutoff
        target_path (string): path to desired output raster, raster is a
            byte type with same dimensions and projection as
            `base_raster_path`. A pixel in this raster will be `select` if
            the corresponding pixel in `base_raster_path` is >=
            `threshold_val`, 0 otherwise or nodata if `base` == nodata.
    Returns:
        None.
    """
    base_nodata = geoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    target_nodata = -9999.

    def threshold_select_op(
            base_array, select_array, threshold_val, base_nodata,
            target_nodata):
        result = numpy.empty(select_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        valid_mask = (base_array != base_nodata) & (
            select_array >= 0) & (select_array <= 1)
        result[valid_mask] = select_array[valid_mask] * numpy.interp(
            base_array[valid_mask], [0, threshold_val], [0.0, 1.0], 0, 1)
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1), (select_raster_path, 1),
         (threshold_val, 'raw'), (base_nodata, 'raw'),
         (target_nodata, 'raw')], threshold_select_op,
        target_path, gdal.GDT_Float32, target_nodata)


def mask_raster(base_path, codes, target_path):
    """Mask `base_path` to 1 where values are in codes. 0 otherwise.
    Parameters:
        base_path (string): path to single band integer raster.
        codes (list): list of integer or tuple integer pairs. Membership in
            `codes` or within the inclusive range of a tuple in `codes`
            is sufficient to mask the corresponding raster integer value
            in `base_path` to 1 for `target_path`.
        target_path (string): path to desired mask raster. Any corresponding
            pixels in `base_path` that also match a value or range in
            `codes` will be masked to 1 in `target_path`. All other values
            are 0.
    Returns:
        None.
    """
    code_list = numpy.array([
        item for sublist in [
            range(x[0], x[1]+1) if isinstance(x, tuple) else [x]
            for x in codes] for item in sublist])
    LOGGER.debug(f'expanded code array {code_list}')

    base_nodata = geoprocessing.get_raster_info(base_path)['nodata'][0]

    def mask_codes_op(base_array, codes_array):
        """Return a bool raster if value in base_array is in codes_array."""
        result = numpy.empty(base_array.shape, dtype=numpy.int8)
        result[:] = _MASK_NODATA
        valid_mask = base_array != base_nodata
        result[valid_mask] = numpy.isin(
            base_array[valid_mask], codes_array)
        return result

    geoprocessing.raster_calculator(
        [(base_path, 1), (code_list, 'raw')], mask_codes_op, target_path,
        gdal.GDT_Byte, 2)


def area_of_pixel_in_ha(pixel_size, center_lat):
    """Calculate Ha area of a wgs84 square pixel.
    Adapted from: https://gis.stackexchange.com/a/127327/2397
    Parameters:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.
    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.
    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = numpy.sqrt(1-(b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*numpy.sin(numpy.radians(f))
        zp = 1 + e*numpy.sin(numpy.radians(f))
        area_list.append(
            numpy.pi * b**2 * (
                numpy.log(zp/zm) / (2*e) +
                numpy.sin(numpy.radians(f)) / (zp*zm)))
    return pixel_size / 360. * (area_list[0]-area_list[1]) / 100**2


def _mult_raster_op(array_a, array_b, nodata_a, nodata_b, target_nodata):
    """Multiply a by b and skip nodata."""
    result = numpy.empty(array_a.shape, dtype=numpy.float32)
    result[:] = target_nodata
    valid_mask = (array_a != nodata_a) & (array_b != nodata_b)
    result[valid_mask] = array_a[valid_mask] * array_b[valid_mask]
    return result


def mult_rasters(raster_a_path, raster_b_path, target_path):
    """Multiply a by b and skip nodata."""
    raster_info_a = geoprocessing.get_raster_info(raster_a_path)
    raster_info_b = geoprocessing.get_raster_info(raster_b_path)

    nodata_a = raster_info_a['nodata'][0]
    nodata_b = raster_info_b['nodata'][0]

    if raster_info_a['raster_size'] != raster_info_b['raster_size']:
        aligned_raster_a_path = os.path.join(
            CHURN_DIR, '%s_aligned%s' % os.path.splitext(os.path.basename(raster_a_path)))
        aligned_raster_b_path = os.path.join(
            CHURN_DIR, '%s_aligned%s' % os.path.splitext(os.path.basename(raster_b_path)))
        geoprocessing.align_and_resize_raster_stack(
            [raster_a_path, raster_b_path],
            [aligned_raster_a_path, aligned_raster_b_path],
            ['near'] * 2, raster_info_a['pixel_size'], 'intersection')
        raster_a_path = aligned_raster_a_path
        raster_b_path = aligned_raster_b_path

    geoprocessing.raster_calculator(
        [(raster_a_path, 1), (raster_b_path, 1), (nodata_a, 'raw'),
         (nodata_b, 'raw'), (_MULT_NODATA, 'raw')], _mult_raster_op,
        target_path, gdal.GDT_Float32, _MULT_NODATA)


EXPECTED_INI_KEYS = ['LANDCOVER', 'POTENTIAL_PEOPLE_FED', 'AG_CODES', 'NATURAL_CODES', 'CALC_PEOPLE_FED', 'CALC_MAP_TO_HABITAT']
FILE_INI_KEYS = ['LANDCOVER', 'POTENTIAL_PEOPLE_FED']
EVAL_KEYS = ['AG_CODES', 'NATURAL_CODES']
BOOL_KEYS = ['CALC_PEOPLE_FED', 'CALC_MAP_TO_HABITAT']


def _validate_config_files(configuration_pattern_list):
    """Raise a value error if anything is amiss.

    Returns:
        list of valid pollination pipeline key to value map configs

    """
    configuration_path_list = [
        path for pattern in configuration_pattern_list
        for path in glob.glob(pattern)]

    section_to_config_file = {}
    config_list = []
    for configuration_path in configuration_path_list:
        config = configparser.ConfigParser(allow_no_value=True)
        section = os.path.basename(os.path.splitext(configuration_path)[0]).lower()
        config.read(configuration_path)
        if section not in config:
            raise ValueError(
                f'expected section {section} in {configuration_path} but '
                f'only saw {list(config)}')
        missing_keys = []
        key_to_value_map = {}
        for expected_key in EXPECTED_INI_KEYS:
            if expected_key not in config[section]:
                missing_keys.add(expected_key)
            else:
                key_to_value_map[expected_key] = config[section][expected_key]
        if missing_keys:
            raise ValueError(
                f'expected the following keys in {configuration_path} but '
                'were not found: ' + ', '.join(missing_keys))

        if section in section_to_config_file:
            raise ValueError(
                f'already processed a configuration called {section} in '
                f'{section_to_config_file[section]} saw another one with '
                f'{configuration_path}')
        section_to_config_file[section] = configuration_path
        config_list.append((section, key_to_value_map))

    return config_list


def _fetch_and_process_config_to_param_map(scenario_config, task_graph):
    """Convert the keys in config to files or valid elements.

    Identifies remote files and stores locally if needed.

    Returns:
        dictionary mapping pollination ini keys to paths to local files or
            valid parameters.
    """
    processed_params = {}
    for file_key in FILE_INI_KEYS:
        url = scenario_config[file_key]
        # initally guess that url is a file
        processed_params[file_key] = url

        if url.startswith('http'):
            target_path = os.path.join(ECOSHARD_DIR, os.path.basename(url))
            # redirect to local target that will be downloaded
            processed_params[file_key] = target_path
            response = requests.head(url)
            if response:
                task_graph.add_task(
                    func=ecoshard.download_url,
                    args=(url, target_path),
                    target_path_list=[target_path],
                    task_name=f'download {url}')
            else:
                raise ValueError(f'{url} does not refer to a valid url got response {response}')
        else:
            if not os.path.exists(url):
                raise ValueError(
                    f'expected an existing file at {url} but not found')

    for eval_key in EVAL_KEYS:
        processed_params[eval_key] = eval(scenario_config[eval_key])
    for bool_key in BOOL_KEYS:
        processed_params[bool_key] = (
            "true" == scenario_config[bool_key].lower())

    LOGGER.info('waiting for downloads to complete')
    task_graph.join()
    return processed_params


def main():
    parser = argparse.ArgumentParser(description='Pollination Analysis')
    parser.add_argument(
        'configuration_path', type=str, nargs='+',
        help=(
            'Paths or patterns to pollination pipline configuration files, expects to see: ' +
            ', '.join(EXPECTED_INI_KEYS)))
    args = parser.parse_args()
    scenario_configs = _validate_config_files(args.configuration_path)

    task_graph = taskgraph.TaskGraph(
        WORKING_DIR, N_WORKERS, reporting_interval=5.0)
    for scenario_name, scenario_config in scenario_configs:
        scenario_params = _fetch_and_process_config_to_param_map(
            scenario_config, task_graph)
        LOGGER.debug(scenario_params)
        LOGGER.info(f"process landcover map: {scenario_params['LANDCOVER']}")
        poll_suff_ag_path, ag_mask_path = calculate_poll_suff(
            task_graph, scenario_params['LANDCOVER'],
            scenario_params['AG_CODES'], scenario_params['NATURAL_CODES'])

        target_poll_fed_on_ag_raster_path = os.path.join(WORKING_DIR, f'pollination_ppl_fed_on_ag_10s_{scenario_name}.tif' )

        # align pool, ag, and potential people fed
        potential_people_fed_path = scenario_params['POTENTIAL_PEOPLE_FED']
        ppf_info = geoprocessing.get_raster_info(potential_people_fed_path)
        base_info = geoprocessing.get_raster_info(scenario_params['LANDCOVER'])

        if ppf_info['bounding_box'] != base_info['bounding_box']:
            LOGGER.info(
                f'warp {potential_people_fed_path} to overlap with '
                f'{scenario_params["LANDCOVER"]}')
            aligned_potential_people_fed = os.path.join(
                CHURN_DIR, f'aligned_{os.path.basename(potential_people_fed_path)}')
            ppf_warp_task = task_graph.add_task(
                func=geoprocessing.warp_raster,
                args=(
                    potential_people_fed_path, base_info['pixel_size'],
                    aligned_potential_people_fed,
                    'near'),
                kwargs={
                    'target_bb': base_info['bounding_box'],
                    'target_projection_wkt': base_info['projection_wkt'],
                    'n_threads': multiprocessing.cpu_count()},
                target_path_list=[aligned_potential_people_fed],)
            potential_people_fed_path = aligned_potential_people_fed
            ppf_warp_task.join()

        if abs(base_info['pixel_size'][0]) > 10:
            LOGGER.info('assuming linear projection')
            pixel_area = abs(numpy.prod(base_info['pixel_size']))
        else:
            LOGGER.info('assuming lat/lng, creating varying pixel area')
            lat_array = numpy.linspace(
                base_info['bounding_box'][3],
                base_info['bounding_box'][1],
                base_info['raster_size'][1])
            pixel_area = area_of_pixel_in_ha(base_info['pixel_size'][0], lat_array)
            pixel_area = pixel_area.reshape(-1, 1)

        potential_people_fed_nodata = geoprocessing.get_raster_info(
            potential_people_fed_path)['nodata']
        poll_suff_ag_nodata = geoprocessing.get_raster_info(
            poll_suff_ag_path)['nodata']

        task_graph.add_task(
            func=geoprocessing.raster_calculator,
            args=(
                [(potential_people_fed_path, 1),
                 (poll_suff_ag_path, 1), pixel_area, (ag_mask_path, 1),
                 (potential_people_fed_nodata, 'raw'),
                 (poll_suff_ag_nodata, 'raw')],
                _pollinatinon_ppl_fed_on_ag, target_poll_fed_on_ag_raster_path,
                gdal.GDT_Float32, _MULT_NODATA),
            target_path_list=[target_poll_fed_on_ag_raster_path],
            task_name='calculate pollination people fed on ag')
    task_graph.join()
    task_graph.close()


def _pollinatinon_ppl_fed_on_ag(
        poll_dep_ppl_fed_array, poll_suff_ag_array,
        pixel_area, ag_mask_array, poll_dep_nodata, poll_suff_nodata):
    """Calculate pollination dependent people fed on ag."""
    result = numpy.full(
        poll_dep_ppl_fed_array.shape, _MULT_NODATA, dtype=numpy.float32)
    valid_mask = (
        (poll_dep_ppl_fed_array != poll_dep_nodata) &
        (poll_suff_ag_array != poll_suff_nodata) & (ag_mask_array == 1))
    if numpy.any(valid_mask):
        result[valid_mask] = (
            poll_dep_ppl_fed_array[valid_mask] * poll_suff_ag_array[valid_mask] *
            pixel_area[valid_mask])
        return result
    else:
        return None


# New Calculations:
#     'expression': 'raster1*raster2*raster3*(raster4>0)+(raster4<1)*-9999',
#            'raster1': "monfreda_2008_yield_poll_dep_ppl_fed_5min.tif", #https://storage.googleapis.com/critical-natural-capital-ecoshards/monfreda_2008_yield_poll_dep_ppl_fed_5min.tif
#            'raster2': r"workspace_poll_suff\churn\poll_suff_hab_ag_coverage_rasters\poll_suff_ag_coverage_prop_10s_forest_conversion_2050_md5_abda51.tif",
#            'raster3': "esa_pixel_area_ha_md5_1dd3298a7c4d25c891a11e01868b5db6.tif", # https://storage.googleapis.com/ecoshard-root/esa_pixel_area_ha_md5_1dd3298a7c4d25c891a11e01868b5db6.tif
#            'raster4': r"workspace_poll_suff\churn\ag_mask\forest_conversion_2050_md5_abda51_ag_mask.tif",

#     Outputs:
#         pollination_ppl_fed_on_ag_10s_forest_conversion_2050


# #python realized_pollination.py --ppl_fed_path pollination_ppl_fed_on_ag_10s_forest_conversion_2050 --hab_mask_path "workspace_poll_suff\churn\hab_mask\forest_conversion_2050_md5_abda51_hab_mask.tif"


# Expected outputs include:
#     poll_suff_hab_ag_coverage_rasters

#     [input landcover name]_ag_mask
#     [input landcover name]_hab_mask

#     pollination_ppl_fed_on_[input landcover name]

#     "realized_pollination output" (this is the output of `realized_pollination.py`)
#         * people fed
#         * norm people fed



    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
