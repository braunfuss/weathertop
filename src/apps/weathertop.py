import numpy as num
import cv2
from skimage.morphology import rectangle, closing, square
import weathertop.process.contour as contour
from matplotlib import pyplot as plt
from skimage.filters import rank, threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, approximate_polygon, subdivide_polygon
from skimage.color import label2rgb
from skimage.draw import ellipse_perimeter
import matplotlib.patches as mpatches
from affine import Affine
from scipy.ndimage import filters
from matplotlib.pyplot import cm
import subprocess
import rasterio
import rasterio.features
import rasterio.warp
import cartopy.crs as ccrs
import cartopy
import cartopy.geodesic as cgeo
from rasterio.transform import from_origin
import math
from pyrocko import orthodrome
# for data from kite:
from kite import Scene
# for grid data (licsar):
from PIL import Image
# just for testing and plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from weathertop.process.l1tf import run_l1tf
from weathertop.plotting.plotting import *
# for skelotonize
from weathertop.process.centerline import *
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely import geometry
import shapely.wkt
import json
import matplotlib.pyplot as plt
from kite.scene import BaseScene, FrameConfig
import sys
import cartopy
import glob
import os
from pyrocko.guts import GutsSafeLoader
import yaml
from pyrocko.guts import expand_stream_args
from weathertop.process.prob import rup_prop
from matplotlib import rc
from pyrocko.client import catalog
import matplotlib as mpl
from cartopy.io import srtm
from cartopy.io import PostprocessedRasterSource, LocatedImage
from cartopy.io.srtm import SRTM3Source, SRTM1Source
import logging
import os
import re
import requests

op = os.path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('kite.clients')


def shade(located_elevations):
    """
    Given an array of elevations in a LocatedImage, add a relief (shadows) to
    give a realistic 3d appearance.

    """
    new_img = srtm.add_shading(located_elevations.image,
                               azimuth=135, altitude=15)
    return LocatedImage(new_img, located_elevations.extent)


rc('axes', linewidth=2)
font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

rc('font', **font)


def _load_all(stream, Loader=GutsSafeLoader):
    return list(yaml.load_all(stream=stream, Loader=Loader))


plt.switch_backend('Qt4Agg')

# Definitions
sz1 = 20
sz2 = 20
selem = rectangle(100, 100)
d2r = num.pi / 180.


@expand_stream_args('r')
def load_all(*args, **kwargs):
    return _load_all(*args, **kwargs)


def load(path, kite_scene=True, grid=False, path_cc=None):
    '''
    Load data from a path or optional from a kite scene.
    '''
    if kite_scene is True:
        sc = Scene.load(path)
        img = sc.displacement
        unw = img.copy()
        where_are_NaNs = num.isnan(img)
        img[where_are_NaNs] = 0
        coh = img  # TODO load in coherence
        dates = [sc.meta.time_slave, sc.meta.time_master]

    if grid is True:
        unw = Image.open(path)
        img = num.array(unw, num.float32)
        where_are_NaNs_dsc = num.isnan(img)
        img[where_are_NaNs_dsc] = 0
        coh_load = Image.open(path_cc)
        coh = num.array(coh_load, num.float32)
        where_are_NaNs = num.isnan(coh)
        coh[where_are_NaNs] = 0
        sc = None
        dates = None

    return img, coh, sc, dates


def read_float(filen, width):
    '''
    Reads data (int16) and returns data arrays.
    '''
    dtype = 'float32'
    d_vec = num.fromfile(filen, dtype=dtype)
    linelen = int(len(d_vec)/width)
    d_vec = d_vec.reshape(linelen, width)

    return d_vec


def get_gradient(src):

    sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=31)
    sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=31)

    grad = num.sqrt(sobelx**2+sobely**2)
    mag = cv2.magnitude(sobelx, sobely)
    ori = cv2.phase(sobelx, sobely, True)
    return grad, mag, ori


def get_coords_from_geotiff(fname, array):
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
    cols, rows = num.meshgrid(num.arange(array.shape[1]),
                              num.arange(array.shape[0]))
    T1 = T0 * Affine.translation(0.5, 0.5)
    rc2en = lambda r, c: (c, r) * T1
    eastings, northings = num.vectorize(rc2en,
                                        otypes=[num.float, num.float])(rows, cols)
    return eastings, northings


def get_contours(phase):
    init_ls = contour.checkerboard_level_set(phase.shape, 6)
    evolution = []
    callback = contour.store_evolution_in(evolution)
    ls = contour.morphological_chan_vese(phase, 10, init_level_set=init_ls,
                                         smoothing=10,
                                         iter_callback=callback)
    img = ls
    img = num.array(ls*1.)
    return img


def get_binned_cont(phase, selem):

    phase_bin = get_contours(phase)
    img = phase_bin
    grad, mag, ori = get_gradient(img)

    px_histograms = rank.windowed_histogram(grad/num.max(grad), selem,
                                            n_bins=4)
    px_histograms = num.sum(px_histograms, axis=2)

    px_histogramsori = rank.windowed_histogram(ori/10, selem, n_bins=4)
    px_histogramsori = num.sum(px_histogramsori, axis=2)
    return px_histograms, px_histogramsori, phase_bin


def get_binned_ori(phase, selem, bin=None):
    if bin is None:
        n_bins = 12
    else:
        n_bins = 100
    grad, mag, ori = get_gradient(phase)
    px_histograms = rank.windowed_histogram(grad/num.max(grad), selem,
                                            n_bins=4)
    px_histograms = num.sum(px_histograms, axis=2)
    px_histogramsori = rank.windowed_histogram(ori/10, selem, n_bins=n_bins)
    px_histogramsori = num.sum(px_histogramsori, axis=2)

    return px_histograms, px_histogramsori


def process(img, coh, longs, lats, scene, x0, y0, x1, y1, fname, plot=True,
            mode=False, loading=False, topo=False, synthetic=False,
            calc_statistics=False, subsample=False, threshold_ss=0.1):

    '''
    Main processing chain.

    Input are:
    :img:  displacement map : numpy array
    :longs; lats: coordinates assoicated to img data , numpy array
    :coh: coherence of data or quality factor of data, same shape as img
    :scene: kite scene
    :x0,y0,x1,y1 : coordinates to apply processing within
    :fname : name of output file

    Optional:
    :topo : enable plot topography from ARCGIS server
    :mode : Choice of mode, can be true (heigh weight to coherence),
           false (standard), basic () or ss (strike-slip mode)
    :synthetic: bool
    :calc_statistics: bool
    :subsample: bool
    '''

    selem = rectangle(100, 100)
    plt_img = img.copy()
    if mode is False:
        ls = img.copy()
        ls[num.where(ls < 0)] = 1
        ls[num.where(ls != 1)] = 0
        ls_dark = img.copy()
        ls_dark[num.where(ls_dark > 0)] = 1
        ls_dark[num.where(ls_dark != 1)] = 0
        mask = filters.gaussian_filter(ls, 1, order=0)
        ls_dark[num.where(mask != 0)] = 0
        ls_dank = ls_dark.copy()

        ls_clear = ls.copy()
        ls_dark = ls_dank

        shape = num.shape(img)

        quantized_img = ls
        grad_mask, mag_mask, ori_mask = get_gradient(ls)

        grad_mask = filters.gaussian_filter(grad_mask, 20, order=0)
        grad_mask_dark, mag_mask_dark, ori_mask_dark = get_gradient(ls_dark)
        grad_mask_dark = filters.gaussian_filter(grad_mask_dark, 20, order=0)
        grad, mag, ori = get_gradient(img)
        grad2, mag2, or2 = get_gradient(grad)
        grad2 = grad2/num.max(grad2)
        grad_mask_dark_filt = filters.gaussian_filter(ls_dark, 30, order=0)
        grad_mask_filt = filters.gaussian_filter(ls, 30, order=0)

        grad_maskls, mag_mask, ori_mask = get_gradient(ls)

        pointy2 = grad*grad_mask

        grad_mask_dark = grad_mask_dark/num.max(grad_mask_dark)
        grad_mask = grad_mask/num.max(grad_mask)

        pointy2 = pointy2/num.max(pointy2)
        pointy2 = filters.gaussian_filter(pointy2, 30, order=0)

        pointy = ((grad_mask_filt*grad_mask_dark_filt))+pointy2
        pointy = pointy/num.max(pointy)
        thres = num.max(pointy)*0.1
        pointy[pointy < thres] = 0
        image = pointy.copy()

        img_abs = abs(img)
        img_filt = filters.gaussian_filter(img_abs, 30, order=0)
        img_filt = img_filt/num.max(img_filt)
        image = pointy*img_filt
        grad_mask, mag_mask, ori_mask = get_gradient(ls)
        coh[coh < num.mean(coh)] = 0
        coh_filt = filters.gaussian_filter(coh, 5, order=0)

    elif mode == 'basic':
        ls = img.copy()
        ls[num.where(ls < 0)] = 1
        ls[num.where(ls != 1)] = 0
        ls_dark = img.copy()
        ls_dark[num.where(ls_dark > 0)] = 1
        ls_dark[num.where(ls_dark != 1)] = 0
        mask = filters.gaussian_filter(ls, 1, order=0)
        ls_dark[num.where(mask != 0)] = 0
        ls_dank = ls_dark.copy()
        ls_clear = ls.copy()
        quantized_img = ls
        grad_mask, mag_mask, ori_mask = get_gradient(quantized_img)

        grad, mag, ori = get_gradient(img)
        grad2, mag2, or2 = get_gradient(grad)
        grad2 = grad2/num.max(grad2)
        grad_mask[grad_mask != 0] = 1

        pointy = grad*grad_mask
        thres = num.max(pointy)*0.1
        pointy[pointy < thres] = 0
        image = pointy.copy()

        pointy2 = grad_mask/num.max(grad_mask)
        thres = num.max(pointy2)*0.1
        pointy2[pointy2 < thres] = 0
        pointy2[pointy2 > 0] = 1
        image = pointy+pointy2

        # weight be coherence
        coh_filt = filters.gaussian_filter(num.abs(coh), 3, order=0)
        grad_mask = filters.gaussian_filter(grad_mask, 20, order=0)
        img2 = filters.gaussian_filter(img, 3, order=0)

        grad_mask, mag_mask, ori_mask = get_gradient(ls)
        grad = grad/num.max(grad)
        grad_mask = filters.gaussian_filter(grad_mask, 20, order=0)
        grad_mask = grad_mask/num.max(grad_mask)
        grad_mask, mag_mask, ori_mask = get_gradient(ls)
        image = coh_filt*grad

    elif mode is True:
        ls = img.copy()
        ls[num.where(ls < 0)] = 1
        ls[num.where(ls != 1)] = 0
        ls_dark = img.copy()
        ls_dark[num.where(ls_dark > 0)] = 1
        ls_dark[num.where(ls_dark != 1)] = 0
        mask = filters.gaussian_filter(ls, 1, order=0)
        ls_dark[num.where(mask != 0)] = 0
        ls_dank = ls_dark.copy()

        ls_dark = ls_dank
        ls_clear = ls.copy()
        shape = num.shape(img)

        ls = get_contours(img)
        quantized_img = ls
        grad_mask, mag_mask, ori_mask = get_gradient(quantized_img)
        selem = rectangle(100, 100)

        grad, mag, ori = get_gradient(img)
        grad2, mag2, or2 = get_gradient(grad)
        grad2 = grad2/num.max(grad2)

        grad_mask[grad_mask != 0] = 1

        pointy = grad*grad_mask
        thres = num.max(pointy)*0.1
        pointy[pointy < thres] = 0
        pointy[pointy > 0] = 1
        image = pointy

        pointy2 = grad_mask/num.max(grad_mask)
        thres = num.max(pointy2)*0.1
        pointy2[pointy2 < thres] = 0
        pointy2[pointy2 > 0] = 1
        image = pointy+pointy2
        coh[coh < num.mean(coh)] = 0
        coh_filt = filters.gaussian_filter(coh, 5, order=0)
        image = image*coh_filt

    elif mode == "ss":
        ls = img.copy()
        ls[num.where(ls < 0)] = 1
        ls[num.where(ls != 1)] = 0
        ls_dark = img.copy()
        ls_dark[num.where(ls_dark > 0)] = 1
        ls_dark[num.where(ls_dark != 1)] = 0
        mask = filters.gaussian_filter(ls, 1, order=0)
        ls_dark[num.where(mask != 0)] = 0
        ls_dank = ls_dark.copy()
        ls_clear = ls.copy()
        ls = get_contours(img)
        quantized_img = ls
        grad_mask, mag_mask, ori_mask = get_gradient(quantized_img)
        grad, mag, ori = get_gradient(img)
        grad2, mag2, or2 = get_gradient(grad)
        grad2 = grad2/num.max(grad2)

        grad_mask[grad_mask != 0] = 1
        pointy = grad*grad_mask

        thres = num.max(pointy)*threshold_ss
        pointy[pointy < thres] = 0
        image = pointy.copy()

        pointy2 = grad_mask/num.max(grad_mask)
        thres = num.max(pointy2)*0.1
        pointy2[pointy2 < thres] = 0
        pointy2[pointy2 > 0] = 1
        image = pointy+pointy2
        coh[coh < num.mean(coh)] = 0
        coh_filt = filters.gaussian_filter(coh, 30, order=0)

        thres = num.max(pointy)*threshold_ss
        pointy[pointy < thres] = 0
        image = pointy.copy()

        pointy2 = grad_mask/num.max(grad_mask)
        thres = num.max(pointy2)*0.1
        pointy2[pointy2 < thres] = 0
        pointy2[pointy2 > 0] = 1
        image = pointy+pointy2
        coh[coh < num.mean(coh)] = 0
        coh_filt = filters.gaussian_filter(coh, 30, order=0)
        grad = filters.gaussian_filter(grad, 30, order=0)
        image = image*coh_filt

    if plot is True:
        plot_process(longs, lats, scene, ls_dark, ls_clear, grad_mask, image,
                     grad, plt_img, coh_filt, fname, topo=topo)

    return image


def writeout(image, fname, sc=None):
    # # save as grid file

    arr = num.rot90(image.T)

    if sc is not None:
        ulNutm = sc.frame.llNutm+sc.frame.dNmeter*sc.rows
        transform = from_origin(sc.frame.llEutm, ulNutm, sc.frame.dEmeter,
                                sc.frame.dNmeter)
        new_dataset = rasterio.open(fname, 'w', driver='GTiff',
                                    height=sc.rows, width=sc.cols,
                                    count=1, dtype=str(arr.dtype),
                                    crs='+proj=utm +zone=%s +units=m +no_defs' %(sc.frame.utm_zone),
                                    transform=transform)

        new_dataset.write(arr, 1)
        new_dataset.close()


def combine(img_asc_path, img_dsc_path, name, weight_asc=1,
            weight_dsc=1, plot=False):
    print('Merging ascending and descending outputs with gdal')

    subprocess.run(["gdal_merge.py", "-o", "work-%s/comb.tif" % name, img_asc_path, img_dsc_path, "-seperate"])
    subprocess.run(['gdal_calc.py', '--calc=%s*A+%s*B' % (weight_asc, weight_dsc), "--outfile=work-%s/merged.tiff" % (name), '-A', "work-%s/comb.tif" % name,
    "--A_band=1", "-B", "work-%s/comb.tif" % name, "--B_band=2", '--overwrite'])
    fname = 'work-%s/merged.tiff' % name
    comb = rasterio.open('work-%s/merged.tiff' % name)
    img = comb.read(1)
    img = img / num.sqrt(num.sum(img**2))
    return img


def to_latlon(fname):
    import rasterio
    import numpy as np
    from affine import Affine
    from pyproj import Proj, transform

    # Read raster
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = num.meshgrid(num.arange(A.shape[2]), num.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = num.vectorize(rc2en, otypes=[num.float, num.float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong', datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)
    return longs, lats


def bounding_box(image, area, sharp=False, simple=False):
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(1))
    if area is None:
        area = 900

    label_image = label(bw)
    image_label_overlay = label2rgb(label_image, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    newlist = sorted(regionprops(label_image), key=lambda region: region.area,
                     reverse=True)
    polys = []
    centers = []
    coords_out = []
    coords_box = []
    ellipses = []
    strikes = []
    minrs = []
    mincs = []
    maxrs = []
    maxcs = []
    max_bound = []
    for region in regionprops(label_image):
        if region.area >= area:  # check if selected area is large enough

            coords = []
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            y0, x0 = region.centroid
            orientation = region.orientation
            strikes.append(num.rad2deg(orientation)+90.)
            ellipses.append([x0, y0, region.major_axis_length,
                            region.minor_axis_length, orientation])
            coords_box.append([minr, minc, maxr, maxc])
            minrs.append(minr)
            mincs.append(minc)
            maxrs.append(maxr)
            maxcs.append(maxc)
            x1 = x0 + math.cos(orientation) * 0.5 * region.major_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * region.major_axis_length
            x1a = x0 - math.cos(orientation) * 0.5 * region.major_axis_length
            y1a = y0 + math.sin(orientation) * 0.5 * region.major_axis_length
            x2 = x0 - math.sin(orientation) * 0.05 * region.minor_axis_length
            y2 = y0 - math.cos(orientation) * 0.05 * region.minor_axis_length
            coords.append([x1, y1])
            coords.append([x1a, y1a])
            coords.append([x0, y0])
            coords.append([x2, y2])
            coords = num.array(coords)
            poly = geometry.Polygon([[p[0], p[1]] for p in coords])
            polys.append(poly)
            hull = poly.convex_hull
            try:
                koor = hull.exterior.coords
                pol = geometry.Polygon([[p[1], p[0]] for p in koor])
                center = Centerline(pol)
                centers.append(center)
            except:
                pass

            ax.plot((x0, x1), (y0, y1), '-r', linewidth=12.5)
            ax.plot((x0, x1a), (y0, y1a), '-r', linewidth=12.5)

            ax.plot((x0, x2), (y0, y2), '-r', linewidth=12.5)
            ax.plot(x0, y0, '.g', markersize=15)
            coords_out.append(coords)
    ax.set_axis_off()
    plt.close()
    try:
        max_bound = [num.min(minrs), num.min(mincs),
                     num.max(maxrs), num.max(maxcs)]

        thresh = threshold_otsu(image)
        bw = closing(image > num.max(image)*0.1, square(80))

        if area is None:
            area = 900

        label_image = label(bw)
        image_label_overlay = label2rgb(label_image, image=image)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_label_overlay)
        newlist = sorted(regionprops(label_image), key=lambda region: region.area,
                         reverse=True)

        region = newlist[0]
        if region.area >= area:  # check if nec.

            coords = []
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            y0, x0 = region.centroid
            orientation = region.orientation
            strikes.append(num.rad2deg(orientation)+90.)
            ellipses.append([x0, y0, region.major_axis_length,
                            region.minor_axis_length, orientation])
            coords_box.append([minr, minc, maxr, maxc])
            minrs.append(minr)
            mincs.append(minc)
            maxrs.append(maxr)
            maxcs.append(maxc)
            x1 = x0 + math.cos(orientation) * 0.5 * region.major_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * region.major_axis_length
            x1a = x0 - math.cos(orientation) * 0.5 * region.major_axis_length
            y1a = y0 + math.sin(orientation) * 0.5 * region.major_axis_length
            x2 = x0 - math.sin(orientation) * 0.05 * region.minor_axis_length
            y2 = y0 - math.cos(orientation) * 0.05 * region.minor_axis_length
            coords.append([x1, y1])
            coords.append([x1a, y1a])
            coords.append([x0, y0])
            coords.append([x2, y2])
            coords = num.array(coords)
            poly = geometry.Polygon([[p[0], p[1]] for p in coords])
            polys.append(poly)
            hull = poly.convex_hull
            try:
                koor = hull.exterior.coords
                pol = geometry.Polygon([[p[1], p[0]] for p in koor])
                center = Centerline(pol)
                centers.append(center)
            except:
                pass

            ax.plot((x0, x1), (y0, y1), '-r', linewidth=12.5)
            ax.plot((x0, x1a), (y0, y1a), '-r', linewidth=12.5)

            ax.plot((x0, x2), (y0, y2), '-r', linewidth=12.5)
            ax.plot(x0, y0, '.g', markersize=15)
            coords_out.append(coords)
        ax.set_axis_off()
        plt.close()
    except:
        pass

    return centers, coords_out, coords_box, strikes, ellipses, max_bound


def skelotonize(image, plot=True):
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(2))
    cleared = clear_border(bw)
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image)
    if plot is True:
        fig, ax = plt.subplots(figsize=(sz1, sz2))
        ax.imshow(label_image)
    polys = []
    centers = []
    for region in regionprops(label_image):
        if region.area >= 400:  # check if nec.
            coords = num.array(region.coords)
            poly = geometry.Polygon([[p[0], p[1]] for p in coords])
            polys.append(poly)
            hull = poly.convex_hull

            koor = hull.exterior.coords
            pol = geometry.Polygon([[p[0], p[1]] for p in koor])
            pol = pol.buffer(-1)
            pol = pol.simplify(4000, preserve_topology=True)  # check
            try:
                center = Centerline(pol)
                centers.append(center)
            except:
                pass
        if region.area >= 400:  # check; should be scaled to pixel size?
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.title('Fault outline skeleton', fontsize=22)
    plt.close()
    return centers


def l1tf_prep(res_faults, plot=True):
    num_nodes = sum(len(f['geometry']['coordinates']) for f in res_faults)
    x = [None] * num_nodes
    y = [None] * num_nodes
    x = []
    y = []
    for f in res_faults:
        for k in f['geometry']['coordinates']:
            x.append(k[0])
            y.append(k[1])
    if plot is True:
        plt.scatter(y, x)
        plt.close()

        plt.plot(x, y)
        plt.close()
        plt.plot(x)
        plt.close()
        plt.plot(y)
        plt.close()
    return num.asarray(x)


def simplify(centers, plot=True):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(sz1, sz2))
    simp_fault = []
    comp_fault = []
    for center in centers:
        coords =[]
        for line in center:
            coord = line.coords
            coords.append(coord)
        coords = num.array(coords)
        geom_object = num.array([(coords[i][m]) for i in range(num.shape(coords)[0]) for m in range(num.shape(coords[i])[0])])
        new_geom_object = geom_object.copy()
        for _ in range(1):
            new_geom_object = subdivide_polygon(new_geom_object, degree=2,
                                                preserve_ends=True)
            new_geom_object = approximate_polygon(new_geom_object,
                                                  tolerance=0.8)
        comp_fault.append(new_geom_object)

        appr_geom_object = geom_object.copy()
        # approximate subdivided polygon with Douglas-Peucker algorithm
        appr_geom_object = approximate_polygon(appr_geom_object, tolerance=10)

        ax1.scatter(appr_geom_object[:, 1], appr_geom_object[:, 0])
        simp_fault.append(appr_geom_object[:, :])

        ax2.scatter(new_geom_object[:, 1], new_geom_object[:, 0])
    ax1.set_title('Simple line')
    ax2.set_title('More complexity')
    plt.close()
    return simp_fault, comp_fault


def df_to_geojson(df, eastings, northings, properties=None):
    coords_fault = []
    n = 0
    geojson = {'type': 'FeatureCollection', 'features': []}
    for p in df:
        feature = {'type': 'Feature',
                   'properties': {'ogc_fid': n,
                                  'ns_average_dip': 60,
                                  'ns_average_rake': 120},
                   'geometry': {'type': 'LineString',
                                'coordinates': []}}
        for k in p:
            kx = num.asarray(k[0])
            ky = num.asarray(k[1])
            feature['geometry']['coordinates'] = [(eastings[int(kx)][int(ky)],
                                                   northings[int(kx)][int(ky)])]
            coords_fault.append((eastings[int(kx)][int(ky)],
                                 northings[int(kx)][int(ky)]))
        n =+ 1

        feature['geometry']['coordinates'] = coords_fault
        geojson['features'].append(feature)
    return geojson


def dump_geojson(fault, eastings, northings, name, tiff=False):
    if tiff is True:
        eastings, northings = get_coords_from_geotiff(fname, img)
    east = num.min(eastings)
    north = num.min(northings)
    database = df_to_geojson(fault, eastings, northings)
    with open('work-%s/fault_lines' % name, 'w') as f:
        json.dump(database, f)

    return database


def aoi_snr(image, area):
    where_are_NaNs = num.isnan(image)
    image[where_are_NaNs] = 0
    grad, mag, ori = get_gradient(image)
    thresh = threshold_otsu(grad)
    single_area = False
    area = 600
    i = 0
    bw = closing(grad > thresh, square(1+i))
    bw2 = closing(grad > num.max(grad)*0.05, square(1+i))
    i = i +1
    label_image = label(bw)
    image_label_overlay = label2rgb(label_image, image=grad)

    newlist = sorted(regionprops(label_image), key=lambda region: region.area,
                     reverse=True)
    count = 0
    region_max = 0
    for region in regionprops(label_image):
        if region.area > region_max:
            if region.area > area:
                count = count +1
                minr, minc, maxr, maxc = region.bbox

    if count == 1:
        single_area = True

    signal = grad[minr:maxr, minc:maxc]
    noise = grad.copy()
    noise[minr:maxr, minc:maxc] = 0
    snr = num.log(num.max(signal)/num.max(noise))
    return abs(snr)


def main():
    if len(sys.argv) < 2:
        print("input: asc_path dsc_path minlat minlon maxlat maxlon --workdir=name m")
    try:
        x0 = float(sys.argv[3])
        y0 = float(sys.argv[4])
        x1 = float(sys.argv[5])
        y1 = float(sys.argv[6])
    except:
        x0 = "eins"
        y0 = "eins"
        x1 = "eins"
        y1 = "eins"

    sharp = False
    loading = False
    plot = True
    topo = False
    synthetic = False
    calc_statistics = False
    subsample = False
    dump_grid = False

    for argv in sys.argv:
        if argv == "--sharp":
            sharp = True
        if argv == "--basic":
            sharp = "basic"
        if argv == "--ss":
            sharp = "ss"
        if argv == "--loading=True":
            loading = True
        if argv == "--loading=true":
            loading = True
        if argv == "--plot=False":
            plot = False
        if argv[0:10] == "--workdir=":
            name = argv[10:]
        if argv == "--topography":
            topo = True
        if argv == "--synthetic":
            synthetic = True
        if argv == "--statistics":
            calc_statistics = True
        if argv == "--subsample":
            subsample = True
        if argv == "--grond_export":
            dump_grid = True

    strikes = []
    lengths = []
    widths = []

    if loading is False:

        img_asc, coh_asc, scene_asc, dates_asc = load(sys.argv[1],
                                                      kite_scene=True)

        try:
            os.mkdir('work-%s' % name)
        except:
            pass
        files = glob.glob('work-%s/*' % name)
        for f in files:
            os.remove(f)
        fname = 'work-%s/asc.mod.tif' % name
        writeout(img_asc, fname, sc=scene_asc)
        longs_asc, lats_asc = to_latlon(fname)

        try:
            global_cmt_catalog = catalog.GlobalCMT()

            events = global_cmt_catalog.get_events(
                time_range=(num.min(dates_asc), num.max(dates_asc)),
                magmin=2.,
                latmin=num.min(lats_asc),
                latmax=num.max(lats_asc),
                lonmin=num.min(longs_asc),
                lonmax=num.max(longs_asc))

            areas = []

            for ev in events:
                areas.append(num.cbrt(ev.moment_tensor.moment)/1000)
            area = num.max(areas)
        except:
            area = 400

        fname = 'work-%s/asc-' % name

        img_asc = process(img_asc, coh_asc, longs_asc, lats_asc, scene_asc,
                          x0, y0, x1, y1, fname, plot=plot, mode=sharp,
                          loading=loading, topo=topo, synthetic=synthetic,
                          calc_statistics=calc_statistics, subsample=subsample)
        fname = 'work-%s/asc.mod.tif' % name
        writeout(img_asc, fname, sc=scene_asc)
        db =1
        dates = []
        img_asc, coh_asc, scene_asc, dates_asc = load(sys.argv[1],
                                                      kite_scene=True)
        dates.append(dates_asc)
        snr_asc = aoi_snr(img_asc, area)

        img_dsc, coh_dsc, scene_dsc, dates_dsc = load(sys.argv[2],
                                                      kite_scene=True)
        dates.append(dates_dsc)

        fname = 'work-%s/dsc.mod.tif' % name
        writeout(img_dsc, fname, sc=scene_dsc)
        longs_dsc, lats_dsc = to_latlon(fname)
        fname = 'work-%s/dsc-' % name
        img_dsc = process(img_dsc, coh_dsc, longs_dsc, lats_dsc, scene_dsc,
                          x0, y0, x1, y1, fname, plot=plot, mode=sharp,
                          loading=loading, topo=topo, synthetic=synthetic,
                          calc_statistics=calc_statistics, subsample=subsample)
        fname = 'work-%s/dsc.mod.tif' % name
        writeout(img_dsc, fname, sc=scene_dsc)

        db = 1
        img_dsc, coh_dsc, scene_dsc, dates = load(sys.argv[2], kite_scene=True)
        snr_dsc = aoi_snr(img_dsc, area)

        minda = num.min(scene_asc.displacement)
        mindd = num.min(scene_dsc.displacement)
        mind = num.min([minda, mindd])
        maxa = num.max(scene_asc.displacement)
        maxdd = num.max(scene_dsc.displacement)
        maxd = num.max([maxa, maxdd])
        max_cum = num.max([abs(maxd), abs(mind)])
        minda = -max_cum
        mindd = -max_cum
        mind = -max_cum
        maxa = max_cum
        maxdd = max_cum
        maxd = max_cum

        if plot is True:
            fname = 'work-%s/asc' % name
            plot_on_map(db, scene_asc, longs_asc, lats_asc, x0, y0, x1, y1,
                        minda, maxa, fname,
                        synthetic=synthetic, topo=topo, kite_scene=True)
            fname = 'work-%s/dsc' % name
            plot_on_map(db, scene_dsc, longs_dsc, lats_dsc, x0, y0, x1, y1,
                        mindd, maxdd, fname,
                        synthetic=synthetic, topo=topo, kite_scene=True)

        fname = 'work-%s/asc.mod.tif' % name
        comb = rasterio.open(fname)
        longs_comb, lats_comb = to_latlon(fname)
        comb_img = comb.read(1)

        centers_bounding, coords_out, coords_box, strike, ellipses, max_bound=bounding_box(comb_img, 400, sharp)
        for st in strike:
            strikes.append(st)
        print("Strike(s) of moment weighted centerline(s) are :%s" % strike)

        if plot is True:
            fname = 'work-%s/asc-comb-' % name

            plot_on_kite_box(coords_box, coords_out, scene_asc, longs_asc,
                             lats_asc, longs_comb, lats_comb, x0, y0, x1, y1,
                             name, ellipses, minda, maxa, fname,
                             synthetic=synthetic, topo=topo)

        fname = 'work-%s/dsc.mod.tif' % name
        comb = rasterio.open(fname)
        longs_comb, lats_comb = to_latlon(fname)
        comb_img = comb.read(1)

        centers_bounding, coords_out, coords_box, strike, ellipses, max_bound = bounding_box(comb_img,
                                                                        400, sharp)

        for st in strike:
            strikes.append(st)
        print("Strike(s) of moment weighted centerline(s) are :%s" % strike)

        if plot is True:
            fname = 'work-%s/dsc-comb-' % name

            plot_on_kite_box(coords_box, coords_out, scene_dsc, longs_dsc,
                             lats_dsc, longs_comb, lats_comb, x0, y0, x1, y1,
                             name, ellipses, mindd, maxdd, fname,
                             synthetic=synthetic, topo=topo)


        comb_img = combine('work-%s/asc.mod.tif' % name, 'work-%s/dsc.mod.tif' % name, name, weight_asc=snr_asc, weight_dsc=snr_dsc, plot=False)
        longs_comb, lats_comb = to_latlon("work-%s/merged.tiff" % name)

    else:
        fname = 'work-%s/merged.tiff' % name
        comb = rasterio.open(fname)
        longs, lats = to_latlon(fname)
        comb_img = comb.read(1)
        easts, norths = get_coords_from_geotiff(fname, comb_img)
        dE = easts[1]-easts[0]
        dN = norths[1]-norths[0]
        ll_long = num.min(longs)
        ll_lat = num.min(lats)
        dates = []
        img_asc, coh_asc, scene_asc, dates_asc = load(sys.argv[1],
                                                      kite_scene=True)
        img_dsc, coh_dsc, scene_dsc, dates_dsc = load(sys.argv[2],
                                                      kite_scene=True)

        minda = num.min(scene_asc.displacement)
        mindd = num.min(scene_dsc.displacement)
        mind = num.min([minda, mindd])
        maxa = num.max(scene_asc.displacement)
        maxdd = num.max(scene_dsc.displacement)
        maxd = num.max([maxa, maxdd])

        if plot is True:
            plt.figure(figsize=(sz1, sz2))
            plt.title('Loaded combined image')
            xr = plt.imshow(comb_img)
            plt.close()

        if subsample is True:
            # Define the scene's frame
            frame = FrameConfig(
                # Lower left geographical reference [deg]
                llLat=ll_lat, llLon=ll_long,
                # Pixel spacing [m] or [degrees]
                spacing='meter', dE=dE, dN=dN)

            displacement = comb_img
            # Look vectors
            # Theta is elevation angle from horizon
            theta = num.full_like(displacement, 48.*d2r)
            # Phi is azimuth towards the satellite, counter-clockwise from East
            phi = num.full_like(displacement, 23.*d2r)

            kite_comb_scene = Scene(
                displacement=displacement,
                phi=phi, theta=theta,
                frame=frame)
            kite_comb_scene.spool()

            # For convenience we set an abbreviation to the quadtree
            qt = kite_comb_scene.quadtree

            # Parametrisation of the quadtree
            qt.epsilon = 0.024        # Variance threshold
            qt.nan_allowed = 0.9      # Percentage of NaN values allowed per tile/leave

            qt.tile_size_max = 12000  # Maximum leave edge length in [m] or [deg]
            qt.tile_size_min = 250    # Minimum leave edge length in [m] or [deg]

            # We save the scene in kite's format
            # sc.save('kite_scene')

            # Or export the quadtree to CSV file
            # qt.export('/tmp/tree.csv')

    # statistical output
    # img_asc, coh_asc, scene_asc = load('muji_kite/asc', kite_scene=True)
    # comb_img = process(img_asc, coh_asc, plot=True)
    # use quadtree subsampling on gradient

    img_asc, coh_asc, scene_asc, dates = load(sys.argv[1], kite_scene=True)
    fname = 'work-%s/asc.mod.tif' %name
    longs_asc, lats_asc = to_latlon(fname)
    db = 1
    longs_comb, lats_comb = to_latlon("work-%s/merged.tiff" % name)
    mindc = num.min(comb_img)
    maxdc = num.max(comb_img)

    try:
        global_cmt_catalog = catalog.GlobalCMT()

        events = global_cmt_catalog.get_events(
            time_range=(num.min(dates), num.max(dates)),
            magmin=2.,
            latmin=num.min(lats_comb),
            latmax=num.max(lats_comb),
            lonmin=num.min(longs_comb),
            lonmax=num.max(longs_comb))

        areas = []

        for ev in events:
            areas.append(num.cbrt(ev.moment_tensor.moment)/1000)
        area = num.max(areas)
    except:
        area = 400

    if dump_grid is True:
        from scipy import signal
        es = longs_comb.flatten()
        es_resamp = signal.decimate(es, 20)

        ns = lats_comb.flatten()
        ns_resamp = signal.decimate(ns, 20)

        comb_img_grid = comb_img.flatten()
        comb_img_grid_resamp = signal.decimate(comb_img_grid, 20)
        fobj_cum = open(os.path.join('work-%s/grad_grid.ASC' % name),
                        'w')
        for x, y, sembcums in zip(es, ns, comb_img_grid.flatten()):
            fobj_cum.write('%.2f %.2f %.20f\n' % (x, y, sembcums))
        fobj_cum.close()

        fobj_cum = open(os.path.join('work-%s/grad_grid_resam.ASC' % name),
                        'w')
        for x, y, sembcums in zip(es_resamp, ns_resamp, comb_img_grid_resamp):
            fobj_cum.write('%.2f %.2f %.20f\n' % (x, y, sembcums))
        fobj_cum.close()

    if plot is True:
        fname = 'work-%s/comb-' % name

        plot_on_map(db, comb_img.copy(), longs_comb, lats_comb, x0, y0, x1, y1,
                    mindc, maxdc, fname,
                    synthetic=synthetic, topo=topo, comb=True)

    centers_bounding, coords_out, coords_box, strike, ellipses, max_bound = bounding_box(comb_img,
                                                                    area, sharp)
    for st in strike:
        strikes.append(st)
    print("Strike(s) of moment weighted centerline(s) are :%s" % strike)

    if plot is True:
        fname = 'work-%s/comb-' % name

        lengths, widths = plot_on_kite_box(coords_box, coords_out, scene_asc,
                                           longs_asc, lats_asc, longs_comb,
                                           lats_comb, x0, y0, x1, y1,
                                           name, ellipses, mind, maxd, fname,
                                           synthetic=synthetic, topo=topo)

        fobj_cum = open(os.path.join('work-%s/priors.ASC' % name), 'w')
        for lens, wid in zip(lengths, widths):
            fobj_cum.write('%.2f %.2f\n' % (lens, wid))
        fobj_cum.close()

        fobj_cum = open(os.path.join('work-%s/priors_strike.ASC' % name), 'w')
        for st in zip(strikes):
            fobj_cum.write('%.2f\n' % (st))
        fobj_cum.close()

        plot_on_kite_line(coords_out, scene_asc, longs_asc, lats_asc,
                          longs_comb, lats_comb, x0, y0, x1, y1, mind, maxd,
                          fname, synthetic=synthetic, topo=topo)

    simp_fault, comp_fault = simplify(centers_bounding)

    db = dump_geojson(simp_fault, longs_comb, lats_comb, name)
    if plot is True:
        plot_on_kite_scatter(db, scene_asc, longs_asc, lats_asc, x0, y0, x1,
                             y1, mind, maxd, fname,
                             synthetic=synthetic, topo=topo,)

    img_dsc, coh_dsc, scene_dsc, dates = load(sys.argv[2], kite_scene=True)
    fname = 'work-%s/dsc.mod.tif' % name
    longs_dsc, lats_dsc = to_latlon(fname)

    fname = 'work-%s/comb-' % name
    if plot is True:
        plot_on_kite_scatter(db, scene_dsc, longs_dsc, lats_dsc, x0, y0, x1,
                             y1, mind, maxd, fname,
                             synthetic=synthetic, topo=topo,)

    centers = skelotonize(comb_img)
    simp_fault, comp_fault = simplify(centers)

    if calc_statistics is True:
        res_faults = rup_prop(db)
        y = l1tf_prep(res_faults)
        run_l1tf(y)
