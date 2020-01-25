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
# for skelotonize
from weathertop.process.centerline import *
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely import geometry
import shapely.wkt
import json
import matplotlib.pyplot as plt
from kite.scene import BaseScene, FrameConfig
import sys
from mpl_toolkits.basemap import Basemap
import glob
import os
from pyrocko.guts import GutsSafeLoader
import yaml
from pyrocko.guts import expand_stream_args
from weathertop.process.prob import rup_prop
from matplotlib import rc
from pyrocko.client import catalog
import matplotlib as mpl


class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return num.ma.masked_array(num.interp(value, x, y), num.isnan(value))


rc('axes', linewidth=2)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

rc('font', **font)

def _load_all(stream, Loader=GutsSafeLoader):
    return list(yaml.load_all(stream=stream, Loader=Loader))

plt.switch_backend('Qt4Agg')

#Definitions
sz1=20
sz2=20
selem = rectangle(100,100)
d2r = num.pi / 180.


@expand_stream_args('r')
def load_all(*args, **kwargs):
    return _load_all(*args, **kwargs)


def addArrow(ax, scene):
    from matplotlib import patches
    phi = num.nanmean(scene.phi)
    los_dx = num.cos(phi + num.pi) * .0625
    los_dy = num.sin(phi + num.pi) * .0625

    az_dx = num.cos(phi - num.pi/2) * .125
    az_dy = num.sin(phi - num.pi/2) * .125

    anchor_x = .9 if los_dx < 0 else .1
    anchor_y = .85 if los_dx < 0 else .975

    az_arrow = patches.FancyArrow(
        x=anchor_x-az_dx, y=anchor_y-az_dy,
        dx=az_dx, dy=az_dy,
        head_width=.025,
        alpha=.8, fc='k',
        head_starts_at_zero=False,
        length_includes_head=True,
        transform=ax.transAxes)

    los_arrow = patches.FancyArrow(
        x=anchor_x-az_dx/2, y=anchor_y-az_dy/2,
        dx=los_dx, dy=los_dy,
        head_width=.02,
        alpha=.8, fc='k',
        head_starts_at_zero=False,
        length_includes_head=True,
        transform=ax.transAxes)

    ax.add_artist(az_arrow)
    ax.add_artist(los_arrow)


def plot_on_kite_scatter(db, scene, eastings, northings, x0, y0, x1, y1, mind, maxd,
                         fname,
                         synthetic=False, topo=False):
            scd = scene
            data_dsc= scd.displacement

            data_dsc[data_dsc==0] = num.nan

            map = Basemap(projection='merc', llcrnrlon=num.min(eastings),llcrnrlat=num.min(northings),urcrnrlon=num.max(eastings),urcrnrlat=num.max(northings),
                          resolution='h', epsg=3395)
            ratio_lat = num.max(northings)/num.min(northings)
            ratio_lon = num.max(eastings)/num.min(eastings)
            try:
                map.drawmapscale(num.min(eastings)+2.1+ratio_lon*0.25, num.min(northings)+ratio_lat*0.18, num.mean(eastings), num.mean(northings), 30, fontsize=18, barstyle='fancy')
            except:
                map.drawmapscale(num.min(eastings)+0.05, num.min(northings)+0.04, num.mean(eastings), num.mean(northings), 10, fontsize=18, barstyle='fancy')

            try:
                parallels = num.linspace(y0,y1,22)
                meridians = num.linspace(x0, x1,22)
            except:
                parallels = num.linspace((num.min(northings)),(num.max(northings)),22)
                meridians = num.linspace((num.min(eastings)),(num.max(eastings)),22)
            xpixels = 800
            if topo is True:
                map.arcgisimage(service='World_Shaded_Relief', xpixels = xpixels, verbose= False)
            map.imshow(data_dsc, cmap="seismic", alpha=0.8, norm=MidpointNormalize(mind, maxd, 0.))
            gj = db
            faults = gj['features']
            coords = [feat['geometry']['coordinates'] for feat in faults]

            i = len(coords)
            colors = iter(cm.rainbow(num.linspace(0, 1, i)))
            for j in coords:
                coords_re_x = []
                coords_re_y = []
                for k in j:
                    coords_re_x.append(k[0])
                    coords_re_y.append(k[1])
                x, y = map(coords_re_x, coords_re_y)
                plt.scatter(x, y, c=next(colors))
            ax = plt.gca()

            meridians = num.around(meridians, decimals=1, out=None)
            parallels = num.around(parallels, decimals=1, out=None)

            ticks = map(meridians, parallels)

            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
            ax.set_xticklabels(meridians, rotation=45, fontsize=22)
            ax.set_yticklabels(parallels, fontsize=22)
            ax.tick_params(direction='out', length=6, width=4)
            plt.grid()

            #map.drawparallels(parallels,labels=[1,0,0,0],fontsize=22)
            #map.drawmeridians(meridians,labels=[1,1,0,1],fontsize=22, rotation=45)
            addArrow(ax, scene)
            try:

                x0, y0 = map(x0, y0)
                x1, y1 = map(x1, y1)
                ax.set_xlim([x0, x1])
                ax.set_ylim([y0, y1])
            except:
                pass
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            try:
                plt.colorbar(cax=cax)
            except TypeError:
                pass
            fig = plt.gcf()
            fig.set_size_inches((11, 11), forward=False)
            plt.savefig(fname+'scatter.svg', format='svg', dpi=300)
            plt.close()


def plot_on_kite_line(coords_out, scene, eastings, northings, eastcomb,
                      northcomb, x0c, y0c, x1c, y1c, mind, maxd, fname,
                      synthetic=False, topo=False):
            scd = scene
            data_dsc= scd.displacement

            data_dsc[data_dsc==0] = num.nan

            map = Basemap(projection='merc', llcrnrlon=num.min(eastings),llcrnrlat=num.min(northings),urcrnrlon=num.max(eastings),urcrnrlat=num.max(northings),
                          resolution='h', epsg=3395)
            ratio_lat = num.max(northings)/num.min(northings)
            ratio_lon = num.max(eastings)/num.min(eastings)
            try:
                map.drawmapscale(num.min(eastings)+2.1+ratio_lon*0.25, num.min(northings)+ratio_lat*0.18, num.mean(eastings), num.mean(northings), 30, fontsize=18, barstyle='fancy')
            except:
                map.drawmapscale(num.min(eastings)+0.05, num.min(northings)+0.04, num.mean(eastings), num.mean(northings), 10, fontsize=18, barstyle='fancy')

            try:
                parallels = num.linspace(y0c, y1c, 22)
                meridians = num.linspace(x0c, x1c, 22)
            except:
                parallels = num.linspace((num.min(northings)),(num.max(northings)),22)
                meridians = num.linspace((num.min(eastings)),(num.max(eastings)),22)
            xpixels = 800
            if topo is True:
                map.arcgisimage(service='World_Shaded_Relief', xpixels = xpixels, verbose= False)
            map.imshow(data_dsc, cmap="seismic", vmin=mind, vmax=maxd, alpha=0.8)
            ax = plt.gca()
            coords_all = []
            for coords in coords_out:

                coords_boxes = []
                for k in coords:
                    kx = k[1]
                    ky = k[0]
                    coords_boxes.append([eastcomb[int(kx)][int(ky)], northcomb[int(kx)][int(ky)]])
                coords_all.append(coords_boxes)
            n = 0

            for coords in coords_all:

                x1, y1 = map(coords[0][0], coords[0][1])
                x1a, y1a = map(coords[1][0], coords[1][1])
                x0, y0 = map(coords[2][0], coords[2][1])
                x2, y2 = map(coords[3][0], coords[3][1])
                n = n+1
                ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                ax.plot((x0, x1a), (y0, y1a), '-r', linewidth=2.5)

                ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                ax.plot(x0, y0, '.g', markersize=15)


            ax = plt.gca()

            meridians = num.around(meridians, decimals=1, out=None)
            parallels = num.around(parallels, decimals=1, out=None)

            ticks = map(meridians, parallels)


            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
            ax.set_xticklabels(meridians, rotation=45, fontsize=22)
            ax.set_yticklabels(parallels, fontsize=22)
            ax.tick_params(direction='out', length=6, width=4)
            plt.grid()
            addArrow(ax, scene)
            try:

                x0c, y0c = map(x0c, y0c)
                x1c, y1c = map(x1c, y1c)
                ax.set_xlim([x0c, x1c])
                ax.set_ylim([y0c, y1c])
            except:
                pass
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(cax=cax)
            fig = plt.gcf()
            fig.set_size_inches((11, 11), forward=False)
            plt.savefig(fname+'line.svg', format='svg', dpi=300)
            plt.close()


def plot_on_kite_box(coords_out, coords_line, scene, eastings, northings,
                     eastcomb, northcomb, x0c, y0c, x1c, y1c, name, ellipses,
                     mind, maxd, fname, synthetic=False, topo=False):
            scd = scene
            data_dsc = scd.displacement
            lengths = []
            widths = []

            data_dsc[data_dsc==0] = num.nan

            map = Basemap(projection='merc', llcrnrlon=num.min(eastings),
                          llcrnrlat=num.min(northings),
                          urcrnrlon=num.max(eastings),
                          urcrnrlat=num.max(northings),
                          resolution='h', epsg=3395)
            ratio_lat = num.max(northings)/num.min(northings)
            ratio_lon = num.max(eastings)/num.min(eastings)
            try:
                map.drawmapscale(num.min(eastings)+2.1+ratio_lon*0.25, num.min(northings)+ratio_lat*0.18, num.mean(eastings), num.mean(northings), 30, fontsize=18, barstyle='fancy')
            except:
                map.drawmapscale(num.min(eastings)+0.05, num.min(northings)+0.04, num.mean(eastings), num.mean(northings), 10, fontsize=18, barstyle='fancy')

            try:
                parallels = num.linspace(y0,y1,22)
                meridians = num.linspace(x0, x1,22)
            except:
                parallels = num.linspace((num.min(northings)),(num.max(northings)),22)
                meridians = num.linspace((num.min(eastings)),(num.max(eastings)),22)
            if topo is True:
                map.arcgisimage(service='World_Shaded_Relief', xpixels=800, verbose=False)

            map.imshow(data_dsc, cmap="seismic", vmin=mind, vmax=maxd,
                       alpha=0.8, norm=MidpointNormalize(mind, maxd, 0.))
            ax = plt.gca()

            coords_all = []
            for coords in coords_line:
                coords_boxes = []
                for k in coords:
                    kx = k[1]
                    ky = k[0]
                    coords_boxes.append([eastcomb[int(kx)][int(ky)], northcomb[int(kx)][int(ky)]])
                coords_all.append(coords_boxes)
            n = 0
            for coords, ell in zip(coords_all, ellipses):

                x1, y1 = map(coords[0][0], coords[0][1])
                x1a, y1a = map(coords[1][0], coords[1][1])
                x0, y0 = map(coords[2][0], coords[2][1])
                x2, y2 = map(coords[3][0], coords[3][1])
                n = n+1
                ax.plot((x0, x1), (y0, y1), 'r--', linewidth=2.5)
                ax.plot((x0, x1a), (y0, y1a), 'r--', linewidth=2.5)

                ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                ax.plot(x0, y0, '.g', markersize=15)
                height = orthodrome.distance_accurate50m(coords[0][0],
                                                         coords[0][1],
                                                         coords[3][0],
                                                         coords[3][1])
                width = orthodrome.distance_accurate50m(coords[2][0],
                                                        coords[2][1],
                                                        coords[3][0],
                                                        coords[3][1])
                lengths.append(height)
                widths.append(width)
                e = mpatches.Ellipse((x0, y0), width=width*2.,
                                     height=height*2.,
                                     angle=num.rad2deg(ell[4])+90, lw=2,
                                     edgecolor='purple', fill=False)
                ax.add_patch(e)

            coords_boxes = []
            for k in coords_out:
                minr, minc, maxr, maxc = k[0], k[1], k[2], k[3]

                kx = k[2]
                ky = k[1]
                coords_boxes.append([eastcomb[int(kx)][int(ky)],
                                     northcomb[int(kx)][int(ky)]])
                kx = k[0]
                ky = k[3]
                coords_boxes.append([eastcomb[int(kx)][int(ky)],
                                     northcomb[int(kx)][int(ky)]])

            n = 0

            for coords in coords_out:
                minc, minr = map(coords_boxes[0+n][0], coords_boxes[0+n][1])
                maxc, maxr = map(coords_boxes[1+n][0], coords_boxes[1+n][1])

                n = n+2
                rect = mpatches.Rectangle((minc, minr),  maxc - minc,
                                          maxr - minr,
                                          fill=False, edgecolor='r',
                                          linewidth=2)
                ax.add_patch(rect)

            try:
                parallels = num.linspace(y0c, y1c, 22)
                meridians = num.linspace(x0c, x1c, 22)
            except:
                parallels = num.linspace((num.min(northings)),
                                         (num.max(northings)), 22)
                meridians = num.linspace((num.min(eastings)),
                                         (num.max(eastings)), 22)

            meridians = num.around(meridians, decimals=1, out=None)
            parallels = num.around(parallels, decimals=1, out=None)

            ticks = map(meridians, parallels)

            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
            ax.set_xticklabels(meridians, rotation=45, fontsize=22)
            ax.set_yticklabels(parallels, fontsize=22)
            ax.tick_params(direction='out', length=6, width=4)
            plt.grid()
            if synthetic is True:
                from pyrocko.gf import RectangularSource

                srcs = load_all(filename='%s.yml' % name)

                for source in srcs:
                    n, e = source.outline(cs='latlon').T
                    e, n = map(e,n)
                    ax.fill(e, n, color=(0, 0, 0), lw = 3)

            addArrow(ax, scene)
            try:

                x0, y0 = map(x0c, y0c)
                x1, y1 = map(x1c, y1c)
                ax.set_xlim([x0, x1])
                ax.set_ylim([y0, y1])
            except:
                pass

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(cax=cax)

            fig = plt.gcf()
            fig.set_size_inches((11, 11), forward=False)
            plt.savefig(fname+'box.svg', format='svg', dpi=300)
            plt.close()

            return widths, lengths


def plot_on_map(db, scene, eastings, northings, x0, y0, x1, y1, mind, maxd,
                fname,
                synthetic=False, topo=False, kite_scene=False, comb=False):
            if kite_scene is True:
                scd = scene
                data_dsc= scd.displacement
            else:
                data_dsc= num.rot90(scene.T)

            if comb is True:
                data_dsc[data_dsc< num.max(data_dsc)*0.1] = num.nan

            else:
                data_dsc[data_dsc==0] = num.nan

            map = Basemap(projection='merc', llcrnrlon=num.min(eastings),llcrnrlat=num.min(northings),urcrnrlon=num.max(eastings),urcrnrlat=num.max(northings),
                          resolution='h', epsg=3395)
            ratio_lat = num.max(northings)/num.min(northings)
            ratio_lon = num.max(eastings)/num.min(eastings)
            try:
                map.drawmapscale(num.min(eastings)+2.1+ratio_lon*0.25, num.min(northings)+ratio_lat*0.18, num.mean(eastings), num.mean(northings), 30, fontsize=18, barstyle='fancy')
            except:
                map.drawmapscale(num.min(eastings)+0.05, num.min(northings)+0.04, num.mean(eastings), num.mean(northings), 10, fontsize=18, barstyle='fancy')

            try:
                parallels = num.linspace(y0, y1,22)
                meridians = num.linspace(x0, x1,22)
            except:
                parallels = num.linspace((num.min(northings)),(num.max(northings)),22)
                meridians = num.linspace((num.min(eastings)),(num.max(eastings)),22)

            xpixels = 800
            if topo is True:
                map.arcgisimage(service='World_Shaded_Relief', xpixels = xpixels, verbose= False)

            map.imshow(data_dsc, cmap="seismic", vmin=mind, vmax=maxd, alpha=0.8)
            ax = plt.gca()

            meridians = num.around(meridians, decimals=1, out=None)
            parallels = num.around(parallels, decimals=1, out=None)

            ticks = map(meridians, parallels)


            ax.set_xticks(ticks[0] )
            ax.set_yticks(ticks[1])
            ax.set_xticklabels(meridians, rotation=45, fontsize=22)
            ax.set_yticklabels(parallels, fontsize=22)
            ax.tick_params(direction='out', length=6, width=4)
            plt.grid()

            #map.drawparallels(parallels,labels=[1,0,0,0],fontsize=22)
            #map.drawmeridians(meridians,labels=[1,1,0,1],fontsize=22, rotation=45)
            if kite_scene is True:

                addArrow(ax, scene)
            try:

                x0, y0 = map(x0, y0)
                x1, y1 = map(x1, y1)
                ax.set_xlim([x0, x1])
                ax.set_ylim([y0, y1])
            except:
                pass

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(cax=cax)
            fig = plt.gcf()
            fig.set_size_inches((11, 11), forward=False)
            plt.savefig(fname+'.svg', format='svg', dpi=300)
            plt.close()

def load(path, kite_scene=True, grid=False, path_cc=None):

    if kite_scene is True:
        sc = Scene.load(path)
        img = sc.displacement
        unw = img.copy()
        where_are_NaNs = num.isnan(img)
        img[where_are_NaNs] = 0
        coh = img # TODO load in coherence
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

    sobelx = cv2.Sobel(src,cv2.CV_64F,1,0,ksize=31)
    sobely = cv2.Sobel(src,cv2.CV_64F,0,1,ksize=31)

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
    eastings, northings = num.vectorize(rc2en, otypes=[num.float, num.float])(rows, cols)
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


def process(img, coh, longs, lats, scene, x0, y0, x1, y1, fname, plot=True, coh_sharp=False, loading = False, topo = False, synthetic = False, calc_statistics = False, subsample = False):
    selem = rectangle(100,100)
    plt_img = img.copy()
    if coh_sharp is False:
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
        coh[coh < num.mean(coh)]=0
        coh_filt = filters.gaussian_filter(coh, 5, order=0)

    elif coh_sharp == 'basic':
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
        grad_mask[grad_mask !=0] = 1

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
    #    grad_mask = filters.gaussian_filter(grad_mask, 20, order=0)

        grad = grad/num.max(grad)
        grad_mask = filters.gaussian_filter(grad_mask, 20, order=0)
        grad_mask = grad_mask/num.max(grad_mask)
        grad_mask, mag_mask, ori_mask = get_gradient(ls)
        image = coh_filt*grad

    elif coh_sharp is True:
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
        quantized_img= ls
        grad_mask, mag_mask,ori_mask = get_gradient(quantized_img)
        selem = rectangle(100,100)

        grad, mag, ori = get_gradient(img)
        grad2, mag2, or2 = get_gradient(grad)
        grad2 = grad2/num.max(grad2)

        grad_mask[grad_mask !=0] = 1

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

    elif coh_sharp == "ss":
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
#        grad_mask, mag_mask, ori_mask = get_gradient(ls)

        grad, mag, ori = get_gradient(img)
        grad2, mag2, or2 = get_gradient(grad)
        grad2 = grad2/num.max(grad2)

        grad_mask[grad_mask !=0] = 1
        pointy = grad*grad_mask

        thres = num.max(pointy)*0.1
        pointy[pointy < thres] = 0
        image = pointy.copy()

        pointy2 = grad_mask/num.max(grad_mask)
        thres = num.max(pointy2)*0.1
        pointy2[pointy2 < thres] = 0
        pointy2[pointy2 > 0] = 1
        image = pointy+pointy2
        coh[coh < num.mean(coh)]=0
        coh_filt = filters.gaussian_filter(coh, 30, order=0)
        #image = image*coh_filt

        thres = num.max(pointy)*0.1
        pointy[pointy < thres] = 0
        image = pointy.copy()

        pointy2 = grad_mask/num.max(grad_mask)
        thres = num.max(pointy2)*0.1
        pointy2[pointy2 < thres] = 0
        pointy2[pointy2 > 0] = 1
        image = pointy+pointy2
        coh[coh < num.mean(coh)]=0
        coh_filt = filters.gaussian_filter(coh, 30, order=0)
        grad = filters.gaussian_filter(grad, 30, order=0)
#        image = image / num.sqrt(num.sum(image**2))
#        coh_filt = image / num.sqrt(num.sum(coh_filt**2))
#        grad = image / num.sqrt(num.sum(grad**2))

        image = image*coh_filt

    if plot is True:
            eastings = longs
            northings = lats
            fig = plt.figure()

            map = Basemap(projection='merc', llcrnrlon=num.min(eastings),llcrnrlat=num.min(northings),urcrnrlon=num.max(eastings),urcrnrlat=num.max(northings),
                          resolution='h', epsg=3395)

            ratio_lat = num.max(northings)/num.min(northings)
            ratio_lon = num.max(eastings)/num.min(eastings)
            try:
                map.drawmapscale(num.min(eastings)+2.1+ratio_lon*0.25, num.min(northings)+ratio_lat*0.18, num.mean(eastings), num.mean(northings), 30, fontsize=18, barstyle='fancy')
            except:
                map.drawmapscale(num.min(eastings)+0.05, num.min(northings)+0.04, num.mean(eastings), num.mean(northings), 10, fontsize=18, barstyle='fancy')

            try:
                parallels = num.linspace(y0,y1,22)
                meridians = num.linspace(x0, x1,22)
            except:
                parallels = num.linspace((num.min(northings)),(num.max(northings)),22)
                meridians = num.linspace((num.min(eastings)),(num.max(eastings)),22)
            xpixels = 800
            if topo is True:
                map.arcgisimage(service='World_Shaded_Relief', xpixels = xpixels, verbose= False)
            ls_dark[ls_dark==0] = num.nan
            ls_clear[ls_clear==0] = num.nan

            #map.imshow(plt_img, cmap='seismic', alpha=0.4)

            map.imshow(ls_dark, cmap='jet')
            map.imshow(ls_clear)

            ax = plt.gca()

            meridians = num.around(meridians, decimals=1, out=None)
            parallels = num.around(parallels, decimals=1, out=None)

            ticks = map(meridians, parallels)

            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
            ax.set_xticklabels(meridians, rotation=45, fontsize=22)
            ax.set_yticklabels(parallels, fontsize=22)
            ax.tick_params(direction='out', length=6, width=4)
            plt.grid()
            try:
                x0c, y0c = map(x0, y0)
                x1c, y1c = map(x1, y1)
                ax.set_xlim([x0c, x1c])
                ax.set_ylim([y0c, y1c])
            except:
                pass
            addArrow(ax, scene)
            fig = plt.gcf()
            fig.set_size_inches((11, 11), forward=False)
            plt.savefig(fname+'mask.svg', format='svg', dpi=300)
            plt.close()

            eastings = longs
            northings = lats
            fig = plt.figure()

            map = Basemap(projection='merc', llcrnrlon=num.min(eastings),llcrnrlat=num.min(northings),urcrnrlon=num.max(eastings),urcrnrlat=num.max(northings),
                          resolution='h', epsg=3395)

            ratio_lat = num.max(northings)/num.min(northings)
            ratio_lon = num.max(eastings)/num.min(eastings)
            try:
                map.drawmapscale(num.min(eastings)+2.1+ratio_lon*0.25, num.min(northings)+ratio_lat*0.18, num.mean(eastings), num.mean(northings), 30, fontsize=18, barstyle='fancy')
            except:
                map.drawmapscale(num.min(eastings)+0.05, num.min(northings)+0.04, num.mean(eastings), num.mean(northings), 10, fontsize=18, barstyle='fancy')

            try:
                parallels = num.linspace(y0,y1,22)
                meridians = num.linspace(x0, x1,22)
            except:
                parallels = num.linspace((num.min(northings)),(num.max(northings)),22)
                meridians = num.linspace((num.min(eastings)),(num.max(eastings)),22)
            xpixels = 800
            if topo is True:
                map.arcgisimage(service='World_Shaded_Relief',
                                xpixels=xpixels, verbose= False)
            ls_clear = grad.copy()
            ls_clear[ls_clear==0] = num.nan
        #    ls_clear[ls_clear<num.max(ls_clear)*0.001] = num.nan

            map.imshow(ls_clear, cmap="bone_r")

            ax = plt.gca()

            meridians = num.around(meridians, decimals=1, out=None)
            parallels = num.around(parallels, decimals=1, out=None)

            ticks = map(meridians, parallels)

            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
            ax.set_xticklabels(meridians, rotation=45, fontsize=22)
            ax.set_yticklabels(parallels, fontsize=22)
            ax.tick_params(direction='out', length=6, width=4)
            plt.grid()

            try:
                x0c, y0c = map(x0, y0)
                x1c, y1c = map(x1, y1)
                ax.set_xlim([x0c, x1c])
                ax.set_ylim([y0c, y1c])
            except:
                pass
            addArrow(ax, scene)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(cax=cax)
            fig = plt.gcf()
            fig.set_size_inches((11, 11), forward=False)
            plt.savefig(fname+'grad.svg', format='svg', dpi=300)
            plt.close()

            eastings = longs
            northings = lats
            fig = plt.figure()

            map = Basemap(projection='merc', llcrnrlon=num.min(eastings),llcrnrlat=num.min(northings),urcrnrlon=num.max(eastings),urcrnrlat=num.max(northings),
                          resolution='h', epsg=3395)

            ratio_lat = num.max(northings)/num.min(northings)
            ratio_lon = num.max(eastings)/num.min(eastings)
            try:
                map.drawmapscale(num.min(eastings)+2.1+ratio_lon*0.25, num.min(northings)+ratio_lat*0.18, num.mean(eastings), num.mean(northings), 30, fontsize=18, barstyle='fancy')
            except:
                map.drawmapscale(num.min(eastings)+0.05, num.min(northings)+0.04, num.mean(eastings), num.mean(northings), 10, fontsize=18, barstyle='fancy')

            try:
                parallels = num.linspace(y0,y1,22)
                meridians = num.linspace(x0, x1,22)
            except:
                parallels = num.linspace((num.min(northings)),(num.max(northings)),22)
                meridians = num.linspace((num.min(eastings)),(num.max(eastings)),22)
            xpixels = 800
            if topo is True:
                map.arcgisimage(service='World_Shaded_Relief', xpixels = xpixels, verbose= False)
            #ls_dark = ls_dark*-1.
            ls_clear = grad_mask.copy()
            ls_clear[ls_clear<num.max(ls_clear)*0.0000001] = num.nan

            #grad2, mag2, or2 = get_gradient(grad_mask)
            #grad2[grad2<num.max(grad2)*0.1] = num.nan

            map.imshow(ls_clear,  cmap="hot")
            map.imshow(plt_img, cmap='seismic', alpha=0.4)

            ax = plt.gca()

            meridians = num.around(meridians, decimals=1, out=None)
            parallels = num.around(parallels, decimals=1, out=None)

            ticks = map(meridians, parallels)

            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
            ax.set_xticklabels(meridians, rotation=45, fontsize=22)
            ax.set_yticklabels(parallels, fontsize=22)
            ax.tick_params(direction='out', length=6, width=4)
            plt.grid()
            try:
                x0c, y0c = map(x0, y0)
                x1c, y1c = map(x1, y1)
                ax.set_xlim([x0c, x1c])
                ax.set_ylim([y0c, y1c])
            except:
                pass
            addArrow(ax, scene)
            fig = plt.gcf()
            fig.set_size_inches((11, 11), forward=False)
            plt.savefig(fname+'mask_grad.svg', format='svg', dpi=300)
            plt.close()

            eastings = longs
            northings = lats
            fig = plt.figure()

            map = Basemap(projection='merc', llcrnrlon=num.min(eastings),llcrnrlat=num.min(northings),urcrnrlon=num.max(eastings),urcrnrlat=num.max(northings),
                          resolution='h', epsg=3395)

            ratio_lat = num.max(northings)/num.min(northings)
            ratio_lon = num.max(eastings)/num.min(eastings)
            try:
                map.drawmapscale(num.min(eastings)+2.1+ratio_lon*0.25, num.min(northings)+ratio_lat*0.18, num.mean(eastings), num.mean(northings), 30, fontsize=18, barstyle='fancy')
            except:
                map.drawmapscale(num.min(eastings)+0.05, num.min(northings)+0.04, num.mean(eastings), num.mean(northings), 10, fontsize=18, barstyle='fancy')

            try:
                parallels = num.linspace(y0,y1,22)
                meridians = num.linspace(x0, x1,22)
            except:
                parallels = num.linspace((num.min(northings)),(num.max(northings)),22)
                meridians = num.linspace((num.min(eastings)),(num.max(eastings)),22)
            xpixels = 800
            if topo is True:

                map.arcgisimage(service='World_Shaded_Relief', xpixels = xpixels, verbose= False)
            #ls_dark = ls_dark*-1.
            ls_clear = coh_filt.copy()
            #ls_clear[ls_clear<num.max(ls_clear)*0.01] = num.nan


            map.imshow(ls_clear, cmap="seismic")

            ax = plt.gca()

            meridians = num.around(meridians, decimals=1, out=None)
            parallels = num.around(parallels, decimals=1, out=None)

            ticks = map(meridians, parallels)

            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
            ax.set_xticklabels(meridians, rotation=45, fontsize=22)
            ax.set_yticklabels(parallels, fontsize=22)
            ax.tick_params(direction='out', length=6, width=4)
            plt.grid()
            try:
                x0c, y0c = map(x0, y0)
                x1c, y1c = map(x1, y1)
                ax.set_xlim([x0c, x1c])
                ax.set_ylim([y0c, y1c])
            except:
                pass
            addArrow(ax, scene)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(cax=cax)
            fig = plt.gcf()
            fig.set_size_inches((11, 11), forward=False)
            plt.savefig(fname+'filt.svg', format='svg', dpi=300)
            plt.close()

            eastings = longs
            northings = lats
            fig = plt.figure()

            map = Basemap(projection='merc', llcrnrlon=num.min(eastings),llcrnrlat=num.min(northings),urcrnrlon=num.max(eastings),urcrnrlat=num.max(northings),
                          resolution='h', epsg=3395)

            ratio_lat = num.max(northings)/num.min(northings)
            ratio_lon = num.max(eastings)/num.min(eastings)
            try:
                map.drawmapscale(num.min(eastings)+2.1+ratio_lon*0.25, num.min(northings)+ratio_lat*0.18, num.mean(eastings), num.mean(northings), 30, fontsize=18, barstyle='fancy')
            except:
                map.drawmapscale(num.min(eastings)+0.05, num.min(northings)+0.04, num.mean(eastings), num.mean(northings), 10, fontsize=18, barstyle='fancy')

            try:
                parallels = num.linspace(y0,y1,22)
                meridians = num.linspace(x0, x1,22)
            except:
                parallels = num.linspace((num.min(northings)),(num.max(northings)),22)
                meridians = num.linspace((num.min(eastings)),(num.max(eastings)),22)
            xpixels = 800
            if topo is True:

                map.arcgisimage(service='World_Shaded_Relief', xpixels = xpixels, verbose= False)
            #ls_dark = ls_dark*-1.
            ls_clear = image.copy()
            ls_clear = ls_clear / num.sqrt(num.sum(ls_clear**2))
            ls_clear[ls_clear<num.max(ls_clear)*0.01] = num.nan


            map.imshow(ls_clear, cmap="hot")

            ax = plt.gca()

            meridians = num.around(meridians, decimals=1, out=None)
            parallels = num.around(parallels, decimals=1, out=None)

            ticks = map(meridians, parallels)

            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
            ax.set_xticklabels(meridians, rotation=45, fontsize=22)
            ax.set_yticklabels(parallels, fontsize=22)
            ax.tick_params(direction='out', length=6, width=4)
            plt.grid()
            try:
                x0c, y0c = map(x0, y0)
                x1c, y1c = map(x1, y1)
                ax.set_xlim([x0c, x1c])
                ax.set_ylim([y0c, y1c])
            except:
                pass
            addArrow(ax, scene)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(cax=cax)
            fig = plt.gcf()
            fig.set_size_inches((11, 11), forward=False)
            plt.savefig(fname+'dir-comb.svg', format='svg', dpi=300)
            plt.close()

    #image = image / num.sqrt(num.sum(image**2))

    return image


def writeout(image, fname, sc=None):
    # # save as grid file

    arr = num.rot90(image.T)

    if sc is not None:
        ulNutm = sc.frame.llNutm+sc.frame.dNmeter*sc.rows
        transform = from_origin(sc.frame.llEutm,ulNutm, sc.frame.dEmeter, sc.frame.dNmeter)
        new_dataset = rasterio.open(fname, 'w', driver='GTiff',
                                    height = sc.rows, width = sc.cols,
                                    count=1, dtype=str(arr.dtype),
                                    crs='+proj=utm +zone=%s +units=m +no_defs' %(sc.frame.utm_zone),
                                    transform=transform)

        new_dataset.write(arr, 1)
        new_dataset.close()

def combine(img_asc_path, img_dsc_path, name, weight_asc=1, weight_dsc=1, plot=False):
    print('Merging ascending and descending outputs with gdal')

    subprocess.run(["gdal_merge.py", "-o", "work-%s/comb.tif" %name, img_asc_path, img_dsc_path, "-seperate"])
    subprocess.run(['gdal_calc.py', '--calc=%s*A+%s*B' % (weight_asc, weight_dsc), "--outfile=work-%s/merged.tiff" % (name), '-A', "work-%s/comb.tif" %name,
    "--A_band=1", "-B", "work-%s/comb.tif" %name, "--B_band=2", '--overwrite'])
    fname = 'work-%s/merged.tiff' %name
    comb = rasterio.open('work-%s/merged.tiff' %name)
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
    newlist = sorted(regionprops(label_image), key=lambda region: region.area, reverse=True)
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
        if region.area >= area: #check if nec.

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
            coords.append([x1,y1])
            coords.append([x1a,y1a])
            coords.append([x0,y0])
            coords.append([x2,y2])
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
        max_bound = [num.min(minrs), num.min(mincs),  num.max(maxrs), num.max(maxcs)]


        thresh = threshold_otsu(image)
        bw = closing(image > num.max(image)*0.1, square(80))
        if area is None:
            area = 900

        label_image = label(bw)
        image_label_overlay = label2rgb(label_image, image=image)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image_label_overlay)
        newlist = sorted(regionprops(label_image), key=lambda region: region.area, reverse=True)

        region = newlist[0]
        if region.area >= area: #check if nec.

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
            coords.append([x1,y1])
            coords.append([x1a,y1a])
            coords.append([x0,y0])
            coords.append([x2,y2])
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
        if region.area >= 400: #check if nec.
            coords = num.array(region.coords)
            poly = geometry.Polygon([[p[0], p[1]] for p in coords])
            polys.append(poly)
            hull = poly.convex_hull

            koor = hull.exterior.coords
            pol = geometry.Polygon([[p[0], p[1]] for p in koor])
            pol = pol.buffer(-1)
            pol = pol.simplify(4000, preserve_topology=True) #check
            try:
                center = Centerline(pol)
                centers.append(center)
            except:
                pass
        if region.area >= 400: #check; should be scaled to pixel size?
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
    comp_fault=[]
    for center in centers:
        coords=[]
        for line in center:
            coord= line.coords
            coords.append(coord)
        coords= num.array(coords)
        hand = num.array([(coords[i][m]) for i in range(num.shape(coords)[0]) for m in range(num.shape(coords[i])[0])])
        new_hand = hand.copy()
        for _ in range(1):
            new_hand = subdivide_polygon(new_hand, degree=2, preserve_ends=True)
            new_hand  = approximate_polygon(new_hand , tolerance=0.8)
        comp_fault.append(new_hand)

        appr_hand = hand.copy()
        # approximate subdivided polygon with Douglas-Peucker algorithm
        appr_hand = approximate_polygon(appr_hand, tolerance=10)

        ax1.scatter(appr_hand[:, 1], appr_hand[:, 0])
        simp_fault.append(appr_hand[:, :])


        ax2.scatter(new_hand[:, 1], new_hand[:, 0])
    ax1.set_title('Simple line')
    ax2.set_title('More complexity')
    plt.close()
    return simp_fault, comp_fault

def df_to_geojson(df, eastings, northings, properties=None):
    coords_fault = []
    n = 0
    geojson = {'type':'FeatureCollection', 'features':[]}
    for p in df:
        feature = {'type':'Feature',
                   'properties':{'ogc_fid':n,
                                 'ns_average_dip': 60,
                                 'ns_average_rake':120},
                   'geometry':{'type':'LineString',
                               'coordinates':[]}}
        for k in p:
            kx= num.asarray(k[0])
            ky= num.asarray(k[1])
            feature['geometry']['coordinates'] = [(eastings[int(kx)][int(ky)], northings[int(kx)][int(ky)])]
            coords_fault.append((eastings[int(kx)][int(ky)], northings[int(kx)][int(ky)]))
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
    with open('work-%s/fault_lines' %name, 'w') as f:
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
    i = i +1
    label_image = label(bw)
    image_label_overlay = label2rgb(label_image, image=grad)

    newlist = sorted(regionprops(label_image), key=lambda region: region.area, reverse=True)
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
            calc_statistics=True
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
                          x0, y0, x1, y1, fname, plot=plot, coh_sharp=sharp,
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

        fname = 'work-%s/dsc.mod.tif' %name
        writeout(img_dsc, fname, sc=scene_dsc)
        longs_dsc, lats_dsc = to_latlon(fname)
        fname = 'work-%s/dsc-' %name
        img_dsc = process(img_dsc, coh_dsc, longs_dsc, lats_dsc, scene_dsc,
                          x0, y0, x1, y1, fname, plot=plot, coh_sharp=sharp,
                          loading=loading, topo=topo, synthetic=synthetic,
                          calc_statistics=calc_statistics, subsample=subsample)
        fname = 'work-%s/dsc.mod.tif' %name
        writeout(img_dsc, fname, sc=scene_dsc)

        db =1
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
            plot_on_map(db, scene_asc, longs_asc, lats_asc, x0, y0 ,x1, y1, minda, maxa, fname,
                        synthetic=synthetic, topo=topo, kite_scene=True)
            fname = 'work-%s/dsc' % name
            plot_on_map(db, scene_dsc, longs_dsc, lats_dsc, x0, y0, x1, y1, mindd, maxdd, fname,
                        synthetic=synthetic, topo=topo, kite_scene=True)


        fname = 'work-%s/asc.mod.tif' % name
        comb = rasterio.open(fname)
        longs_comb, lats_comb = to_latlon(fname)
        comb_img = comb.read(1)

        centers_bounding, coords_out, coords_box, strike, ellipses, max_bound = bounding_box(comb_img,
                                                                        400, sharp)
        for st in strike:
            strikes.append(st)
        print("Strike(s) of moment weighted centerline(s) are :%s" % strike)

        if plot is True:
            fname = 'work-%s/asc-comb-' %name

            plot_on_kite_box(coords_box, coords_out, scene_asc, longs_asc,
                             lats_asc, longs_comb, lats_comb, x0,y0,x1,y1,
                             name, ellipses, minda, maxa, fname,
                             synthetic=synthetic, topo=topo)

        fname = 'work-%s/dsc.mod.tif' %name
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


        comb_img = combine('work-%s/asc.mod.tif' % name, 'work-%s/dsc.mod.tif' % name, name, weight_asc=snr_asc, weight_dsc= snr_dsc, plot=False)
        longs_comb, lats_comb = to_latlon("work-%s/merged.tiff" % name)

    else:
        fname = 'work-%s/merged.tiff' %name
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
            #sc.save('kite_scene')

            # Or export the quadtree to CSV file
            #qt.export('/tmp/tree.csv')



    # statistical output
    #img_asc, coh_asc, scene_asc = load('muji_kite/asc', kite_scene=True)
    #comb_img = process(img_asc, coh_asc, plot=True)
    # use quadtree subsampling on gradient

    img_asc, coh_asc, scene_asc, dates = load(sys.argv[1], kite_scene=True)
    fname = 'work-%s/asc.mod.tif' %name
    #writeout(img_asc, fname, sc=scene_asc)
    longs_asc, lats_asc = to_latlon(fname)
    db =1
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
        fobj_cum = open(os.path.join('work-%s/grad_grid.ASC' % name),'w')
        for x, y, sembcums in zip(es, ns, comb_img_grid.flatten()):
            fobj_cum.write('%.2f %.2f %.20f\n' % (x, y, sembcums))
        fobj_cum.close()

        fobj_cum = open(os.path.join('work-%s/grad_grid_resam.ASC' % name), 'w')
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

    db = dump_geojson(simp_fault, longs_comb, lats_comb, name) #check
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
