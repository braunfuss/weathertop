from kite import Scene
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib as mpl
from cartopy.io import srtm
from cartopy.io import PostprocessedRasterSource, LocatedImage
from cartopy.io.srtm import SRTM3Source, SRTM1Source
from matplotlib import rc
import numpy as num
import cartopy.geodesic as cgeo
import rasterio
import rasterio.features
import rasterio.warp
import cartopy.crs as ccrs
import cartopy
from pyrocko import orthodrome
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm


def plot_process(longs, lats, scene, ls_dark, ls_clear, grad_mask, image,
                 grad, plt_img, coh_filt, fname, topo=False):
    eastings = longs
    northings = lats
    fig = plt.figure()

    extent = [num.min(eastings), num.max(eastings), num.min(northings),
              num.max(northings)]
    central_lon = num.mean(extent[:2])
    central_lat = num.mean(extent[2:])
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)

    if topo is True:
        ax = add_topo(ax)
    ls_dark[ls_dark == 0] = num.nan
    ls_clear[ls_clear == 0] = num.nan
    scale_bar(ax, (0.1, 0.1), 5_0)
    ax.imshow(num.rot90(ls_dark.T), origin='upper', extent=extent,
              transform=ccrs.PlateCarree(), cmap='jet')
    ax.imshow(num.rot90(ls_clear.T), origin='upper', extent=extent,
              transform=ccrs.PlateCarree())

    ax.gridlines(draw_labels=True)

    addArrow(ax, scene)
    fig = plt.gcf()
    fig.set_size_inches((11, 11), forward=False)
    plt.savefig(fname+'mask.svg', format='svg', dpi=300)
    plt.close()

    extent = [num.min(eastings), num.max(eastings), num.min(northings),
              num.max(northings)]
    central_lon = num.mean(extent[:2])
    central_lat = num.mean(extent[2:])

    f, ax = plt.subplots(1, 1,
                         subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent(extent)

    if topo is True:
        ax = add_topo(ax)
    ls_dark[ls_dark == 0] = num.nan
    ls_clear[ls_clear == 0] = num.nan
    scale_bar(ax, (0.1, 0.1), 5_0)
    ls_clear = grad.copy()
    ls_clear[ls_clear == 0] = num.nan

    h = ax.imshow(num.rot90(ls_clear.T), origin='upper', extent=extent,
                  transform=ccrs.PlateCarree(), cmap="bone_r")

    gl = ax.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.xlabels_top = False
    addArrow(ax, scene)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    f.add_axes(cax)
    plt.colorbar(h, cax=cax)
    fig = plt.gcf()
    fig.set_size_inches((11, 11), forward=False)
    plt.savefig(fname+'grad.svg', format='svg', dpi=300)
    plt.close()

    eastings = longs
    northings = lats
    fig = plt.figure()

    extent = [num.min(eastings), num.max(eastings), num.min(northings),
              num.max(northings)]
    central_lon = num.mean(extent[:2])
    central_lat = num.mean(extent[2:])
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)

    if topo is True:
        ax = add_topo(ax)
    ls_clear = grad_mask.copy()
    ls_clear[ls_clear < num.max(ls_clear)*0.0000001] = num.nan

    ax.imshow(num.rot90(ls_clear.T), origin='upper', extent=extent,
              transform=ccrs.PlateCarree(), cmap="hot")
    h = ax.imshow(num.rot90(plt_img.T), origin='upper', extent=extent,
                  transform=ccrs.PlateCarree(), cmap="seismic", alpha=0.4)

    scale_bar(ax, (0.1, 0.1), 5_0)
    ax.gridlines(draw_labels=True)
    addArrow(ax, scene)
    fig = plt.gcf()
    fig.set_size_inches((11, 11), forward=False)
    plt.savefig(fname+'mask_grad.svg', format='svg', dpi=300)
    plt.close()

    eastings = longs
    northings = lats
    fig = plt.figure()
    extent = [num.min(eastings), num.max(eastings), num.min(northings),
              num.max(northings)]
    central_lon = num.mean(extent[:2])
    central_lat = num.mean(extent[2:])
    f, ax = plt.subplots(1, 1,
                         subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent(extent)

    if topo is True:
        ax = add_topo(ax)

    ls_clear = coh_filt.copy()

    h = ax.imshow(num.rot90(ls_clear.T), origin='upper', extent=extent,
                  transform=ccrs.PlateCarree(), cmap='seismic')

    scale_bar(ax, (0.1, 0.1), 5_0)
    gl = ax.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.xlabels_top = False
    addArrow(ax, scene)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    f.add_axes(cax)
    plt.colorbar(h, cax=cax)

    fig = plt.gcf()
    fig.set_size_inches((11, 11), forward=False)
    plt.savefig(fname+'filt.svg', format='svg', dpi=300)
    plt.close()

    eastings = longs
    northings = lats
    fig = plt.figure()
    extent = [num.min(eastings), num.max(eastings), num.min(northings),
              num.max(northings)]
    central_lon = num.mean(extent[:2])
    central_lat = num.mean(extent[2:])
    f, ax = plt.subplots(1, 1,
                         subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent(extent)

    if topo is True:
        ax = add_topo(ax)
    ls_clear = image.copy()
    ls_clear = ls_clear / num.sqrt(num.sum(ls_clear**2))
    ls_clear[ls_clear < num.max(ls_clear)*0.01] = num.nan

    ax.imshow(num.rot90(ls_clear.T), origin='upper', extent=extent,
              transform=ccrs.PlateCarree(), cmap='hot')
    scale_bar(ax, (0.1, 0.1), 5_0)
    gl = ax.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.xlabels_top = False
    addArrow(ax, scene)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    f.add_axes(cax)
    plt.colorbar(h, cax=cax)

    fig = plt.gcf()
    fig.set_size_inches((11, 11), forward=False)
    plt.savefig(fname+'dir-comb.svg', format='svg', dpi=300)
    plt.close()


    def add_topo(ax):
        # shade function when the data is retrieved.
        shaded_srtm = PostprocessedRasterSource(SRTM1Source(), shade)
        # Add the shaded SRTM source to our map with a grayscale colormap.
        ax.add_raster(shaded_srtm, cmap='Greys')
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
        ax.add_feature(cartopy.feature.RIVERS)
        return ax


class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return num.ma.masked_array(num.interp(value, x, y), num.isnan(value))


def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if num.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than "
                         f"given distance ({distance}).")

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not num.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    # Direction vector of the line in axes coordinates.
    direction = num.array([num.cos(angle), num.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys).base[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
              tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
              ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
              **kwargs):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates.
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
                   **kwargs}
    text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
                   **text_kwargs, **kwargs}

    # Convert all units and types.
    location = num.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * num.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad,
                            tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * num.array([-num.sin(angle_rad), num.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location, f"{length} {unit_name}", rotation_mode='anchor',
            transform=ax.transAxes, **text_kwargs)


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


def plot_on_kite_scatter(db, scene, eastings, northings, x0, y0, x1, y1, mind,
                         maxd, fname,
                         synthetic=False, topo=False):
    '''
    Plotting function for plotting scatter points from a database (db)
    with coordinates eastings and northings on data from a kite scene withhin
    frame given by x0, y0, x1 and y1 and saves
    the image to a folder under name fname. Optional draw of topography.
    '''
    scd = scene
    data_dsc = scd.displacement

    data_dsc[data_dsc == 0] = num.nan

    extent = [num.min(eastings), num.max(eastings), num.min(northings),
              num.max(northings)]
    central_lon = num.mean(extent[:2])
    central_lat = num.mean(extent[2:])
    f, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent(extent)
    if topo is True:
        # shade function when the data is retrieved.
        shaded_srtm = PostprocessedRasterSource(SRTM1Source(), shade)
        # Add the shaded SRTM source to our map with a grayscale colormap.
        ax.add_raster(shaded_srtm, cmap='Greys')
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
        ax.add_feature(cartopy.feature.RIVERS)

    h = ax.imshow(num.rot90(data_dsc.T), origin='upper', extent=extent,
                  transform=ccrs.PlateCarree(), cmap="seismic", alpha=0.8,
                  norm=MidpointNormalize(mind, maxd, 0.))

    gl = ax.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.xlabels_top = False
    addArrow(ax, scene)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    f.add_axes(cax)
    plt.colorbar(h, cax=cax)

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
        x, y = coords_re_x, coords_re_y
        plt.scatter(x, y, c=next(colors))

    plt.grid()

    addArrow(ax, scene)
    try:
        x0, y0 = map(x0, y0)
        x1, y1 = map(x1, y1)
        ax.set_xlim([x0, x1])
        ax.set_ylim([y0, y1])
    except:
        pass
    divider = make_axes_locatable(ax)
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
    '''
    Plotting function for plotting a line from coordinates coords_out,
    with coordinates eastings and northings on data from a kite scene withhin
    frame given by x0c, y0c, x1c and y1c and saves
    the image to a folder under name fname. Optional draw of topography.
    '''
    scd = scene
    data_dsc = scd.displacement

    data_dsc[data_dsc == 0] = num.nan
    extent = [num.min(eastings), num.max(eastings), num.min(northings),
              num.max(northings)]
    central_lon = num.mean(extent[:2])
    central_lat = num.mean(extent[2:])
    f, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent(extent)

    if topo is True:
        # shade function when the data is retrieved.
        shaded_srtm = PostprocessedRasterSource(SRTM1Source(), shade)
        # Add the shaded SRTM source to our map with a grayscale colormap.
        ax.add_raster(shaded_srtm, cmap='Greys')
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
        ax.add_feature(cartopy.feature.RIVERS)

    scale_bar(ax, (0.1, 0.1), 5_0)
    h = ax.imshow(num.rot90(data_dsc.T), origin='upper', extent=extent,
                  transform=ccrs.PlateCarree(), cmap="seismic", vmin=mind,
                  vmax=maxd, alpha=0.8)

    ax.gridlines(draw_labels=True)

    coords_all = []
    for coords in coords_out:

        coords_boxes = []
        for k in coords:
            kx = k[1]
            ky = k[0]
            coords_boxes.append([eastcomb[int(kx)][int(ky)],
                                 northcomb[int(kx)][int(ky)]])
        coords_all.append(coords_boxes)
    n = 0

    for coords in coords_all:

        x1, y1 = coords[0][0], coords[0][1]
        x1a, y1a = coords[1][0], coords[1][1]
        x0, y0 = coords[2][0], coords[2][1]
        x2, y2 = coords[3][0], coords[3][1]
        n = n+1
        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x1a), (y0, y1a), '-r', linewidth=2.5)

        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

    gl = ax.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.xlabels_top = False
    addArrow(ax, scene)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    f.add_axes(cax)
    plt.colorbar(h, cax=cax)

    addArrow(ax, scene)

    fig = plt.gcf()
    fig.set_size_inches((11, 11), forward=False)
    plt.savefig(fname+'line.svg', format='svg', dpi=300)
    plt.close()


def plot_on_kite_box(coords_out, coords_line, scene, eastings, northings,
                     eastcomb, northcomb, x0c, y0c, x1c, y1c, name, ellipses,
                     mind, maxd, fname, synthetic=False, topo=False):
    '''
    Plotting function for plotting a rectangle from coords_out and coords_line
    with coordinates eastings and northings on data from a kite scene withhin
    frame given by x0, y0, x1 and y1 and saves
    the image to a folder under name fname. Optional draw of topography.
    '''
    scd = scene
    data_dsc = scd.displacement
    lengths = []
    widths = []

    data_dsc[data_dsc == 0] = num.nan
    extent = [num.min(eastings), num.max(eastings), num.min(northings),
              num.max(northings)]
    central_lon = num.mean(extent[:2])
    central_lat = num.mean(extent[2:])
    f, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent(extent)

    if topo is True:
        # shade function when the data is retrieved.
        shaded_srtm = PostprocessedRasterSource(SRTM1Source(), shade)
        # Add the shaded SRTM source to our map with a grayscale colormap.
        ax.add_raster(shaded_srtm, cmap='Greys')
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
        ax.add_feature(cartopy.feature.RIVERS)

    scale_bar(ax, (0.1, 0.1), 5_0)
    h = ax.imshow(num.rot90(data_dsc.T), origin='upper', extent=extent,
                  transform=ccrs.PlateCarree(), cmap="seismic",
                  vmin=mind, vmax=maxd,
                  alpha=0.8, norm=MidpointNormalize(mind, maxd, 0.))

    ax.gridlines(draw_labels=True)

    coords_all = []
    for coords in coords_line:
        coords_boxes = []
        for k in coords:
            kx = k[1]
            ky = k[0]
            coords_boxes.append([eastcomb[int(kx)][int(ky)],
                                 northcomb[int(kx)][int(ky)]])
        coords_all.append(coords_boxes)
    n = 0
    for coords, ell in zip(coords_all, ellipses):

        x1, y1 = coords[0][0], coords[0][1]
        x1a, y1a = coords[1][0], coords[1][1]
        x0, y0 = coords[2][0], coords[2][1]
        x2, y2 = coords[3][0], coords[3][1]
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
        minc, minr = coords_boxes[0+n][0], coords_boxes[0+n][1]
        maxc, maxr = coords_boxes[1+n][0], coords_boxes[1+n][1]

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

    if synthetic is True:
        from pyrocko.gf import RectangularSource

        srcs = load_all(filename='%s.yml' % name)

        for source in srcs:
            n, e = source.outline(cs='latlon').T
            ax.fill(e, n, color=(0, 0, 0), lw = 3)

    addArrow(ax, scene)

    gl = ax.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.xlabels_top = False
    addArrow(ax, scene)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    f.add_axes(cax)
    plt.colorbar(h, cax=cax)

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
        data_dsc = scd.displacement
    else:
        data_dsc = num.rot90(scene.T)

    if comb is True:
        data_dsc[data_dsc < num.max(data_dsc)*0.1] = num.nan

    else:
        data_dsc[data_dsc == 0] = num.nan

    extent = [num.min(eastings), num.max(eastings), num.min(northings),
              num.max(northings)]
    central_lon = num.mean(extent[:2])
    central_lat = num.mean(extent[2:])
    f, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.set_extent(extent)

    if topo is True:
        # shade function when the data is retrieved.
        shaded_srtm = PostprocessedRasterSource(SRTM1Source(), shade)
        # Add the shaded SRTM source to our map with a grayscale colormap.
        ax.add_raster(shaded_srtm, cmap='Greys')
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
        ax.add_feature(cartopy.feature.RIVERS)
    scale_bar(ax, (0.1, 0.1), 5_0)

    h = ax.imshow(num.rot90(data_dsc.T), origin='upper', extent=extent,
                  transform=ccrs.PlateCarree(), cmap="seismic", vmin=mind,
                  vmax=maxd, alpha=0.8)

    if kite_scene is True:
        addArrow(ax, scene)

    gl = ax.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.xlabels_top = False
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    f.add_axes(cax)
    plt.colorbar(h, cax=cax)

    fig = plt.gcf()
    fig.set_size_inches((11, 11), forward=False)
    plt.savefig(fname+'.svg', format='svg', dpi=300)
    plt.close()
