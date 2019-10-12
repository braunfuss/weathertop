import math
from pyrocko import gf
from pyrocko.guts import List
from pyrocko import orthodrome
# for data from kite:
from kite import Scene

import os.path
from kite.scene import Scene, FrameConfig
from pyrocko import gf
import numpy as num


import matplotlib.pyplot as plt




def get_noise(scene):
    amplitude = 1

    beta = [5./3, 8./3, 2./3]
    regimes = [.15, .99, 1.]
    nE = scene.frame.rows
    nN = scene.frame.cols

    if (nE+nN) % 2 != 0:
        raise ArithmeticError('Dimensions of synthetic scene must '
                              'both be even!')

    dE = scene.frame.dE
    dN = scene.frame.dN

    rfield = num.random.rand(nE, nN)
    spec = num.fft.fft2(rfield)

    kE = num.fft.fftfreq(nE, dE)
    kN = num.fft.fftfreq(nN, dN)
    k_rad = num.sqrt(kN[:, num.newaxis]**2 + kE[num.newaxis, :]**2)

    regimes = num.array(regimes)

    k0 = 0.
    k1 = regimes[0] * k_rad.max()
    k2 = regimes[1] * k_rad.max()

    r0 = num.logical_and(k_rad > k0, k_rad < k1)
    r1 = num.logical_and(k_rad >= k1, k_rad < k2)
    r2 = k_rad >= k2

    beta = num.array(beta)
    # From Hanssen (2001)
    #   beta+1 is used as beta, since, the power exponent
    #   is defined for a 1D slice of the 2D spectrum:
    #   austin94: "Adler, 1981, shows that the surface profile
    #   created by the intersection of a plane and a
    #   2-D fractal surface is itself fractal with
    #   a fractal dimension  equal to that of the 2D
    #   surface decreased by one."
    beta += 1.
    # From Hanssen (2001)
    #   The power beta/2 is used because the power spectral
    #   density is proportional to the amplitude squared
    #   Here we work with the amplitude, instead of the power
    #   so we should take sqrt( k.^beta) = k.^(beta/2)  RH
    # beta /= 2.

    amp = num.zeros_like(k_rad)
    amp[r0] = k_rad[r0] ** -beta[0]
    amp[r0] /= amp[r0].max()

    amp[r1] = k_rad[r1] ** -beta[1]
    amp[r1] /= amp[r1].max() / amp[r0].min()

    amp[r2] = k_rad[r2] ** -beta[2]
    amp[r2] /= amp[r2].max() / amp[r1].min()

    amp[k_rad == 0.] = amp.max()

    spec *= amplitude * num.sqrt(amp)
    noise = num.abs(num.fft.ifft2(spec))
    noise -= num.mean(noise)

    return noise


class CombiSource(gf.Source):
    '''Composite source model.'''

    discretized_source_class = gf.DiscretizedMTSource

    subsources = List.T(gf.Source.T())

    def __init__(self, subsources=[], **kwargs):

        if subsources:

            lats = num.array(
                [subsource.lat for subsource in subsources], dtype=num.float)
            lons = num.array(
                [subsource.lon for subsource in subsources], dtype=num.float)

            assert num.all(lats == lats[0]) and num.all(lons == lons[0])
            lat, lon = lats[0], lons[0]

            # if not same use:
            # lat, lon = center_latlon(subsources)

            depth = float(num.mean([p.depth for p in subsources]))
            t = float(num.mean([p.time for p in subsources]))
            kwargs.update(time=t, lat=float(lat), lon=float(lon), depth=depth)

        gf.Source.__init__(self, subsources=subsources, **kwargs)

    def get_factor(self):
        return 1.0

    def discretize_basesource(self, store, target=None):
        dsources = []
        t0 = self.subsources[0].time
        for sf in self.subsources:
            assert t0 == sf.time
            ds = sf.discretize_basesource(store, target)
            ds.m6s *= sf.get_factor()
            dsources.append(ds)

        return gf.DiscretizedMTSource.combine(dsources)




def create_kite_scene_dsc(store_id, dip, depth, patches, llLat=0.,
                                      llLon=0.):

    km = 1e3
    d2r = num.pi/180.
    engine = gf.LocalEngine(store_superdirs=['.'])
    # Define the scene's frame
    frame = FrameConfig(
        # Lower left geographical reference [deg]
        llLat=llLat, llLon=llLon,
        # Pixel spacing [m] or [degrees]
        spacing='degrees', dE=550, dN=550)

    npx_east = 1400
    npx_north = 1400

    displacement = num.empty((npx_east, npx_north))

    theta = num.full_like(displacement, 50.*d2r)
    # Phi is azimuth towards the satellite, counter-clockwise from East
    phi = num.full_like(displacement, -13.*d2r)

    scene = Scene(
        displacement=displacement,
        phi=phi, theta=theta,
        frame=frame)

    satellite_target = gf.KiteSceneTarget(
        scene,
        store_id=store_id)

    sources = CombiSource(subsources=patches)


    # Forward model!
    result = engine.process(
        sources, satellite_target,
        # Use all available cores
        nthreads=0)

    kite_scenes = result.kite_scenes()
    return kite_scenes


def create_kite_scene_asc(store_id, dip, depth, patches, llLat=0.,
                                      llLon=0.):

    km = 1e3
    d2r = num.pi/180.
    engine = gf.LocalEngine(store_superdirs=['.'])
    # Define the scene's frame
    frame = FrameConfig(
        # Lower left geographical reference [deg]
        llLat=llLat, llLon=llLon,
        # Pixel spacing [m] or [degrees]
        spacing='degrees', dE=550, dN=550)

    # Resolution of the scene
    npx_east = 1400
    npx_north = 1400

    # 2D arrays for displacement and look vector
    displacement = num.empty((npx_east, npx_north))

    # Look vectors
    # Theta is elevation angle from horizon
    theta = num.full_like(displacement, 56.*d2r)
    # Phi is azimuth towards the satellite, counter-clockwise from East
    phi = num.full_like(displacement, -166.*d2r)

    scene = Scene(
        displacement=displacement,
        phi=phi, theta=theta,
        frame=frame)

    satellite_target = gf.KiteSceneTarget(
        scene,
        store_id=store_id)

    sources = CombiSource(subsources=patches)

    result = engine.process(
        sources, satellite_target,
        # Use all available cores
        nthreads=0)

    kite_scenes = result.kite_scenes()
    return kite_scenes


store_id = 'gf_abruzzo_nearfield_vmod_Ameri'
if not os.path.exists(store_id):
    gf.ws.download_gf_store(site='kinherd', store_id=store_id)

km = 1e3

dip = 88
depth = 2

patches = []
rect_source = gf.RectangularSource(
    # Geographical position [deg]
    lat=0., lon=0.,
    # Relative cartesian offsets [m]
    north_shift=16*km, east_shift=8*km,
    depth=depth*km,
    # Dimensions of the fault [m]
    width=1*km, length=3*km, # what is width /length ratio
    strike=104., dip=dip, rake=-160.,
    # Slip in [m]
    slip=1., anchor='top')
patches.append(rect_source)

rect_source2 = gf.RectangularSource(
    # Geographical position [deg]
    lat=0., lon=0.,
    # Relative cartesian offsets [m]
    north_shift=16*km, east_shift=22*km,
    depth=depth*km,
    # Dimensions of the fault [m]
    width=1*km, length=3*km, # what is width /length ratio
    strike=95., dip=dip, rake=-160.,
    # Slip in [m]
    slip=1., anchor='top')
patches.append(rect_source2)

kite_scene = create_kite_scene_dsc(store_id, dip, depth, patches)
noise = get_noise(kite_scene[0])
kite_scene[0].displacement = kite_scene[0].displacement + noise
kite_scene[0].save('dsc_two_sources_syn_ss_%s_%s'% (dip,depth))

kite_scene = create_kite_scene_asc(store_id, dip, depth, patches)
noise = get_noise(kite_scene[0])
kite_scene[0].displacement = kite_scene[0].displacement + noise
kite_scene[0].save('asc_two_sources_syn_ss_%s_%s'% (dip,depth))
