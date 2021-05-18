import logging
import os
import re
import requests
import urllib.request
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as num

op = os.path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('kite.clients')


def _download_file(url, outfile):
    logger.debug('Downloading %s to %s', url, outfile)
    r = requests.get(url)

    with open(outfile, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return outfile


def search_tracks(tmin, tmax, lat, lon, destination, track_number):
    fp = urllib.request.urlopen("http://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/%s" % (track_number))
    page = fp.readlines()
    try:
        for line in page[11:-4]:
            line = line.decode("utf8")
            idx = line.find("href=")
            sub = line[idx+6:idx+23]
            fp = urllib.request.urlopen("http://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/%s/%s/metadata/%s-poly.txt" % (track_number, sub, sub))
            page = fp.readlines()
            coords = []
            for line in page:
                line = line.decode("utf8")
                coord = line.split()
                coords.append(num.array([float(coord[0]), float(coord[1])]))
            point = Point(float(lat), float(lon))
            polygon = Polygon(coords)
            if polygon.contains(point) is True:
                fp = urllib.request.urlopen("http://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/%s/%s/interferograms" % (track_number, sub))
                page = fp.readlines()
                nr_lines = len(page)
                for line in page:
                    line = line.decode("utf8")
                    idx_sub = line.find('href="2')
                    if idx_sub is not -1:
                        date1 = int(line[idx_sub+6:idx_sub+14])
                        date2 = int(line[idx_sub+15:idx_sub+23])
                        if date1 >= int(tmin) and date2 <= int(tmax):
                            unw_url = "http://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/%s/%s/interferograms/%s" % (track_number, sub, line[idx_sub+6:idx_sub+23])
                            download_licsar(unw_url+".unw.tif", destination=destination)
    except:
        pass


def download_licsar_stack(tmin, tmax, lat, lon, destination="."):
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    tracks = range(1, 175)
    results = [pool.apply(search_tracks, args=(tmin, tmax, lat, lon, destination, track_number)) for track_number in tracks]
    pool.close()


def download_licsar(unw_url, destination='.'):
    if not unw_url.endswith('.unw.tif'):
        raise ValueError('%s does not end with .unw.tif!' % unw_url)
    print(unw_url)
    scene_name = op.basename(unw_url)
    url_dir = op.dirname(unw_url)
    product_name = op.basename(unw_url)

    os.makedirs(destination, exist_ok=True)

    logger.info('Downloading surface displacement data from LiCSAR: %s',
                product_name)
    unw_file = op.join(destination, scene_name)
    _download_file(unw_url, unw_file)

    logger.info('Downloading LOS angles...')
    scene_id = url_dir.split('/')[-3]
    meta_url = op.normpath(op.join(url_dir, '../../metadata'))

    for unit in ('E', 'N', 'U'):
        fn = '%s.geo.%s.tif' % (scene_id, unit)
        los_url = op.join(meta_url, fn)
        los_url = re.sub(r'^(http:/)\b', r'\1/', los_url, 0)
        outfn = op.normpath(op.join(destination, fn))
        _download_file(los_url, outfn)

    logger.info('Download complete! Open with\n\n\tspool --load=%s',
                unw_file)


def main():
    import sys
    if len(sys.argv)>3:
        download_licsar_stack(*sys.argv[1:])
    else:
        download_licsar(*sys.argv[1:])
