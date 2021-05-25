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
logger = logging.getLogger('weathertop.clients')


def _download_file(url, outfile):
    logger.debug('Downloading %s to %s', url, outfile)
    r = requests.get(url)
    with open(outfile, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return outfile


def search_tracks_date(tmin, tmax, lat, lon, destination, track_number,
                       pair_only, date):

    ascending = False
    descending = False
    keep_url = None
    date = float(date)
    pair_only = str(pair_only)
    fp = urllib.request.urlopen("http://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/%s" % (track_number))
    page = fp.readlines()
    try:
        for line in page[11:-4]:
            date_min = 99999999999999999.
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
                            if date is not None:
                                date_diff = (date2-date)+(date1-date)
                                if date_diff < date_min and date < date2 and date > date1:
                                    keep_url = unw_url
                                    date_min = date_diff
                                    track_name = str(sub)[0:4]

    except:
        pass

    if keep_url is not None:
        unw_url = keep_url
        download_licsar(unw_url+".unw.tif",
                        destination=destination,
                        track_name=track_name)


def search_tracks(tmin, tmax, lat, lon, destination, track_number,
                  pair_only):
    ascending = False
    descending = False
    pair_only = str(pair_only)
    fp = urllib.request.urlopen("http://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/%s" % (track_number))
    page = fp.readlines()
    try:
        for line in page[11:-4]:
            date_min = 99999999999999999.
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
                            if date is not None:
                                date_diff = date2-date1
                                if date_diff < date_min:
                                    keep_url = unw_url

                            track_name = str(sub)[0:4]
                            if pair_only == "True":
                                if track_name[3] == "A" and ascending == False:
                                    download_licsar(unw_url+".unw.tif",
                                                    destination=destination,
                                                    track_name=track_name)
                                    ascending = True

                                elif track_name[3] == "D" and descending == False:
                                    download_licsar(unw_url+".unw.tif",
                                                    destination=destination,
                                                    track_name=track_name)
                                    descending = True

                            else:
                                download_licsar(unw_url+".unw.tif",
                                                destination=destination,
                                                track_name=track_name)
                                if track_name[3] == "A":
                                    ascending = True
                                if track_name[3] == "D":
                                    descending = True
                            if pair_only == "True":
                                if ascending is True and descending is True:
                                    break
    except:
        pass


def download_licsar_stack(tmin, tmax, lat, lon, destination=".",
                          pair_only=False, date=None):
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    tracks = range(1, 175)
    if date is not None:
        results = [pool.apply(search_tracks_date, args=(tmin, tmax, lat, lon,
                                                        destination, track_number,
                                                        str(pair_only), date)) for track_number in tracks]
    else:
        results = [pool.apply(search_tracks, args=(tmin, tmax, lat, lon,
                                                   destination, track_number,
                                                   str(pair_only))) for track_number in tracks]
    pool.close()


def download_licsar(unw_url, destination='.', track_name="_"):
    if not unw_url.endswith('.unw.tif'):
        raise ValueError('%s does not end with .unw.tif!' % unw_url)
    scene_name = op.basename(unw_url)
    url_dir = op.dirname(unw_url)
    product_name = op.basename(unw_url)

    os.makedirs(destination, exist_ok=True)

    logger.info('Downloading surface displacement data from LiCSAR: %s',
                product_name)
    unw_file = op.join(destination, track_name+"_"+scene_name)
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

    logger.info('Download complete! You can open with\n\n\tspool --load=%s',
                unw_file)


def main():
    import sys
    if len(sys.argv)>3:
        download_licsar_stack(*sys.argv[1:])
    else:
        download_licsar(*sys.argv[1:])
