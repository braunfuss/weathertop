## prob part
# todo conserve motion/minimum work effort probability (longer segments)
#you will need pandas (until somebody explains xarray to me)! and possibly numba and shaply
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
#import numba
#from numba import jit, float64
import time
import shapely
from shapely.geometry import LineString
#from multiprocess import Process, Manager, Pool
from scipy import sparse
import copy
from ast import literal_eval
import pyrocko
from multiprocess import Process, Manager, Pool
manager = Manager() # Multiprocess manager

shared_d = manager.dict()  # shared-memory dictionary
# definitions of slip types
dip_d = {'Normal-Sinistral' : 70.,
         'Sinistral' : 90.,
         'Dextral Normal' : 70.,
         'Normal' : 50.,
         'Thrust' : 25.,
         'Dextral' : 90.,
         # None, not sure how to handle this
         'Sinistral-Normal' : 70.}

rake_d = {'Normal-Sinistral' : -45.,
         'Sinistral' : 0.,
         'Dextral Normal' : -135.,
         'Normal' : -90.,
         'Thrust' : 90.,
         'Dextral' : 180.,
         # None, not sure how to handle this
         'Sinistral-Normal' : -45.}

def rup_prop(db, load=False):
    if load is True:
        fname = 'muji_fault_lines'
        with open(fname, 'r') as f:
            gj = json.load(f)
    gj = db
    faults = gj['features']
    coords = [feat['geometry']['coordinates'] for feat in faults]
    found = []
    for i in faults: #add condition for finding
        found.append(i)
    # Compile pairwise distance function
    coords_re = []
    for k in coords:
            coords_re.append(k)
    coords = coords_re
    _ = pairwise_dist(np.radians(coords[0]))

    res_faults = copy.deepcopy(gj)
    # write results to geojson to inspect
    with open('faults', 'w') as f:
        json.dump(res_faults, f)

    node_df = pd.concat((fault_to_df(fault) for fault in res_faults['features']),
                        ignore_index=True)
    # Matrix of distances between each node
    node_dists = pd.DataFrame(index=node_df.index, columns=node_df.index,
                              data=pairwise_dist(np.radians(node_df[['lon', 'lat']].values)))
    print(node_df)
    fids = [f['properties']['ogc_fid'] for f in res_faults['features']]

    seg_df = pd.concat((make_segs(fid, node_df) for fid in fids), ignore_index=True)
    print(seg_df)
    seg_df['strike'] = seg_df.apply(calc_strike, node_df=node_df, axis=1)
    seg_df['dip'] = seg_df.apply(get_dip, gj=gj, axis=1)
    seg_df['rake'] = seg_df.apply(get_rake, gj=gj, axis=1)
    rupture_df = pd.concat((make_rupts(fid, seg_df) for fid in fids), ignore_index=True)
    print(rupture_df)
    rupture_df['strike_probs'] = rupture_df.apply(rupt_strike_probs, seg_df=seg_df, axis=1)
    rupture_df.strike_probs.describe()
    rupture_df['mean_strike'] = rupture_df.apply(mean_strike,seg_df=seg_df, axis=1)
    rupture_df['rake'] = rupture_df.apply(get_seg_rake,seg_df=seg_df, axis=1)
    rupture_df['dip'] = rupture_df.apply(get_seg_dip, seg_df=seg_df, axis=1)
    rupture_df['rup_length'] = rupture_df.apply(rupture_length,seg_df=seg_df, axis=1)

    t0 = time.time()

    p = Pool(7)

    ijs = ((i, j, rupture_df, node_dists) for i in rupture_df.index[:]  # Calculate upper-diagonal distance matrix
                  for j in rupture_df.index[i:])# indices
    foo = Foo()
    z = sum(p.imap_unordered(foo.calc_rup_dists,ijs, chunksize=100)) #do calcs; sum=kept ruptures

    #z = sum(p.apply_async(calc_rup_dists,args=(gx))
    t1 = time.time()
    print('{0} changes, done in {1} s'.format(z, int(t1-t0)))
    mult_rups = shared_d.items() # make list of results
    #issue with <=3!
    multifault_rupture_df = pd.concat((make_multifault_df(item) for item in mult_rups),
                                      axis=1,
                                      ignore_index=True).transpose()
    multifault_rupture_df[['r1', 'r2']] = multifault_rupture_df[['r1', 'r2']].astype(int)
    multifault_rupture_df['fid1'] = rupture_df.ix[multifault_rupture_df['r1'], 'fid'].values
    multifault_rupture_df['fid2'] = rupture_df.ix[multifault_rupture_df['r2'], 'fid'].values
    multifault_rupture_df.dist_km.describe()
    multifault_rupture_df['dist_prob'] = np.exp(-multifault_rupture_df.dist_km/5)
    multifault_rupture_df.dist_prob.describe()
    multifault_rupture_df['strike_probs'] = multifault_rupture_df.apply(strike_compatibility, rupture_df=rupture_df,
                                                                        axis=1)
    multifault_rupture_df['dip_probs'] = multifault_rupture_df.apply(dip_compatibility, rupture_df=rupture_df,
                                                                     axis=1)
    multifault_rupture_df['rake_probs'] = multifault_rupture_df.apply(rake_compatibility, rupture_df=rupture_df,
                                                                      axis=1)


    multifault_rupture_df['final_probs'] = ( multifault_rupture_df.dist_prob
                                           * multifault_rupture_df.strike_probs)
                                         #  * multifault_rupture_df.dip_probs
                                          # * multifault_rupture_df.rake_probs)
    multifault_rupture_df.final_probs.describe()
    n_01 = sum((multifault_rupture_df.final_probs > 0.01))
    n_poss = int((len(rupture_df)**2 - len(rupture_df))/2)

    seg_df = pd.concat((make_segs(fid, node_df) for fid in fids), ignore_index=True)

    print('{0} possible ruptures out of {1} ({2:1f}%)'.format(n_01, n_poss, (100 * n_01 /n_poss)))
    return faults, multifault_rupture_df, seg_df, rupture_df, node_df



'''
Note that the angle differences may need to be checked for
bugs, i.e. differences in strike angles should not be
greater than 90 (strike is not a vector), while
a difference of 180 in rake is more meaningful.

'''

def strike_compatibility(multirupt, rupture_df=None):

    strike1 = rupture_df.ix[multirupt.r1, 'mean_strike']
    strike2 = rupture_df.ix[multirupt.r2, 'mean_strike']

    return np.abs(np.cos(np.radians(strike2-strike1)))


def rake_compatibility(multirupt, rupture_df=None):

    rake1 = rupture_df.ix[multirupt.r1, 'rake']
    rake2 = rupture_df.ix[multirupt.r2, 'rake']

    return np.abs(np.cos(np.radians(rake2-rake1)))


def dip_compatibility(multirupt, rupture_df=None):

    dip1 = rupture_df.ix[multirupt.r1, 'dip']
    dip2 = rupture_df.ix[multirupt.r2, 'dip']

    return np.abs(np.cos(np.radians(dip2-dip1)))


def make_multifault_df(item):
    key, val = item
    r1, r2 = key

    return pd.Series((r1, r2, val), index=['r1', 'r2', 'dist_km'])
# Calculate minimum distance between nodes in each rupture
# distance set to 30km!

class Foo():
    @staticmethod
    def calc_rup_dists(ij, d=shared_d, max_dist=30.):
        i, j, rupture_df, node_dists = ij

        if i == j:
            return False

        if i != j:
            #rup_dists[i,j] = 0.
            #return
            #return False

            rup0 = rupture_df.ix[i]
            rup1 = rupture_df.ix[j]

            md = min_rup_dist(rup0, rup1, node_dists)
            if md <= max_dist:
                if md > 0.:
                    d[(i, j)] = md
                    return True
                else:
                    return False
            else:
                return False

def min_rup_dist(rup0, rup1, node_dists):
    #if rup0.fid == rup1.fid:
    #    return -1.
    segs0 = rup0.segment_ids
    segs1 = rup1.segment_ids

    if np.isscalar(segs0):
        if np.isscalar(segs1):
            return node_dists.ix[segs0, segs1]
        else:
            return min(node_dists.ix[segs0, j] for j in segs1)

    elif np.isscalar(segs1):
        return min(node_dists.ix[i, segs1] for i in segs0)

    else:
        return min(node_dists.ix[i, j] for i in segs0 for j in segs1)

def mean_strike(rupture, seg_df=None):
    strikes = seg_df.ix[rupture.segment_ids, 'strike']
    if np.isscalar(strikes):
        return strikes
    else:
        return np.mean(strikes)

def rupture_length(rupture, seg_df=None):
    seg_lengths = seg_df.ix[rupture.segment_ids, 'seg_length']

    return np.sum(seg_lengths)

def get_seg_dip(rupture, seg_df=None):
    dips = seg_df[seg_df.fid == rupture.fid]['dip']
    if np.isscalar(dips):
        return dips
    else:
        return dips.values[0]

def get_seg_rake(rupture, seg_df=None):
    rakes = seg_df[seg_df.fid == rupture.fid]['rake']
    if np.isscalar(rakes):
        return rakes
    else:
        return rakes.values[0]

def make_segs(fid, node_df):
    df = node_df

    fdf = df[df.fid==fid]
    if len(fdf) <1:
        fdf = df

    inds = list(fdf.index)
    starts = inds[:-1]
    stops = inds[1:]
    pairs = list(zip(starts, stops))

    lens = [gc_dist(node_df.iloc[s0][['lon', 'lat']].values,
                    node_df.iloc[s1][['lon', 'lat']].values)
            for (s0, s1) in pairs]

    seg_df = pd.DataFrame(columns=['start', 'stop'],
                          data=pairs)
    seg_df['fid'] = fid
    seg_df['seg_length'] = lens
    return seg_df

def subsegs(seg_list, n):
    '''
    Returns all contiguous segments of length n
    from segment list
    '''
    N = len(seg_list)

    if n == N-1:
        return [((s)) for s in seg_list]
    else:
        return [seg_list[i : N-n+i] for i in range(n+1)]


def segs(seg_list):
    '''
    Returns all possible contiguous segments
    from segment list.
    '''

    segz = [seg_list]
    n = len(seg_list)

    subsegz = [subsegs(seg_list, i) for i in range(n)]

    return [s for sub in subsegz for s in sub]

# Make dataframe of all possible continuous floating ruptures on the faults
#Floating ruptures are defined as any possible sequence of contiguous segments
#on the fault. For a fault with $n$ segments, there are $\frac{n(n+1)}{2}$ possible
#floating ruptures. For a fault with 4 segments, this is $\frac{4\cdot5}{2}$ or 10 ruptures.

def make_rupts(fid, df):
    fdf = df[df.fid == fid]

    inds = tuple(fdf.index)

    rupts = segs(inds)
    nrups = len(rupts)

    rupt_df = pd.DataFrame(pd.Series(rupts), columns=['segment_ids'])
    rupt_df['fid'] = fid

    return rupt_df

def calc_strike(seg, node_df=None):

    p0 = node_df.ix[seg.start, ['lon', 'lat']]
    p1 = node_df.ix[ seg.stop, ['lon', 'lat']]

    lon0, lat0 = np.radians(p0[['lon', 'lat']])
    lon1, lat1 = np.radians(p1[['lon', 'lat']])

    y = np.sin(lon1-lon0) * np.cos(lat1)
    x = np.cos(lat0) * np.sin(lat1) - np.sin(lat0) * np.cos(lat1) * np.cos(lon1-lon0)

    angle = np.degrees(np.arctan2(y,x))

    az = 90 - angle #-(angle - 90)

    while az < 0:
        az += 360
    while az > 360:
        az -= 360

    return az

def get_dip(f, gj=None):

    props = gj['features'][int(f.fid)]['properties']

    dip = props['ns_average_dip']
    if dip is not None:
        if type(dip) == str:
            try:
                dip = literal_eval(dip)
            except SyntaxError: # frequently '(90,,)' etc.
                dip = dip.strip('()')
                dip = literal_eval(dip.split(',')[0])

        if np.isscalar(dip):
            dip = float(dip)
        else:
            dip = float(dip[0])

    else:
        try:
            dip = dip_d[props['slip_type']]
        except KeyError: # in case slip_type == None:
            dip = 60. # whatever

    return dip

def get_rake(f, gj=None):

    props = gj['features'][int(f.fid)]['properties']

    rake = props['ns_average_rake']
    if rake is not None:
        if type(rake) == str:
            try:
                rake = literal_eval(rake)
            except SyntaxError: # frequently '(90,,)' etc.
                rake = rake.strip('()')
                rake = literal_eval(rake.split(',')[0])

        if np.isscalar(rake):
            rake = float(rake)
        else:
            rake = float(rake[0])

    else:
        try:
            rake = rake_d[props['slip_type']]
        except KeyError: # in case slip_type == None:
            rake = 60. # whatever

    return rake


def rupt_strike_probs(rupture, seg_df=None):
    strikes = seg_df.ix[rupture.segment_ids, 'strike']
    if np.isscalar(strikes):
        probs = 1.
    else:
        strike_rads = np.radians(strikes)
        probs = np.product(np.cos(np.diff(strike_rads)))
    return probs


def gc_dist(p0, p1, R=6371):
    '''
    Returns the (approximate) great-circle distance
    between two points (in long-lat). Returned
    distance in kilometers.
     '''
    p0 = np.radians(p0)
    p1 = np.radians(p1)

    x = (p0[0]-p1[0]) * np.cos((p0[1]+p1[1]) /2)
    y = p0[1] - p1[1]

    return R * np.sqrt((x**2 + y**2))


# this is a just-in-time compiled version using Numba

R = 6371  # earth radius
#@jit(float64[:](float64[:]))
def pairwise_dist_old(X):
    '''
    Pairwise distance function (using same calculations as gc_dist)
    but accelerated for making a pairwise distance matrix for all
    points in list
    '''
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = ((X[i,0] - X[j,0]) * np.cos((X[i,1] - X[j,1])/2))**2
            d += (X[i,1]-X[j,1])**2
            D[i,j] = np.sqrt(d) * R

    return D

def pairwise_dist(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                d = ((X[i,0] - X[j,0]) * np.cos((X[i,1] - X[j,1])/2))**2
                d += (X[i,1]-X[j,1])**2
                D[i,j] = np.sqrt(d) * R
    return D

def fault_to_df(fault):
    df = pd.DataFrame(data=np.array(fault['geometry']['coordinates']),
                      columns=['lon', 'lat'])
    df['fid'] = 1 #fault ID
    return df
