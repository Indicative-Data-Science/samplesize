import gnssmapper as gm
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygeos.measurement import total_bounds


def create_obs2(mapfile, bounds, start, end, samples):
    map_ = gpd.read_file(mapfile)
    height = pd.read_csv(
        '../maps/tq/tq.csv', names=pd.read_csv('../maps/tcr/docs/BHA_Header.csv').columns)
    map_ = map_.merge(height, left_on='fid', right_on='OS_TOPO_TOID')
    map_['height'] = (map_.RelH2+map_.RelHMax)/2
    map_ = map_[['height', 'geometry']]
    map_ = map_.set_crs(7405, allow_override=True)
    gm.geo.pyproj.network.set_network_enabled()
    if bounds is None:
        bounds=map_.total_bounds
    receiverpoints = gm.sim.point_process(map_, bounds, start, end, samples)
    obs = gm.observe(receiverpoints, ['G', 'R', 'C', 'E'])
    obs = gm.geo.to_crs(obs, map_.crs)
    receiverpoints == gm.geo.to_crs(receiverpoints, map_.crs)
    intersections = gm.algo.projected_height(map_, obs.geometry)
    xy = np.array([l.coords[0][0:2] for l in obs.geometry])
    x, y = xy[:, 0], xy[:, 1]
    intersections['x'] = x
    intersections['y'] = y
    intersections['t'] = obs.time
    return intersections


if __name__ == "__main__":
    import sys
    samples = int(sys.argv[1])
    map = sys.argv[2]
    writepath = sys.argv[3]
    if map == "tcr":
        path = '../maps/tcr/tcr.gpkg'
        bounds = None#np.array([529335.15, 181870.425, 529535.15, 182070.425])
    if map == "bank":
        path = '../maps/bank/bank.gpkg'
        bounds = None#np.array([532600, 180950, 532800, 181150])
    if map == "wharf":
        path = '../maps/wharf/wharf.gpkg'
        bounds = None#np.array([537507.033, 180181.963, 537707.033, 180381.963])

    start = pd.Timestamp('2021-01-01 00:00:00')
    end = pd.Timestamp('2021-01-01 23:59:59')
    obs = create_obs2(path, bounds, start, end, samples)
    obs.to_csv(writepath)