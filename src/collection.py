from itertools import chain,repeat
import gnssmapper as gm
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely.ops
import shapely.geometry


def create_obs(map_,bounds,start, end,samples,*args):
    gm.geo.pyproj.network.set_network_enabled()
    """ create a df of observations denoted by their x-y locations on ground and w-z locations on building """    
    receiverpoints = gm.sim.point_process(map_,bounds,start, end,samples,*args)
    obs = gm.observe(receiverpoints, ['G', 'R', 'C', 'E'])
    obs = gm.geo.to_crs(obs, map_.crs)
    buildings = chain(*repeat(map_.geometry, len(obs)))
    points = list(gm.geo.intersection_projected(obs.geometry, buildings))
    pointsarray = np.array([(np.nan,np.nan,np.nan) if p.is_empty else p.coords[0] for p in points])
    a,b,z = pointsarray[:,0],pointsarray[:,1],pointsarray[:,2]
    line = map_.geometry[0].exterior
    w = np.array([np.nan if p.is_empty else line.project(p) for p in points])
    xy = np.array([l.coords[0][0:2] for l in obs.geometry])
    x,y= xy[:,0],xy[:,1]
    obs_points=shapely.geometry.asMultiPoint(xy)
    az = np.where(b>y,0,180)+np.rad2deg(np.arctan((a-x)/(b-y)))
    az=np.mod(az,360)
    d_ray = ((a-x)**2 +(b-y)**2)**0.5
    d_building = np.array([p.distance(map_.geometry[0]) for p in obs_points])
    return pd.DataFrame({'x':x,'y':y,'w':w,'z':z,'a':a,'b':b,'d_ray':d_ray,'d_building':d_building,'az':az,'el':gm.observations.elevation(obs.geometry),'time':obs.time})
    
def create_image(x,y,i,resolution,bounds,mean=True):
    minx,miny,maxx,maxy=bounds
    mask = ~np.isnan(x) & ~np.isnan(y)
    i_=i[mask]
    x_=x[mask]
    y_=y[mask]
    points=pd.DataFrame()
    points['x'] = np.floor(np.maximum((np.minimum(x_,maxx) - minx),0) * resolution).astype('int64')
    points['y'] = np.floor(np.maximum((np.minimum(y_,maxy) - miny),0) * resolution).astype('int64')
    points['i'] = i_
    if mean:
        pixels=points.groupby(['x','y']).i.mean().reset_index()
    else:
        pixels=points.groupby(['x','y']).i.sum().reset_index()
    nrows=np.max(points.x)+1
    ncols = np.max(points.y)+1
    img = np.zeros((ncols,nrows))
    img[:]=np.nan
    img[ncols-1-pixels.y,pixels.x]=pixels.i
    return img

if __name__=="__main__":
    import sys
    samples=int(sys.argv[1])
    writepath=sys.argv[2]
    map_= gpd.GeoDataFrame({'height':[10],'geometry':[shapely.wkt.loads("POLYGON((528005 183005, 528005 182995,527995 182995, 527995 183005,528005 183005))")]},crs=7405)
    box = shapely.geometry.box(527935, 182935, 528065,183065)
    start = pd.Timestamp('2021-01-01 00:00:00')
    end = pd.Timestamp('2021-01-01 23:59:59')
    obs= create_obs(map_,box.bounds,start,end,samples)
    obs.to_csv(writepath)