from itertools import chain, repeat
import gnssmapper as gm
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely


def _indicator(x, height, distance=np.inf):
    if distance == np.inf:
        return ~np.isnan(x)
    return ((height-distance < x) & (x < height+distance))


def int_metric(x, height, distance=np.inf):
    """ Intersection metric for intersection points. Number of points within a given distance of height.
    """
    distances = np.abs(x[~np.isnan(x)]-height)

    if distance == np.inf:
        return len(distances)

    else:
        return np.sum(distances <= distance) 


def window_metric(x, height, n):
    """Window metric for intersection points. Required distance from height to include n points .
    """
    
    distances = np.abs(x[~np.isnan(x)]-height)

    if len(distances) < n:
        return np.inf
    
    else:
        return np.sort(distances)[n-1]


if __name__=="__main__":
    import sys
    data=pd.read_csv(sys.argv[1])
    true_height=pd.read.csv

    def generate_datasets(filepath,height):
        data=pd.read_csv(filepath)
        data=data[_indicator(data.height,height,np.inf),]
        max_size=data.shape[0]
        windows= [2**i for i in range(6)]
        counts= [10**i for i in range(1,6)]
        int_indicators = [_indicator(data.height,height,d) for d in windows]
        window_indicators = [_indicator(data.height,height,window_metric(data.height,height,c)) for c in counts]
        indicators=int_indicators.extend(window_indicators)
        for w in windows:
            indicator = _indicator(data.height,height,w)
            insample_sizes = [2**i for i in range(5,np.floor(np.log2(np.sum(indicator)))]
            for i in insample_sizes:
                outsample
                outsample_sizes = insample_sizes[2**i for i in range(np.sum(~indicators)
            insample_sizes 


    sample_size=[2**i for i in range]

    for _ in range()