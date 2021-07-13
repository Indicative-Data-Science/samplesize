import pandas as pd
import numpy as np
import gnssmapper as gm


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


def generate_samples(data, height):
    int_d = [2**i for i in range(5)]
    samples = []
    for d, out_d in zip(int_d, int_d[1:]+[np.inf]):
        in_obs = _indicator(data.height, height, d)
        out_obs = _indicator(data.height, height, out_d) & ~in_obs
        in_idx = np.flatnonzero(in_obs)
        out_idx = np.flatnonzero(out_obs)
        max_size = np.sum(in_obs)+np.sum(out_obs)
        sample_size = [
            2**i for i in range(1, np.ceil(np.log2(max_size), 0.25).astype('int'))]
        insample_proportion = np.arange(0.1, 1.1, 0.1)
        reps = range(30)
        insample_size = (np.floor(p*s).astype('int')
                         for s in sample_size for p in insample_proportion)
        outsample_size = (np.ceil((1-p)*s).astype('int')
                          for s in sample_size for p in insample_proportion)
        for i, o in zip(insample_size, outsample_size):
            print(f'indicator {d},insample {i}, outsample {o}')
            for _ in reps:
                in_idx_sample = np.random.choice(
                    in_idx, size=min(i, in_idx.shape[0]), replace=False)
                out_idx_sample = np.random.choice(
                    out_idx, size=min(o, out_idx.shape[0]), replace=False)
                idx_sample = np.concatenate((in_idx_sample, out_idx_sample))
                sample = np.zeros_like(in_obs)
                sample[idx_sample] = True
                samples.append(sample)
    out = np.stack(samples, axis=-1)
    print('returning output')
    return out


def run_sample(data, sample_array, height):
    int_d = [2**i for i in range(5)]+[np.inf]
    window_n = [10**i for i in range(4)]
    colnames = ["I_"+str(d) for d in int_d]+["W_"+str(n)
                                             for n in window_n]+['est_lb', 'est', 'est_ub']
    results = []
    for i, sample in enumerate(sample_array.T):
        if i % 100 == 0:
            print(i/sample_array.shape[1])
        h = data.height[sample].values
        ss = data.SS[sample].values
        metrics = [int_metric(h, height, d) for d in int_d] + \
            [window_metric(h, height, n) for n in window_n]
        est = gm.algo.fit_edge(h, ss, np.array([40, 0.1, 30, 10]))
        results.append(np.array(metrics+list(est)))
    out = np.stack(results, axis=0)
    out = pd.DataFrame(out, columns=colnames)
    return(out)


if __name__ == "__main__":
    import sys
    data = pd.read_csv(sys.argv[1])
    height = int(sys.argv[2])
    writepath = sys.argv[3]
    samples = generate_samples(data, height)
    np.save(writepath, samples)
    results = run_sample(data, samples, height)
    results.to_csv(writepath+".csv")
