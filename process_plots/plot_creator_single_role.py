import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import urllib
from collections import OrderedDict

def mean_confidence_interval(data, confidence=0.99):
    a = 1.0*np.array(data)
    n = len(a)
    mean, se = np.mean(a), scipy.stats.sem(a)
    interval = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return mean, interval

FILE_COUNT = 123
PLOT_POINTS = 75
TENSORBOARD_URL = "http://localhost:6006/data/scalars?run=C%5C{0}%5C{1}%5Crole_0&tag=val_accuracy&format=csv"

games = ["chinese_checkers_6","connect4","breakthrough"]
for game in games:
    run_data = OrderedDict()
    for i in range(1, FILE_COUNT):
        url = TENSORBOARD_URL.format(game, i)
        filename = "runs/" + game + ".csv"
        urllib.urlretrieve(url, filename)

        csv = pd.read_csv(filename, usecols=[1,2])
        csv_len = len(csv.values)
        print game , csv_len ,  (csv_len % PLOT_POINTS) , csv_len - (csv_len % PLOT_POINTS)
        key_chunk_size = csv_len / PLOT_POINTS
        arr = []
        keys = []
        
        first, val =  csv.values[0]
        if first not in run_data:
            run_data[first] = []
        run_data[first].append(val)
        
        for x,y in csv.values[:csv_len - (csv_len % PLOT_POINTS)]:
            if len(keys) == key_chunk_size:
                key = np.median(keys)
                if key not in run_data:
                    run_data[key] = arr
                else:
                    run_data[key] = run_data[np.median(keys)] + arr
                arr = []
                keys = [x]
            else:
                keys.append(x)
            arr.append(y)
        key = np.median(keys)
        run_data[key] = arr

    interval_data = []
    for key, val_arr in run_data.items():
        std = np.std(1.0*np.array(val_arr))
        mean, interval_95 = mean_confidence_interval(val_arr, 0.95)
        interval_data.append([key, mean, std])
    filename = game + "_long_run.csv"
    interval_data = np.array(interval_data)
    np.savetxt(filename, interval_data, delimiter=",")
