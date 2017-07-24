import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import urllib


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    mean, se = np.mean(a), scipy.stats.sem(a)
    interval = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return mean, interval


FILE_COUNT = 5
TENSORBOARD_URL = "http://localhost:6006/data/scalars?run=C%5C{0}%5C{2}%5Crole_0{1}&tag=val_accuracy&format=csv"

games = ["connect4","chinese_checkers_6"]
for game in games:
    for run_type in ["", "_pretrained"]:
        run_data = {}
        for i in range(1, FILE_COUNT):
            url = TENSORBOARD_URL.format(game,run_type,i)
            print(url)
            filename = game+ "_" + str(i) + run_type + ".csv"
            urllib.urlretrieve(url, filename)

            csv = pd.read_csv(filename, usecols=[1,2])
            for x,y in csv.values:
                if x in run_data.keys():
                    run_data[x].append(y)
                else:
                    run_data[x] = [y]
        interval_data = []
        for key, val_arr in run_data.items():
            mean, interval = mean_confidence_interval(val_arr)
            interval_data.append([key, mean, interval])
        filename = game + run_type + "_interval_data.csv"
        interval_data = np.array(interval_data)
        np.savetxt(filename, interval_data, delimiter=",")
