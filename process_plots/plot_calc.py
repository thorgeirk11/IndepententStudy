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


FILE_COUNT = 150
TENSORBOARD_URL = "http://localhost:6006/data/scalars?run=C%5C{0}%5C{2}%5Crole_0{1}&tag=val_accuracy&format=csv"

games = ["connect4","chinese_checkers_6","breakthrough"]
for game in games:
    for run_type in ["", "_pretrained"]:
        run_data = OrderedDict()
        for i in range(1, FILE_COUNT):
            url = TENSORBOARD_URL.format(game, run_type, i)
            print(url)
            filename = "runs/" + game + "_" + str(i) + run_type + ".csv"
            urllib.urlretrieve(url, filename)

            csv = pd.read_csv(filename, usecols=[1,2])
            r = 0
            meanArr = []
            for x,y in csv.values:
                meanArr.append(y)
                if r % 3 == 1:
                    mean = np.mean(meanArr)
                    if x - 30 in run_data.keys():
                        run_data[x - 30].append(mean)
                    else:
                        run_data[x - 30] = [mean]
                    meanArr = []
                r += 1
        interval_data = []
        for key, val_arr in run_data.items():
            std = np.std(val_arr)
            mean, interval_95 = mean_confidence_interval(val_arr, 0.95)
            _, interval_99 = mean_confidence_interval(val_arr, 0.99)
            interval_data.append([key, mean, std, mean + interval_95, mean + interval_99])
        filename = game + run_type + "_interval_data.csv"
        interval_data = np.array(interval_data)
        np.savetxt(filename, interval_data, delimiter=",")
