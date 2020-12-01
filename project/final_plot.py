# PLOTS
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import datetime
from pandas import Grouper
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot



# dict for json files
jsons = dict()

path = 'search/json'
# list files
files = os.listdir(path)

# open json files
for fi in files:
    print(fi)
    with open(os.path.join(path, fi), 'r') as f:
        jsons[fi] = json.load(f)

print(jsons)

days = [ ' mon ' , ' tue ' , ' wed ' , ' thr ' , ' fri ' , ' sat ', ' sun ' ]
for model in jsons.items():
    name = model[0].split(".")[0]
    data = model[1]

    print(name)
    avg = sum(data)/len(data)
    minimum = min(data)
    maximum = max(data)
    print("avg: %.3f" % avg)
    print("min: %.3f" % minimum)
    print("max: %.3f" % maximum)
    print("-"*50)

    plt.plot(days, data, label=name)

plt.title('Final Comparison of Models')
plt.ylabel('RMSE')
plt.ylim([2, 4])
plt.legend(loc="lower right")
plt.show()

#plt.savefig('final.png')
