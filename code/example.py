import pandas as pd
import numpy as np
import arima_ogd as AOGD
import arima_ons as AONS

file_path0 = '/path/to/the/file0.csv'
file_path1 = '/path/to/the/file1.csv'

seq_d0 = pd.read_csv(file_path0)
seq_d1 = pd.read_csv(file_path1)

options = {
    'mk': 10,
    'init_w': np.random.uniform(size=10),
    't_tick': 1,
}


options['lrate'] = 1
[RMSE_ogd1,w] = AOGD.arima_ogd(seq_d1,options)

options['lrate'] = 1.75
options['epsilon'] = 10**(-0.5)
[RMSE_ons1,w] = AONS.arima_ons(seq_d1,options)

options['lrate'] = 10**(-3)
[RMSE_ogd0,w] = AOGD.arima_ogd(seq_d0,options)

options['lrate'] = 10**(3)
options['epsilon'] = 10**(-5.5)
[RMSE_ons0,w] = AONS.arima_ons(seq_d0,options)


