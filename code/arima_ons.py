from math import sqrt
import numpy as np

def arima_ons(data, options):
    mk, lrate, w, epsilon = options['mk'], options['lrate'], options['init_w'], options['epsilon']
    RMSE = []
    SE = 0
    A_trans = np.identity(mk)*epsilon
    for i in range(mk, data.shape[0]):
        diff = w*data[i-mk:i-1].T - data[i]
        grad = 2*data[i-mk:i-1]*diff
        A_trans = A_trans - A_trans * grad.T * grad * A_trans/(1 + grad * A_trans * grad.T)
        w = w - lrate * grad * A_trans
        SE = SE + diff**2
        if i % options['t_tick'] == 0: 
	        RMSE.append(sqrt(SE/i))
        
    return RMSE, w
        
