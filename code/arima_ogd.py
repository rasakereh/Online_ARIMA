from math import sqrt

def arima_ogd(data, options):
	mk, lrate, w = options['mk'], options['lrate'], options['init_w']
	RMSE = []
	SE = 0
	for i in range(mk, data.shape[0]):
		diff = w*data[i-mk:i-1].T - data[i]
		w = w - data[i-mk:i-1]*2*diff/sqrt(i-mk+1)*lrate
		SE = SE + diff**2
		if i % options['t_tick'] == 0: 
			RMSE.append(sqrt(SE/i))
	
	return RMSE, w

