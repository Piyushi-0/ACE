import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, copy, sys, os, csv
import numpy as np
import scipy.optimize, joblib
from matplotlib.colors import LogNorm

def swap(line, match):
        l = [0]*17
        for i in range(len(line)):
                l[match[i]] = line[i]
        return l

def get_train_test_data(foldername, filename, max_size,num_of_features=44):
	series = []
	sizes = []	
	try:
	        f = open("./" + foldername + "/" + filename)
	except:
	        pass
	reader = csv.reader(f)
	lines = []
	for line in reader:
		line = swap(line, match) #swap if headers are not same as original headers
	        lines.append(line)

	temp_arr = np.array(lines[1:min(max_size + 1,len(lines))]).astype(float)

	series.append(temp_arr) 
	sizes.append(temp_arr.shape[0])
                

	feature_maxs = np.load("feature_maxs.npy")
        feature_mins = np.load("feature_mins.npy")

	for ind in range(len(series)):
		#min-max normalization of series
		series[ind] = (series[ind] - feature_mins)/np.where((feature_maxs - feature_mins) != 0.0, (feature_maxs - feature_mins), 1.0)
		#series[ind] = np.lib.pad(series[ind], ((0,max(0,max_size - series[ind].shape[0])),(0,0)), 'constant')
	series = np.array(series)
	#print series	
	#print series.shape
	return series
    
headers_file = open("headers.csv","r").read()
headers = headers_file.split(",")

num_features = 17
num_hidden = 15
output_size = num_features
match = [11, 12, 3, 4, 1, 7, 2, 0, 6, 14, 5, 13, 15, 16, 8, 9, 10]

#sys.argv[1] = predict
effect = sys.argv[2]
foldername = sys.argv[3]

for filename in os.listdir("./" + foldername + "/"):
	test_set = np.array(get_train_test_data(foldername, filename, max_size=3000, num_of_features=num_features))	

	done_attr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
	average_causal_effects = []

	start_time = int(sys.argv[4]) 
	for attr in done_attr:
	    predictive_means = joblib.load(effect+"/predictive_means_"+str(attr))
	    integrated_value = joblib.load(effect+"/integrated_value_"+str(attr))
	    #run causal_analysis_on_one_sample
	    instance = test_set[0][:,attr]
	    counter = 0
	    average_causal_effect = []
	    #print instance.shape
	    for time in range(start_time,start_time-20,-1):
        	order = predictive_means[counter].shape[0]
	        x = instance[time]

        	expected_y_do_x = 0
	        for o in range(order):
        	    expected_y_do_x += float(predictive_means[counter][o])*(x**o)

	        average_causal_effect.append(expected_y_do_x - integrated_value[counter])
        
        	counter += 1
	    average_causal_effects.append(average_causal_effect)
	average_causal_effects = np.array(average_causal_effects)

	fig, ax = plt.subplots()
	im = ax.imshow(np.absolute(average_causal_effects), cmap = 'coolwarm', interpolation = 'nearest', norm = LogNorm())
	ax.set_xticks(np.arange(20))
	ax.set_yticks(np.arange(len(headers)))
	ax.set_yticklabels(headers)
	ax.set_xticklabels([i for i in range(1, 21)])
	plt.colorbar(im, ax = ax)
	plt.show()
	plt.savefig(effect+"_causal_analysis_"+filename+"_"+str(sys.argv[4])+".png", dpi=300)

    
    
