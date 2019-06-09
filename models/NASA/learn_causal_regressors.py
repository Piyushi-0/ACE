import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, copy, sys, os, csv
import numpy as np
import scipy.optimize, joblib
from matplotlib.colors import LogNorm


def get_train_test_data(foldername, max_size,num_of_features=44):
	series = []
	sizes = []
	feature_mins = [np.inf]*num_of_features
	feature_maxs = [-1*np.inf]*num_of_features
	for filename in os.listdir("./" + foldername + "/"):
		try:
		        f = open("./" + foldername + "/" + filename)
		except:
		        continue
		reader = csv.reader(f)

		lines = []
		for line in reader:
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
	print series.shape
	return series  

headers_file = open("headers.csv","r").read()
headers = headers_file.split(",")

def func(x, m, n, X, Y, raw_eigvals):
    precision = (np.eye(m)*x[0]) + (x[1]*np.dot(X,X.T))
    eigenvals = x[1]*raw_eigvals 	  
    try:
	temp_inv = np.linalg.inv(precision)
    except:
	return np.inf
    factor = np.dot(np.dot(np.dot(Y.T,X.T),np.linalg.inv(precision)),np.dot(X,Y))
    log_marginal_likelihood = (0.5*m*np.log(x[0]) + 0.5*n*np.log(x[1]) - 0.5*x[1]*np.dot(Y.T,Y) + 0.5*x[1]*x[1]*factor - 0.5*np.log(np.linalg.det(precision)))
    return -log_marginal_likelihood
   

def learn_causal_regressors(attribute, effect, feature):
    #causal strength for predicting the 100th time step LATG of attribute index
    output_index = [100,0, effect]
    print "causal_factor", headers[attribute]
    integrated_feature_val_at_timestep = []
    predictive_means_at_timestep = []
    for t in range(100,80,-1):
        feature_index = [t,0,attribute]
        f = open(feature+'/expectation_do_x_for_cause_'+str(feature_index[0]) + "-"+str(feature_index[2]) + "_and_effect_" + str(output_index[0]) + "-" + str(output_index[2]))

        expectation_do_x = np.array(pickle.load(f))
        x = np.linspace(0.0,1.0,100)
        n = x.shape[0]
        X = np.array([1]*n).reshape(1,-1)
        Y = expectation_do_x.reshape(-1,1)
        evidence = []
        alpha_beta = []

        #model selection
        for polynomial_order in range(40):
            m = polynomial_order + 1
            eigenvals = np.linalg.eigvals(np.dot(X,X.T))
            #print polynomial_order, eigenvals
            res = scipy.optimize.minimize(func, np.array([1e-2,1e-2]), args=(m,n, X, Y, eigenvals), bounds = ((0, None), (0, None)))
            model_evidence = float(res.fun)*-1.0
            marginal_log_likelihood = model_evidence - 0.5*n*np.log(2*np.pi)
            alpha_beta.append(res.x)
            X = np.vstack((X, x**m))
            if alpha_beta[-1][0] <= 1e-12:
                    marginal_log_likelihood = -np.inf
            evidence.append(marginal_log_likelihood)
        chosen_degree = np.argmax(np.array(evidence))
        alpha = alpha_beta[np.argmax(np.array(evidence))][0]
        beta = alpha_beta[np.argmax(np.array(evidence))][1]

        X = np.array([1]*n).reshape(1,-1)	
        for polynomial_order in range(chosen_degree):
            X = np.vstack((X, x**(polynomial_order + 1)))

        predictive_precision = (np.eye(chosen_degree + 1)*alpha) + (beta*np.dot(X,X.T))
        predictive_mean = beta*np.dot(np.linalg.inv(predictive_precision),np.dot(X,Y))

        num_of_sample_points = 10000
        x_n = np.linspace(0.0,1.0,num_of_sample_points)
        X_n = np.array([1]*num_of_sample_points).reshape(1,-1)	
        for polynomial_order in range(chosen_degree):
            X_n = np.vstack((X_n, x_n**(polynomial_order+1)))

        

        predictive_mean_val = np.dot(predictive_mean.T, X_n).reshape(-1,)
        predictive_var = (1.0/beta) + np.diagonal(np.dot(np.dot(X_n.T,np.linalg.inv(predictive_precision)),X_n))
       
        low_CI = predictive_mean_val - np.sqrt(predictive_var)
        upper_CI =  predictive_mean_val + np.sqrt(predictive_var)
        #plt.plot(np.arange(predictive_mean_val.shape[0]), predictive_mean_val, label="predictive mean")
        #plt.fill_between(np.arange(predictive_mean_val.shape[0]), low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '95% CI')
        #plt.show()
        #plt.clf()
        #print np.sqrt(predictive_var)/predictive_mean_val
        #integrating for average causal effect
        integrated_val = 0
        for order in range(chosen_degree+1):
            integrated_val += float(predictive_mean[order])/(order + 1)
        integrated_feature_val_at_timestep.append(integrated_val)
        predictive_means_at_timestep.append(predictive_mean)
    return np.array(predictive_means_at_timestep), np.array(integrated_feature_val_at_timestep)

#test_set = np.array(get_train_test_data(foldername, max_size=3000, num_of_features=num_features))	

done_attr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
average_causal_effects = []

if sys.argv[1] == "learn":
	effect = sys.argv[2]
	feature = sys.argv[3]
	for attr in done_attr:
		predictive_means, integrated_value = learn_causal_regressors(attr, effect, feature)
    joblib.dump(predictive_means, "predictive_means_"+str(attr))
		joblib.dump(integrated_value, "integrated_value_"+str(attr))
