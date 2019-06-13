# ACE
Code for the paper, <b>Neural Network Attributions: A Causal Perspective.</b>

Presented at <b>ICML 2019</b>,<br>
Authors: Aditya Chattopadhyay, Piyushi Manupriya, Anirban Sarkar and Vineeth N Balasubramanian

<pre>

Codes tested with:
scikit-learn (0.19.1)
scipy (1.1.0)
torch (0.4.1)
joblib (0.12.5)
matplotlib (2.2.2)
numpy (1.15.2)

<b>On NASA dataset</b>:

Usage:
Please run in the following order-
python lstm.py
python find_tau.py f
python aircraft_causal_interventions.py f
python learn_causal_regressors.py learn eff_n
python causal_analysis_final.py predict eff_n fp st

Arguments:
  f - foldername where flight sequences are stored.
  st - start time
  fp - file path
  eff_n - index of attribute which is the effect
 
 Please put 'output_layer_trained_model', 'lstm_trained_model', 'feature_maxs.npy',
 'feature_mins.npy' obtained from lstm.py in the same directory as other codes. 
 'Headers.csv' is a file with only the names(headers) of attributes.

<b>On MNIST dataset</b>:

Usage:
Please run in the following order-
sh run_mnist_mod.sh
MNIST.ipynb

Code for Class-conditional Beta-VAE that we use in our MNIST experiment is a modified 
version of Beta-VAE code from https://github.com/1Konny/Beta-VAE(acknowledged).
After training the model, please put checkpoints in the directory of MNIST.ipynb to 
visualize causal effects.
The 'random_z.gif' used in the notebook is the gif you get in outputs folder after training.

<b>On Synthetic Dataset</b>:

Usage:
Please run in the following order-
python train_lstm.py
python toy_causal.py
python evaluate_lstm.py

To run toy_causal.py and evaluate_lstm.py, please put 'output_layer_trained_model'
and 'lstm_trained_model' obtained from train_lstm.py in the same directory.

<b>On IRIS Dataset</b>:

Usage:
Please run in the following order-
nn_iris.ipynb
python decision_tree.py
plot_iris.ipynb
causal_iris.ipynb
python learn_cr.py

To run toy_causal.py and evaluate_lstm.py, please put 'output_layer_trained_model'
and 'lstm_trained_model' obtained from train_lstm.py in the same directory.
</pre><br>  
<b>Acknowledgements</b><br>
https://github.com/1Konny/Beta-VAE

