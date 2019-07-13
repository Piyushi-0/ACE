# ACE
Code for the paper, <b>Neural Network Attributions: A Causal Perspective.</b>

Presented at <b>ICML 2019</b>,<br>
Authors: Aditya Chattopadhyay, Piyushi Manupriya, Anirban Sarkar and Vineeth N Balasubramanian

<pre>
ACE is calculated using Taylor's Series for NASA, Synthetic Dataset, Iris codes. 
Monte Carlo Sampling is used for MNIST.

Dependencies:
scikit-learn (0.19.1)
scipy (1.1.0)
torch (0.4.1)
joblib (0.12.5)
matplotlib (2.2.2)
numpy (1.15.2)

<b>On NASA dataset</b>:
<a href="https://c3.nasa.gov/dashlink/projects/85/">NASA Dataset</a>

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
 
 'Headers.csv' is a file with only the names(headers) of attributes.

<b>On MNIST dataset</b>:

Usage:
Please run in the following order-
sh run_mnist_mod.sh
MNIST.ipynb

Code for Class-conditional Beta-VAE that we use in our MNIST experiment is a modified 
version of Beta-VAE code from <a href="https://github.com/1Konny/Beta-VAE">https://github.com/1Konny/Beta-VAE</a>(acknowledged).

The 'random_z.gif' used in the notebook is the gif you get in outputs folder after training.

<b>On Synthetic Dataset</b>:

Usage:
Please run in the following order-
toy_dataset.ipynb
python evaluate_lstm.py

<b>On IRIS Dataset</b>:

Usage:
Please run in the following order-
python train.py
ACE.ipynb

</pre><br>  
<b>Acknowledgements</b><br>
https://github.com/1Konny/Beta-VAE

