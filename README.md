# ACE
Code for the paper, <b>Neural Network Attributions: A Causal Perspective.</b>

To be presented at <b>ICML 2019</b>,<br>
Authors: Aditya Chattopadhyay, Piyushi Manupriya, Anirban Sarkar and Vineeth N Balasubramanian

<pre>
<b>On NASA dataset</b>:

Dependencies:
scikit-learn (0.19.1)
scipy (0.17.0)
torch (0.4.0)
joblib (0.11)
matplotlib (1.5.1)
numpy (1.14.5)

Usage:
python lstm.py
python find_tau.py f
python aircraft_causal_interventions.py f
python learn_causal_regressors.py learn eff_n eff
python causal_analysis_final.py predict eff_n fp st

Arguments:
  f - foldername where flight sequences are stored.
  st - start time
  fn - file path
  eff_n - index of attribute whose ACE to find
  eff - attribute whose ACE to find
  
<b>On MNIST dataset</b>:

MNIST.ipynb
Code for Class-conditional Beta-VAE that we use in our MNIST experiment is a modified 
version of Beta-VAE code from https://github.com/1Konny/Beta-VAE .
After training the model, please put checkpoints in the directory of MNIST.ipynb to 
visualize causal effects.
The 'random_z.gif' used in the notebook is the gif you get in outputs folder after training.
</pre><br>  
<b>Acknowledgements</b><br>
https://github.com/1Konny/Beta-VAE

