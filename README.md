# ACE
Code for the paper, <b>Neural Network Attributions: A Causal Perspective.</b>

To be presented at <b>ICML 2019</b>,
Authors:<br>
Aditya Chattopadhyay, Piyushi Manupriya, Anirban Sarkar and Vineeth N Balasubramanian

<pre>
<b>On NASA dataset</b>:

Dependencies:
scikit-learn (0.19.1)
scipy (0.17.0)
torch (0.4.0)
joblib (0.11)
matplotlib (1.5.1)
numpy (1.14.5)

Features:
Causal attributions: causal_analysis_final.py

Usage:
python causal_analysis_final.py predict effect foldername start_time
ex: python causal_analysis_final.py predict GS X-Plane_Data_Set/wrong_flap 200
python learn_causal_regressors.py learn effect_num_header effect
ex: python learn_causal_regressors.py learn 5 LATG 

<b>On MNIST dataset</b>:
  MNIST.ipynb
<b>Acknowledgements</b>
https://github.com/1Konny/Beta-VAE
<pre>
