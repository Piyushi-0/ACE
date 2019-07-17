# ACE
<b>Neural Network Attributions: A Causal Perspective</b><br>
Aditya Chattopadhyay, Piyushi Manupriya, Anirban Sarkar, Vineeth N Balasubramanian<br>
Presented at ICML 2019
<pre>
<b>Dependencies</b>:
scikit-learn (0.19.1)
scipy (0.17.0)
torch (0.4.0)
joblib (0.11)
matplotlib (1.5.1)
numpy (1.14.5)

<b>Usage</b>-
MNIST:
  sh run_mnist_mod.sh
  ipython nbconvert --to python MNIST.ipynb
  python MNIST.py
  
Iris:
  python decision_tree.py
  ipython nbconvert --to python train.ipynb
  python train.py
  ipython nbconvert --to python ACE.ipynb
  python ACE.py
  
Synthetic Dataset:
  ipython nbconvert --to python toy_dataset.ipynb
  python toy_dataset.py
  python evaluate_lstm.py
  
Aircraft:
  python lstm.py
  python find_tau.py
  python aircraft_causal_interventions.py foldername
  eg. python aircraft_causal_interventions.py "40"
  python learn_causal_regressors.py learn effect_num_header effect
  eg. python learn_causal_regressors.py learn 5 LATG
  python causal_analysis_final.py predict effect foldername start_time
  eg. python causal_analysis_final.py predict GS "40" 100
</pre>

NASA dataset used in Aircraft code is uploaded at https://drive.google.com/open?id=1rEZ3veRpcKH5OZKAoXuVTyC9oMnn78ra <br>
Class-conditional Beta VAE code used in MNIST experiments is a adapted from Beta VAE code from https://github.com/1Konny/Beta-VAE <br>

<p>
  <b>If you use this code, please cite our paper: <br> </b>
    Aditya Chattopadhyay, Piyushi Manupriya, Anirban Sarkar, Vineeth N Balasubramanian. "Neural Network Attributions: A Causal Perspective", in International Conference on Machine Learning (ICML), 2019.<br>
    <b><a href="https://github.com/Piyushi-0/ACE/blob/master/pmlr-v97-chattopadhyay19a.bib">Bibtex</a><br>
    and consider giving a star to our repository.
  </b>
</p>

<b>References:</b><br>
https://github.com/1Konny/Beta-VAE<br>
https://c3.nasa.gov/dashlink/projects/85/resources/?type=ds

