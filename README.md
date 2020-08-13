The following instructions are for installing a modified iGibson for AMR-Policies and running the third example seen in "Learning to Actively Reduce Memory Requirements for Robot Control Tasks."

### Setup
Installation instructions are a combination from http://svl.stanford.edu/igibson/docs/ and https://github.com/StanfordVL/GibsonSim2RealChallenge. Verified on Ubuntu 18.04.
- Make conda environment
```
  conda create -n igibson-amr python=3.6 anaconda
  conda activate igibson-amr
```
- Install Cuda 10.0
  - Extra install information if needed https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
- Install CuDNN 7.6.5 for Cuda 10.0 (Runtime and Developer library)
  - Installation guide https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
- Install EGL dependency
```
  sudo apt-get install libegl1-mesa-dev
  ```
- Install pytorch
- Install tensorflow 1.15.0
```
  pip install tensorflow-gpu==1.15.0
```
### Gibson Install
- Install iGibson
```
cd AMR-Policies
source activate igibson-amr
pip install -e .
```
- Add assets and datasets to folders "assets" and "dataset" in `AMR-Policies/gibson2`
  - Assets download: http://svl.stanford.edu/igibson/docs/installation.html
  - Need Rs + Placida scene downloaded for dataset folder. Fill out form https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform
  - Placida is found in "Interactive Gibson dataset, 10 scenes, with replaced objects and textures"
  - Follow Gibson installation guide for Rs download
- Verify iGibson install
  - Should see TurtleBot in apt taking random actions (need cv2 version 4.2.0 not 4.3.0 for this to work)
  ```
  cd examples/demo
  python env_example.py
  ```
- Install tf-agents gibson-sim2real branch
  - Navigate to "agents" folder in `AMR-Policies/gibson2`
  - Install
  ```
  cd AMR-Policies/gibson2/agents
  pip install -e .
  ```
- The following files were added to iGibson and tf-agents
  - turtlebot_AMR.yaml to `AMR-Policies/examples/configs`
  - actor_distribution_rnn_m_network.py to `AMR-Policies/gibson2/agents/tf_agents/networks`
    - Note: a modified version of TF-agent's actor_distribution_rnn_network.py [[1]](#1)
  - rnn_encoding_network.py to `AMR-Policies/gibson2/agents/tf_agents/networks`
    - Note: a modified version of TF-agent's lstm_encoding_network.py [[1]](#1)
  - test1.npy, test2_rot.npy, test2_x.npy, test2_y.npy to `AMR-Policies/gibson2/agents/tf_agents/agents/reinforce/examples/v1`
  - train_eval_rnn_m_reinforce.py and train_shell.sh to `AMR-Policies/gibson2/agents/tf_agents/agents/reinforce/examples/v1`
    - Note: modified versions of train_eval_rnn.py and train_single_env.sh found in `gibson2/agents/tf_agents/agents/sac/examples/v1` [[1]](#1)
- The following files are modified AMR versions
  - reinforce_agent.py at `AMR-Policies/gibson2/agents/tf_agents/agents/reinforce` [[1]](#1)
  - suite_gibson.py at `AMR-Policies/gibson2/agents/tf_agents/environments` [[1]](#1)
  - locomotor_env.py at `AMR-Policies/gibson2/envs` [[2]](#2)
- Replace the following file with modified:
  - recurrent.py at `tensorflow_core/python/keras/layers` [[3]](#3)

### Training
```
cd AMR-Policies/gibson2/agents/tf_agents/agents/reinforce/examples/v1
./train_shell.sh
```

### Testing
- In train_shell.sh, change
    - headless to gui
    - Add flag `--eval-only True`
- To see previous checkpoint policies, change checkpoint header in `test/train/checkpoint` and `test/train/policy/checkpoint`
- flag `--random_init_m 2`, 2: training, 3: test, 4: test with enlarged set of initial conditions with testing data in `AMR-Policies/gibson2/agents/tf_agents/agents/reinforce/examples/v1`

## References
<a id="1">[1]</a> 
S. Guadarrama, A. Korattikara, O. Ramirez, P. Castro, E. Holly, S. Fishman, K. Wang, E. Gonina, N. Wu, E. Kokiopoulou, L. Sbaiz, J. Smith, G. Bart ́ok, J. Berent, C. Harris, V. Vanhoucke, and E. Brevdo.   TF-Agents:  A library for reinforcement learning in tensorflow,  2018.   URL https://github.com/tensorflow/agents.

<a id="2">[2]</a> 
F. Xia, W. B. Shen, C. Li, P. Kasimbeg, M. E. Tchapmi, A. Toshev, R. Martı́n-Martı́n, and
S. Savarese. Interactive Gibson benchmark: A benchmark for interactive navigation in clut-
tered environments. IEEE Robotics and Automation Letters, 5(2):713–720, 2020.

<a id="3">[3]</a> 
M. Abadi, A. Agarwal, P. Barham, E. Brevdo,
Z. Chen, C. Citro, G. S. Corrado, A. Davis,
J. Dean, M. Devin, S. Ghemawat, I. Goodfellow,
A. Harp, G. Irving, M. Isard, R. Jozefowicz, Y. Jia,
L. Kaiser, M. Kudlur, J. Levenberg, D. Mané, M. Schuster,
R. Monga, S. Moore, D. Murray, C. Olah, J. Shlens,
B. Steiner, I. Sutskever, K. Talwar, P. Tucker,
V. Vanhoucke, V. Vasudevan, F. Viégas,
O. Vinyals, P. Warden, M. Wattenberg, M. Wicke,
Y. Yu, and X. Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.
