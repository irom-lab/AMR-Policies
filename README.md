### Setup
Installation instructions are a combination from http://svl.stanford.edu/igibson/docs/ and https://github.com/StanfordVL/GibsonSim2RealChallenge. Verified on Ubuntu 18.04.
- Make conda environment
```
  conda create -n igibson python=3.6 anaconda
  conda activate igibson
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
- Download and install iGibson
```
git clone https://github.com/StanfordVL/iGibson --recursive
cd iGibson
source activate igibson
pip install -e .
```
- Add folders "assets" and "dataset" to `iGibson/gibson2`
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
  - Add "agents" folder to iGibson/gibson2
  - Copy repo downloaded form https://github.com/StanfordVL/agents/tree/gibson_sim2real to agents folder
  - Install
  ```
  cd iGibson/gibson2/agents
  pip install -e .
  ```
- Add AMR related files
  - turtlebot_AMR.yaml to `iGibson/examples/configs`
  - actor_distribution_rnn_m_network.py to `iGibson/gibson2/agents/tf_agents/networks`
    - Note: a modified version of TF-agent's actor_distribution_rnn_network.py [[1]](#1)
  - rnn_encoding_network.py to `iGibson/gibson2/agents/tf_agents/networks`
    - Note: a modified version of TF-agent's lstm_encoding_network.py [[1]](#1)
  - test1.npy, test2_rot.npy, test2_x.npy, test2_y.npy to `iGibson/gibson2/agents/tf_agents/agents/reinforce/examples/v1`
  - train_eval_rnn_m_reinforce.py and train_shell.sh to `iGibson/gibson2/agents/tf_agents/agents/reinforce/examples/v1`
    - Note: modified versions of train_eval_rnn.py and train_single_env.sh found in `iGibson/gibson2/agents/tf_agents/agents/sac/examples/v1` [[1]](#1)
- Replace files with modified AMR versions
  - reinforce_agent.py at `iGibson/gibson2/agents/tf_agents/agents/reinforce` [[1]](#1)
  - suite_gibson.py at `iGibson/gibson2/agents/tf_agents/environments` [[1]](#1)
  - locomotor_env.py at `iGibson/gibson2/envs` [[2]](#2)
  - recurrent.py at `tensorflow_core/python/keras/layers` [[3]](#3)

### Training
```
cd iGibson/gibson2/agents/tf_agents/agents/reinforce/examples/v1
./train_shell.sh
```

### Testing
- In train_shell.sh, change
    - headless to gui
    - Add flag `--eval-only True`
- To see previous checkpoint policies, change checkpoint header in `test/train/checkpoint` and `test/train/policy/checkpoint`
- flag `--random_init_m 2`, 2: training, 3: test, 4: test with enlarged set of initial conditions

## References
<a id="1">[1]</a> 
S. Guadarrama, A. Korattikara, O. Ramirez, P. Castro, E. Holly, S. Fishman, K. Wang, E. Gonina, N. Wu, E. Kokiopoulou, L. Sbaiz, J. Smith, G. Bart ́ok, J. Berent, C. Harris, V. Vanhoucke, and E. Brevdo.   TF-Agents:  A library for reinforcement learning in tensorflow,  2018.   URL https://github.com/tensorflow/agents.

<a id="2">[2]</a> 
F. Xia, W. B. Shen, C. Li, P. Kasimbeg, M. E. Tchapmi, A. Toshev, R. Martı́n-Martı́n, and
S. Savarese. Interactive Gibson benchmark: A benchmark for interactive navigation in clut-
tered environments. IEEE Robotics and Automation Letters, 5(2):713–720, 2020.

<a id="3">[3]</a> 
Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.
