### Setup
- Make conda environment
  - conda create -n igibson python=3.6 anaconda
  - conda activate igibson
- Install Cuda 10.0
  - Extra install information if needed https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
- Install CuDNN 7.6.5 for Cuda 10.0 (Runtime and Developer library)
  - Installation guide https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
- Install EGL dependency
  - sudo apt-get install libegl1-mesa-dev
- Install pytorch
- Install tensorflow 1.15.0
  - pip install tensorflow-gpu==1.15.0

### Gibson Install
- Repo: iron-lab/Gibson-AMR-
- Install igibson
  - cd iGibson
  - source activate igibson
  - pip install -e .
- Add folders assets and dataset to iGibson/gibson2
  - Assets download: http://svl.stanford.edu/igibson/docs/installation.html
  - Additional assets to be put in asset folder (interactive objects): 
  - Need Rs + Placida scene downloaded for dataset..fill out form https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform
- Verify igibson
  - Should see turtlebot in apt taking random actions (need cv2 version 4.2.0 not 4.3.0 for this to work)
  - cd examples/demo
  - python env_example.py
- Install tf-agents - 
  - cd iGibson/gibson2/agents
  - pip install -e .
- Verify training works w/o errors
  - cd iGibson/gibson2/agents/tf_agents/agents/reinforce/examples/v1
  - ./train_single_env.sh

### Training Info
- ./train_shell.sh to set some high level flags
  - Flags are defined in train_eval_rnn_m_reinforce.py along with training code
- Robot config file is located in iGibson/examples/configs as turtlebot_AMR.yaml
  - Specified here:
    - Init + target loc
    - Reward weights
    - Sensors
    - Etc
- Files I have made edits to:
  - iGibson/gibson2/agents/tf_agents/environments/suite_gbson.py
    - To create my own env_type ‘gibson_meg’ for more flexibility with randomness of initial and target positions
- iGibson/gibson2/envs/locomotor_env.py
  - Two new classes 
    - NavigateRandomInitEnvSim2Real
      - Interactive objects specified here and where they can be generated pos wise (sorta hacky)
    - NavigateRandomInitEnv
      - If random_init_m = 0 in train_single_env.sh then will use init pos + orientation in yaml config, else random pos + orientation
- iGibson/gibson2/agents/tf_agents/agents/reinforce/reinforce_agent.py
  - Group lasso added to total loss
- iGibson/gibson2/agents/tf_agents/networks/rnn_enconding_network.py 
  - Instead of lstm_encoding_network.py
  - Use RNN cells instead of LSTM
- iGibson/gibson2/agents/tf_agents/networks/actor_distribution_rnn_m_network.py
  - Instead of actor_distribution_rnn_network.py
  - Calls on rnn_encoding_network
- To render test checkpoint:
  - In train_shell.sh
    - Headless to gui
    - Add flag eval-only True


#  iGibson: the Interactive Gibson Environment

<img src="./docs/images/igibsonlogo.png" width="600"> <img src="./docs/images/igibson.gif" width="250"> 

### Large Scale Interactive Simulation Environments for Robot Learning

iGibson, the Interactive Gibson Environment, is a simulation environment providing fast visual rendering and physics simulation (based on Bullet). It is packed with a dataset with hundreds of large 3D environments reconstructed from real homes and offices, and interactive objects that can be pushed and actuated. iGibson allows researchers to train and evaluate robotic agents that use RGB images and/or other visual sensors to solve indoor (interactive) navigation and manipulation tasks such as opening doors, picking and placing objects, or searching in cabinets.

### Latest Updates
[04/28/2020] Added support for Mac OSX :computer:

### Citation
If you use iGibson or its assets and models, consider citing the following publication:

```
@article{xia2020interactive,
         title={Interactive Gibson Benchmark: A Benchmark for Interactive Navigation in Cluttered Environments},
         author={Xia, Fei and Shen, William B and Li, Chengshu and Kasimbeg, Priya and Tchapmi, Micael Edmond and Toshev, Alexander and Mart{\'\i}n-Mart{\'\i}n, Roberto and Savarese, Silvio},
         journal={IEEE Robotics and Automation Letters},
         volume={5},
         number={2},
         pages={713--720},
         year={2020},
         publisher={IEEE}
}
```


### Release
This is the repository for iGibson (gibson2) 0.0.4 release. Bug reports, suggestions for improvement, as well as community developments are encouraged and appreciated. The support for our previous version of the environment, [Gibson v1](http://github.com/StanfordVL/GibsonEnv/), will be moved to this repository.

### Documentation
The documentation for this repository can be found here: [iGibson Environment Documentation](http://svl.stanford.edu/igibson/docs/). It includes installation guide (including data download), quickstart guide, code examples, and APIs.

If you want to know more about iGibson, you can also check out [our webpage](http://svl.stanford.edu/igibson), [our RAL+ICRA20 paper](https://arxiv.org/abs/1910.14442) and [our (outdated) technical report](http://svl.stanford.edu/igibson/assets/gibsonv2paper.pdf).

### Dowloading Dataset of 3D Environments
There are several datasets of 3D reconstructed large real-world environments (homes and offices) that you can download and use with iGibson. All of them will be accessible once you fill in this [form](https://forms.gle/36TW9uVpjrE1Mkf9A).

You will have access to ten environments with annotated instances of furniture (chairs, tables, desks, doors, sofas) that can be interacted with, and to the original 572 reconstructed 3D environments without annotated objects from [Gibson v1](http://github.com/StanfordVL/GibsonEnv/).

You will also have access to a [fully annotated environment: Rs_interactive](https://storage.googleapis.com/gibson_scenes/Rs_interactive.tar.gz) where close to 200 articulated objects are placed in their original locations of a real house and ready for interaction. ([The original environment: Rs](https://storage.googleapis.com/gibson_scenes/Rs.tar.gz) is also available). More info can be found in the [installation guide](http://svl.stanford.edu/igibson/docs/installation.html).

