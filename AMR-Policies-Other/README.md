# AMR-Policies
Examples 1 and 2

### Requirements
The code was verified to work with the following:
 - Ubuntu 18.04
 - Python 3.7.6
 - Torch 1.4.0
 - Ray 0.8.4
 - Numpy 1.18.1
 
### Discrete Navigation Example
To train: 
```
python train_Discrete.py
```
### Maze Navigation Example
To train:
```
python train_Maze.py
```
To evaluate:
```
python test_policy.py
```
Training can be time consuming, so we provided some pre-trained models for AMR-PG and PG located in `models` folder.
