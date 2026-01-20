# Mitigating Overestimation in Offline Reinforcement Learning via One-Class Classification

This repository is the official implementation of \textbf{Mitigating Overestimation in Offline Reinforcement Learning via One-Class Classification}. 

This repository is another version of our repository. There're some modifications - theoretical basics, so update rule is changed and some anomaly models are included.

The implementation is based on [rlkit](https://github.com/rail-berkeley/rlkit)

## Requirements

Python = 3.8.11

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

RLOCC-SAC

```train
python ./examples/RLOCC_SAC.py --env <env_ids>
```



RLOCC-BEAR
```train
python ./examples/RLOCC_BEAR.py --env <env_ids>
```

with hyperparameters
'''hyperparameters
--env <env_ids>
--gpu <gpu numbers>
--qf_lr <critic learning rate>
--policy_lr <policy learning rate>
--seed <seed>
--nepochs <epochs for RL algorithm>
--epochs_svdd <epochs for svdd>
--epochs_ae <epochs for pretraining autoencoder>
--lr_svdd <learning rate of svdd>
--svdd_latent_dim <svdd_latent_dim>
--svdd_hidden_dim <svdd_hidden_dim>

For RLOCC_BEAR, the following hyperparameters exist:
--mmd_sigma <mmd_sigma> (default=20, type=float)
--kernel_type <type of kernel> (default='gaussian', type=str)
--target_mmd_thresh <target mmd thresh> (default=0.07, type=float)
--num_samples <number of samples> (default=100, type=int)
'''

For further hyperparameters, please see each file in examples folder.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

Pretrained parameters for Deep SVDD model is in the folder, weights.

## Results

Our model achieves the following performance on :

MuJoCo tasks

| Task Name              | RLOCC-SAC (OURS) | RLOCC-BEAR (OURS) | MOPO | MOReL | RAMBO | COMBO | BEAR  | ATAC | CQL  | ARMOR | IQL  | BC   |
|------------------------|------------------|-------------------|------|-------|-------|-------|-------|------|------|-------|------|------|
| hopper-med             | **96.68**        | **95.95**         | 28.0 | 95.4  | 92.8  | 97.2  | 30.77 | 85.6 | 86.6 | 101.4 | 66.3 | 29.0 |
| walker2d-med           | **107.88**       | 90.33             | 17.8 | 77.8  | 86.9  | 81.9  | 56.02 | 89.6 | 74.5 | 90.7  | 78.3 | 6.6  |
| halfcheetah-med        | **72.50**        | 45.10             | 42.3 | 42.1  | **77.6** | 54.2  | 37.14 | 53.3 | 44.4 | 54.2  | 47.4 | 36.1 |
| hopper-med-rep         | 95.88            | 89.35             | 67.5 | 93.6  | 96.6  | 89.5  | 31.13 | 102.5| 48.6 | 97.1  | 94.7 | 11.8 |
| walker2d-med-rep       | **101.84**       | 66.61             | 39.0 | 49.8  | 85.0  | 56.0  | 13.66 | 92.5 | 32.6 | 85.6  | 73.9 | 11.3 |
| halfcheetah-med-rep    | **78.60**        | 42.16             | 53.1 | 40.2  | 68.9  | 55.1  | 36.21 | 48.0 | 46.2 | 50.5  | 44.2 | 38.4 |
| hopper-med-exp         | **100.72**       | **113.34**        | 23.7 | 108.7 | 83.3  | 111.1 | 67.26 | 111.9| 111.0| 103.4 | 91.5 | 111.9|
| walker2d-med-exp       | **108.63**       | 96.34             | 44.6 | 95.6  | 68.3  | 103.3 | 43.80 | **114.2**| 98.7 | 112.2 | 109.6| 6.4  |
| halfcheetah-med-exp    | 79.61            | **92.90**         | 63.3 | 53.3  | 93.7  | 90.0  | 44.16 | **94.8**| 62.4 | 93.5  | 86.7 | 35.8 |

Adroit tasks

| Task Name        | RLOCC-SAC (OURS) | ATAC  | CQL   | ARMOR | IQL  | BC   |
|------------------|------------------|-------|-------|-------|------|------|
| pen-human        | **73.61**        | 53.1  | 37.5  | 72.8  | 71.5 | 34.4 |
| hammer-human     | **5.15**         | 1.5   | 4.4   | 1.9   | 1.4  | 1.5  |
| door-human       | **10.88**        | 2.5   | 9.9   | 6.3   | 4.3  | 0.5  |
| relocate-human   | 0.01             | 0.1   | 0.2   | 0.4   | 0.1  | 0.0  |
| pen-cloned       | **52.76**        | 43.7  | 39.2  | 51.4  | 37.3 | **56.9** |
| hammer-cloned    | 0.40             | 1.1   | 2.1   | 0.7   | 2.1  | 0.8  |
| door-cloned      | -0.0             | 3.7   | 0.4   | -0.1  | 1.6  | -0.1 |
| relocate-cloned  | -0.18            | 0.2   | -0.1  | -0.0  | -0.2 | -0.1 |

## Contributing
This code is built on the github repository https://github.com/rail-berkeley/rlkit.

For more information or usage, please refer to https://github.com/rail-berkeley/rlkit.

This code is released under the [LICENSE](LICENSE) terms.
