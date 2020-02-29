This repo contains code for our paper [State-only Imitation with Transition Dynamics Mismatch](https://arxiv.org/abs/2002.11879) published at ICLR 2020.

The code heavily uses the RL machinery from [this awesome repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) with RL algorithms implemented in PyTorch. We also use some functionality from [OpenAI baselines](https://github.com/openai/baselines). The code was tested with the following packages:

* python 3.6.6
* pytorch 0.4.1
* gym  0.10.8

## Running command
To run MuJoCo experiments, use the command below. Update the path to the directory with the expert demonstrations. The demonstrations used in our experiments can be downloaded from [this Google drive](https://drive.google.com/drive/folders/1c71B5A5puBiK0itApZWzvfpmIYYNs5gp?usp=sharing) link. Edit _default_config.yaml_ to change the hyperparameters.

```
python main.py --env-name "Hopper-v2" --config-file "default_config.yaml" --experts-dir <add-path-here> --seed=$RANDOM
```

## Credits
1. [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
2. [OpenAI baselines](https://github.com/openai/baselines)
