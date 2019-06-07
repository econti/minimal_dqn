# Minimal PyTorch DQN

Minimal PyTorch 1.1.0 implementations of:
- Deep Q-Networks [[reference](https://www.nature.com/articles/nature14236)]
- Double Deep Q-Networks [[reference](https://arxiv.org/abs/1509.06461)]
- Bayesian Deep Q-Networks [[reference](https://arxiv.org/abs/1802.04412)] (in progress)

### Installation
```
virutalenv env
. env/bin/activate
pip install -r requirements.txt
```

MacOS users will also need to install libomp to get PyTorch working due to [issue #20030](https://github.com/pytorch/pytorch/issues/20030)
```
brew install libomp
```

### Usage

DDQN on Atari
```
python main.py \
    --env "PongNoFrameskip-v4" --CnnDQN --learning_rate 0.00001 \
    --target_update_rate 0.1 --replay_size 100000 --start_train_ts 10000 \
    --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 30000 --max_ts 1400000 \
    --batch_size 32 --gamma 0.99 --log_every 10000
```

DDQN on Cartpole
```
python main.py \
    --env "CartPole-v0" --learning_rate 0.001 --target_update_rate 0.1 \
    --replay_size 5000 --start_train_ts 32 --epsilon_start 1.0 --epsilon_end 0.01 \
    --epsilon_decay 500 --max_ts 10000 --batch_size 32 --gamma 0.99 --log_every 200
```

Bayesian DQN on Atari
```
python main.py \
    --env "PongNoFrameskip-v4" --CnnDQN --learning_rate 0.00001 \
    --target_update_rate 0.1 --replay_size 100000 --start_train_ts 10000 \
    --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 30000 --max_ts 1400000 \
    --batch_size 32 --gamma 0.99 --log_every 10000 --BayesianDQN
```

Some code borrowed from:
 - https://github.com/higgsfield/RL-Adventure
 - https://github.com/kazizzad/BDQN-MxNet-Gluon
