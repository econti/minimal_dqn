### Minimal PyTorch DQN Implementation

Minimal PyTorch 1.1.0 implementations of:
- Deep Q-Networks
- Double Deep Q-Networks
- Bayesian Deep Q-Networks (in progress)

Installation
```
virutalenv env
. env/bin/activate
pip install -r requirements.txt
```

MacOS users will also need to install libomp to get PyTorch working
```
brew install libomp
```

Usage
```
python main.py \
    --env "PongNoFrameskip-v4" --CnnDQN --learning_rate 0.00001 \
    --target_update_rate 0.1 --replay_size 100000 --start_train_ts 10000 \
    --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 30000 --max_ts 1400000 \
    --batch_size 32 --gamma 0.99 --log_every 10000
```
