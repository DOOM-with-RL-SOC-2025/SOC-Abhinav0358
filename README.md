# Teaching AI to Play DOOM 🎮

This is my project where I train an AI agent to play the classic DOOM game using reinforcement learning. It's been a fun journey watching the AI go from randomly shooting walls to actually becoming decent at the game!

The core idea is pretty straightforward - I give the AI screenshots of the game, and it learns to take actions (move left, move right, shoot) to maximize its score. I'm using PPO (Proximal Policy Optimization) which is currently one of the best algorithms for this kind of visual learning.

## What makes this cool

I've implemented several things that I'm pretty excited about:
- The AI can handle different DOOM scenarios (I started with basic aiming, then moved to defending a position)
- It uses a convolutional neural network to "see" the game just like humans do
- You can actually watch it train in real-time, which is mesmerizing
- Everything saves automatically so you don't lose progress if something crashes
- The whole thing follows modern RL standards, so it's easy to extend

## Getting it running

You'll need Python 3.8 or newer. I've tested this on Windows but it should work fine on Mac/Linux too. Fair warning - training can be pretty memory intensive, so 4GB+ RAM is recommended. A GPU helps a lot but isn't strictly necessary.

Here's how to get started:

**First, grab the code:**
```bash
git clone https://github.com/DOOM-with-RL-SOC-2025/SOC-Abhinav0358.git
cd SOC-Abhinav0358
```

**Set up a virtual environment** (trust me, you want this):
```bash
python -m venv env

# On Windows:
env\Scripts\activate

# On Mac/Linux:
source env/bin/activate
```

**Install the dependencies:**
```bash
pip install -r requirements.txt
```

If that doesn't work for some reason, you can install everything manually:
```bash
pip install stable-baselines3[extra]
pip install gymnasium
pip install vizdoom
pip install opencv-python
pip install numpy
```

## How everything is organized

I've tried to keep the code structure pretty clean. Here's what does what:

```
SOC-Abhinav0358/
├── vizdoomenv.py            # This wraps ViZDoom to work with modern RL libraries
├── train.py                 # Main training script - run this to start training
├── test.py                  # Load a trained model and watch it play
├── trainNlog.py             # Handles saving checkpoints during training
├── basic-tut.py             # Simple tutorial script I used for learning
├── ViZDoom/                 # The actual game engine and scenarios
│   └── scenarios/           # Different game levels (basic.cfg, defend_the_center.cfg)
├── train/                   # Your trained models get saved here
└── logs/                    # TensorBoard logs for monitoring training
```

## Actually running the thing

**To start training:**
```bash
python train.py
```

That's it! The AI will start learning. You'll see output showing how it's doing - at first the rewards will be pretty negative (it's bad at the game), but they should improve over time.

**To test a trained model:**
```bash
python test.py
```

This loads a saved model and lets you watch it play. It's honestly pretty satisfying to see the AI that was once shooting at walls now actually hitting targets.

## What to expect during training

The training process is pretty interesting to watch. I've set it up to train for 100,000 steps by default, which takes about an hour on my machine.

Here's roughly what happens:
- **First 10,000 steps**: The AI is basically random, just exploring
- **10,000-30,000 steps**: It starts to figure out basic patterns
- **30,000+ steps**: It gets good at the specific task

You can monitor progress in real-time by running:
```bash
tensorboard --logdir=logs/
```
Then open http://localhost:6006 in your browser. You'll get nice graphs showing how the AI's performance improves over time.

## Tweaking the training

If you want to experiment with different settings, here are the main knobs you can turn:

```python
model = PPO('CnnPolicy', env,
    learning_rate=0.0001,     # How fast the AI learns (smaller = more stable)
    n_steps=4096,             # How many game steps before updating the brain
    total_timesteps=100000,   # Total training duration
    verbose=1                 # How much output you want to see
)
```

I've found these settings work pretty well, but feel free to experiment. If training seems unstable (rewards jumping around wildly), try reducing the learning rate. If it's learning too slowly, you can increase it a bit.

## Results I've gotten

After about 40,000 training steps on the basic scenario, my AI can hit targets pretty consistently - around 89% success rate, which is way better than my own DOOM skills honestly.

The "defend the center" scenario is harder - it takes about 60,000 steps to get good performance, but once trained it can handle multiple enemies pretty well.

## How it actually works

The technical details are pretty cool if you're into that sort of thing:

The AI sees the game as 100x160 pixel grayscale images (I convert from color to save memory). It processes these through a convolutional neural network - basically the same type of network that's used for image recognition.

The network has two "heads":
- **Actor**: Decides what action to take (move left, right, or shoot)  
- **Critic**: Estimates how good the current situation is

The PPO algorithm balances exploration (trying new things) with exploitation (doing what it knows works). It's pretty elegant - the AI naturally learns to be more conservative when it's doing well and more exploratory when it's struggling.

## If you want to contribute

I'd love to see what improvements people come up with! Some ideas I've been thinking about:
- More complex DOOM scenarios
- Different neural network architectures  
- Comparison with other RL algorithms
- Better reward shaping

If you want to contribute, just fork the repo, make your changes, and send a pull request. The usual GitHub workflow.

---

*Built with curiosity and way too much coffee ☕*  
