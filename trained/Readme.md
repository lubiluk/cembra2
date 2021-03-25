
### 0/
train_0.py

SAC
```
hidden_sizes=[64, 64, 64]

env = TimeLimit(gym.make("PepperReach-v0", gui=False, dense=True),
            max_episode_steps=100)

model.train(steps_per_epoch=1000, epochs=1000)
```
Success rate up to 1.0

### 0/
train_1.py

SAC
```
hidden_sizes=[64, 64, 64]

env = TimeLimit(gym.make("PepperPush-v0", gui=False), max_episode_steps=100)
alpha=0.0002
model.train(steps_per_epoch=1000, epochs=1000)
```
Success rate 0.1

### vision_0.pth
vision_train_0.py
CNN extractor
returns x, y, z of the object or 0,0,0 if none
79% Accuracy