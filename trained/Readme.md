
### vision_0.pth
vision_train_0.py
CNN extractor from cnn_0
saved weights only
returns x, y, z of the object or 0,0,0 if none
79% Accuracy

### reach_sb.zip

train_reach_sb.py
net_arch=[256, 256]
buffer_size=1_000_000,
batch_size=256,
learning_rate=0.0003,
learning_starts=1000,
gamma=0.95,
ent_coef='auto',

91% success rate