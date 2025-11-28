from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, loss as gloss, data as gdata

# Model
net = nn.Sequential()
net.add(nn.Dense(5, activation='relu'), nn.Dense(1))
net.initialize()

# Dummy Data
X = nd.random.uniform(shape=(20, 5))
y = nd.random.uniform(shape=(20, 1))

# Setup Trainer + Loss
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
loss_fn = gloss.L2Loss()

# Single training epoch (tiny)
for epoch in range(2):
    total_loss = 0
    for Xb, yb in gdata.DataLoader(gdata.ArrayDataset(X, y), batch_size=5, shuffle=True):
        with autograd.record():
            y_pred = net(Xb)
            loss = loss_fn(y_pred, yb)
        loss.backward()
        trainer.step(batch_size=5)
        total_loss += loss.mean().asscalar()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

