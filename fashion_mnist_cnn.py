import os
import mxnet as mx
from mxnet import gluon, autograd, nd, image, metric
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from mxnet.gluon.data import DataLoader


# -----------------------------
# 1. Configuration & Utilities
# -----------------------------

BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
PARAMS_FILE = "fashion_cnn.params"

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_ctx():
    """Choose GPU if available, otherwise CPU."""
    try:
        if mx.context.num_gpus() > 0:
            print("Using GPU")
            return mx.gpu()
    except Exception:
        pass
    print("Using CPU")
    return mx.cpu()


def get_transform():
    """Image transforms: resize, to tensor, normalize."""
    return transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(0.13, 0.31)
    ])


# -----------------------------
# 2. Data Loading & Preprocessing
# -----------------------------

def get_dataloaders(batch_size=BATCH_SIZE):
    """
    Download FashionMNIST, apply transforms, and
    return train/test DataLoaders.
    """
    transformer = get_transform()

    train_ds = datasets.FashionMNIST(train=True).transform_first(transformer)
    test_ds = datasets.FashionMNIST(train=False).transform_first(transformer)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    return train_loader, test_loader


# -----------------------------
# 3. Define CNN with Gluon
# -----------------------------

def build_net(ctx):
    """
    Define a simple CNN:
    Conv -> Pool -> Conv -> Pool -> Flatten -> Dense -> Dense
    """
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(32, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2),
            nn.Conv2D(64, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2),
            nn.Flatten(),
            nn.Dense(128, activation='relu'),
            nn.Dense(10)  # 10 Fashion-MNIST classes
        )
    net.initialize(ctx=ctx)
    return net


# -----------------------------
# 4. Training Loop
# -----------------------------

def train(net, train_loader, ctx, num_epochs=NUM_EPOCHS,
          lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'adam',
                            {'learning_rate': lr})

    for epoch in range(num_epochs):
        cumulative_loss = 0.0
        n_batches = 0

        for data, label in train_loader:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size)

            cumulative_loss += loss.mean().asscalar()
            n_batches += 1

        avg_loss = cumulative_loss / max(1, n_batches)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Average Loss: {avg_loss:.4f}")


# -----------------------------
# 5. Evaluation
# -----------------------------

def evaluate(net, test_loader, ctx):
    acc = metric.Accuracy()
    for data, label in test_loader:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        preds = net(data)
        acc.update(preds=[preds], labels=[label])

    name, value = acc.get()
    print(f"Test {name}: {value:.4f}")
    return value


# -----------------------------
# 6. Save & Reload Model
# -----------------------------

def save_model(net, filename=PARAMS_FILE):
    net.save_parameters(filename)
    print(f"Saved model parameters to '{filename}'")


def load_model(ctx, filename=PARAMS_FILE):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Parameter file '{filename}' not found. "
            f"Train and save the model first."
        )

    net = build_net(ctx)
    net.load_parameters(filename, ctx=ctx)
    print(f"Loaded model parameters from '{filename}'")
    return net


# -----------------------------
# 7. Inference on New Images
# -----------------------------

def predict_image(net, img_path, ctx):
    """
    Run inference on a custom image file.
    Expects a grayscale or RGB image. Resizes & normalizes just like training.
    """
    if not os.path.exists(img_path):
        print(f"[WARN] Image '{img_path}' not found. Skipping inference.")
        return

    print(f"Running inference on '{img_path}'")

    transformer = get_transform()

    # Read image (keeps channels; we let transform handle the rest)
    img = image.imread(img_path)

    # Apply same preprocessing: Resize -> ToTensor -> Normalize
    img = transformer(img)  # (C, H, W), float32

    # Add batch dimension: (1, C, H, W)
    img = img.expand_dims(axis=0).as_in_context(ctx)

    logits = net(img)
    pred_label_idx = int(nd.argmax(logits, axis=1).asscalar())
    pred_label_name = FASHION_CLASSES[pred_label_idx]

    print(f"Predicted Label Index: {pred_label_idx}")
    print(f"Predicted Label Name : {pred_label_name}")


# -----------------------------
# 8. End-to-end Pipeline
# -----------------------------

def main():
    # Step 1: Select device
    ctx = get_ctx()

    # Step 2: Load & preprocess data
    print("Loading FashionMNIST data...")
    train_loader, test_loader = get_dataloaders()

    # Step 3: Build CNN
    print("Building CNN model...")
    net = build_net(ctx)

    # Step 4: Train
    print("Starting training...")
    train(net, train_loader, ctx)

    # Step 5: Evaluate
    print("Evaluating on test set...")
    evaluate(net, test_loader, ctx)

    # Step 6: Save model parameters
    save_model(net, PARAMS_FILE)

    # Step 7: Reload model (to show it works)
    print("Reloading model from disk...")
    loaded_net = load_model(ctx, PARAMS_FILE)

    # Step 8: Inference on a sample image (optional)
    # Put your own image file here (e.g., 'sample_shoe.png')
    sample_image_path = "sample_shoe.png"
    predict_image(loaded_net, sample_image_path, ctx)


if __name__ == "__main__":
    main()

