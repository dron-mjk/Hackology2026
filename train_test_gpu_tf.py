#!/usr/bin/env python3
"""Quick TensorFlow GPU training test script.

Usage:
  python train_test_gpu_tf.py --epochs 2 --batch-size 128
"""

import argparse
import tensorflow as tf


def main(args):
    print("TensorFlow version:", tf.__version__)

    gpus = tf.config.list_physical_devices('GPU')
    print("Physical GPUs:", gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled memory growth for GPUs.")
        except Exception as e:
            print("Could not set memory growth:", e)
    else:
        print("No GPU detected — training will run on CPU.")

    print("Logical devices:", tf.config.list_logical_devices())

    # Load a small dataset (MNIST)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    # Simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"]) 

    print("Starting training...")
    model.fit(x_train, y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_split=0.1)

    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    args = parser.parse_args()
    main(args)
