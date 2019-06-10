from ..io.read_dataset import load_audionet_dataset
from .splits import splits
from ..models import audionet
import math
import os
import tensorflow as tf
import gc
import numpy as np
import glob


def make_tuple(record):
    return (record['data'], tf.one_hot(record['digit'], 10))


def split(task, type):
    set = tf.constant(list(splits[task][type][0]), dtype=tf.int64)
    return lambda record: tf.reduce_any(tf.equal(record['vp'], set))


def train(dataset_path, checkpoint_path, logdir, batch_size, epochs):
    dataset = load_audionet_dataset(dataset_path)

    train_dataset = dataset.filter(split('digit', 'train')) \
        .map(make_tuple) \
        .shuffle(10000, seed=42).batch(batch_size) \
        .repeat()

    train_nb_samples = len(splits['digit']['train'][0])*500

    validation_dataset = dataset.filter(split('digit', 'validate')) \
        .map(make_tuple) \
        .shuffle(10000, seed=42).batch(batch_size) \
        .repeat()

    validation_nb_samples = len(splits['digit']['validate'][0])*500

    test_dataset = dataset.filter(split('digit', 'test')) \
        .map(make_tuple) \
        .shuffle(10000, seed=42).batch(batch_size)

    test_nb_samples = len(splits['digit']['test'][0])*500

    model = audionet.build_model()

    if not os.path.isdir(logdir): os.mkdir(logdir)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                 batch_size=batch_size)

    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, "model.{epoch:02d}-{val_acc:.2f}"),
                                                            save_weights_only=True)

    gc_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch,_: gc.collect())

    history = model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=int(math.ceil(train_nb_samples/batch_size)),
              validation_data=validation_dataset,
              validation_steps=int(math.ceil(validation_nb_samples/batch_size)),
              shuffle=False,
              callbacks=[tb_callback, checkpoint_callback])

    best_epoch = np.argmax(history.history['val_acc']) + 1

    best_epoch_checkpoint = get_epoch_checkpoint(checkpoint_path, best_epoch)

    model.load_weights(best_epoch_checkpoint)

    scores = model.evaluate(test_dataset, steps=test_nb_samples)

    with open(os.path.join(checkpoint_path,f"evalution_epoch{best_epoch}.txt"), "w") as fh:
        for i, name in enumerate(model.metrics_names):
            fh.write(f"{name} : {scores[i]}\n")

def get_epoch_checkpoint(checkpoint_path, epoch):
    epoch_checkpoint = glob.glob(os.path.join(checkpoint_path, f"model.{epoch}-*.data*"))
    assert len(epoch_checkpoint) == 1
    epoch_checkpoint = epoch_checkpoint[0].split(".data")[0]
    return epoch_checkpoint

def test(dataset_path, checkpoint_path, epoch, batch_size):
    dataset = load_audionet_dataset(dataset_path)
    test_dataset = dataset.filter(split('digit', 'test')) \
        .map(make_tuple) \
        .shuffle(10000, seed=42).batch(batch_size)
    test_nb_samples = len(splits['digit']['test'][0])*500

    model = audionet.build_model()

    epoch_checkpoint = get_epoch_checkpoint(checkpoint_path, epoch)

    model.load_weights(epoch_checkpoint)

    scores = model.evaluate(test_dataset, steps=int(math.ceil(test_nb_samples/batch_size)))

    with open(os.path.join(checkpoint_path,f"evalution_epoch{epoch}.txt"), "w") as fh:
        for i, name in enumerate(model.metrics_names):
            fh.write(f"{name} : {scores[i]}\n")
