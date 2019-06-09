from read_data_tf import load_audionet_dataset
from preprocess_data import splits
from models import AudioNetPython
import math
import argparse

def make_tuple(record):
    return (record['data'], record['digit'])

def split(task, type):
    return lambda record : record['vp'] in splits[task][type][0]

def train_audionet(dataset_path, checkpoint_path, logdir, batch_size):
    dataset = load_audionet_dataset(dataset_path)

    train_dataset = dataset.filter(split('digit', 'train'))
                            .map(make_tuple)
                            .shuffle(10000, seed=42).batch(batch_size)
                            .repeat()

    train_nb_samples = len(splits['digit']['train'][0])*500

    validation_dataset = dataset.filter(split('digit', 'validate'))
                            .map(make_tuple)
                            .shuffle(10000, seed=42).batch(batch_size)
                            .repeat()

    validation_nb_samples = len(splits['digit']['validate'][0])*500

    test_dataset = dataset.filter(split('digit', 'test'))
                            .map(make_tuple)
                            .shuffle(10000, seed=42).batch(batch_size)
                            .repeat()

    test_nb_samples = len(splits['digit']['test'][0])*500

    model = AudioNetPython.build_model()

    tb_callback = tf.keras.callbacks.TensorBoard(logdir=logdir,
                                                histogram_freq=1, 
                                                write_grads=True, 
                                                batch_size=batch_size)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                            save_weights_only=True, 
                                                            save_best_only=True)

    model.fit(train_dataset,
            epochs=5, 
            steps_per_epoch=math.ceil(train_nb_samples/batch_size),
            validation_data=validation_dataset,
            validation_steps=math.ceil(validation_nb_samples/batch_size),
            shuffle=False,
            callbacks=[tb_callback, checkpoint_callback])

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Training script for tensorflow.keras AudioNet model.")
    parser.add_argument('-i','--input_dataset', help="path to TFRecord file", required=True)
    parser.add_argument('-o','--checkpoint_output', help="path to checkpoint folder", required=True)
    parser.add_argument('-l','--logdir', help="path to logdir", required=True)
    parser.add_argument('-b','--batch_size', help="Batch size", required=True)

    args = parser.parse_args()

    train_audionet(args.input_dataset, args.checkpoint_output, args.logdir, args.batch_size)


