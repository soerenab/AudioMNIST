
import tensorflow as tf

def parse_audionet_records(record):
    audionet_feature = {
        'data' : tf.FixedLenFeature([8000], tf.float32),
        'digit' : tf.FixedLenFeature([], tf.int64),
        'gender' : tf.FixedLenFeature([], tf.int64),
        'vp' : tf.FixedLenFeature([], tf.int64)
    }
    example = tf.parse_single_example(record, audionet_feature)
    example['data'] = tf.reshape(example['data'], (8000,1))
    return example

def parse_alexnet_records(record):
    audionet_feature = {
        'data' : tf.FixedLenFeature([227*227], tf.float32),
        'digit' : tf.FixedLenFeature([], tf.int64),
        'gender' : tf.FixedLenFeature([], tf.int64),
        'vp' : tf.FixedLenFeature([], tf.int64)
    }
    example = tf.parse_single_example(record, audionet_feature)
    example['data'] = tf.reshape(example['data'], (227,227,1))
    return example

def load_audionet_dataset(path):
    raw_dataset = tf.data.TFRecordDataset(path)
    dataset = raw_dataset.map(parse_audionet_records)
    return dataset

def load_alexnet_dataset(path):
    raw_dataset = tf.data.TFRecordDataset(path)
    dataset = raw_dataset.map(parse_alexnet_records)
    return dataset