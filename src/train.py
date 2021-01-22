# Zweck:
# Author: David Kostka
# Datum: 20.1.2021

# TODO: Script zum ausführen von Training
# python train.py

import os
from time import time
from absl import app, flags, logging
from absl.flags import FLAGS

import util.dataset as ds
import util.model as md

import pandas as pd
import numpy as np
import tensorflow as tf

print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

models_path = 'data/models/'
#model_name = 'single_nao_test'
logs_path = 'data/logs/'
chkpts_path = 'data/checkpoints/'
graphs_path = 'data/graphs/'
tables_path = 'data/tables/'

FLAGS = flags.FLAGS
flags.DEFINE_string('train_ds_path', '../data/datasets/simulator/1_1/single_nao_train.record', 'Path to trainingset')
flags.DEFINE_string('val_ds_path', '../data/datasets/simulator/1_1/single_nao_val.record', 'Path to validation set')
flags.DEFINE_string('model_path', 'single_nao_test', 'The path of the exported trained model files')

model_name = 'experimodel'

def compile_model(model):
    model.compile(loss=md.nao_loss, 
                optimizer="adam", 
                metrics=[md.bbox_mse, md.conf_bce]
                )
                
    logging.info(model.summary())
    #tf.keras.utils.plot_model(model, graphs_path + model_name + '.png', show_shapes=True)
    return model

def get_optimised_datasets():
    train_dset = ds.load_tfrecord_dataset(FLAGS.train_ds_path)
    val_dset = ds.load_tfrecord_dataset(FLAGS.val_ds_path)

    train_dset_opt = train_dset.shuffle(buffer_size=512).batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dset_opt = val_dset.shuffle(buffer_size=512).batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dset_opt, val_dset_opt

def train(model, trainset, valset):
    callbacks = [   #ReduceLROnPlateau(verbose=1),
                    EarlyStopping(patience=5, verbose=1, restore_best_weights=True, monitor='val_loss'), # Im Moment: Loss von bbox betrachten, später val_loss
                    ModelCheckpoint(chkpts_path + model_name + '/' + model_name + '_{epoch}.tf', verbose=1, save_weights_only=True),
                    TensorBoard(log_dir=logs_path + model_name + '/' + '{}'.format(time())) ]
                    
    model.fit(
        trainset,
        validation_data=valset,
        epochs=5,
        callbacks=[callbacks]
    )

def save_model(model):
    model_path = os.path.join(FLAGS.model_path, model_name)
    # Save Model as File
    model.save(model_path)

    # Save as Format compatible with CompiledNN
    model.save(model_path + '.hdf5', save_format='h5')

    # Convert and Save to TFLite Model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    with open(model_path + '.tflite', 'wb') as f:
        f.write(tflite_model)

def main(argv):
    tf.get_logger().setLevel('ERROR')
    model_name = os.path.basename(FLAGS.model_path)
    model = md.SingleNaoModel(ds.target_size, bnorm=False)
    model = compile_model(model)
    trainset, valset = get_optimised_datasets()
    train(model, trainset, valset)
    save_model(model)

if __name__ == '__main__':
    app.run(main)