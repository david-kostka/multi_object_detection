'''
    This script calculates evaluation metrics on a model and test dataset

    Currently, it prints the following metrics:
    - total loss
    - bbox mse
    - confidence bce
    - false positive rate @ threshold
    - true positive rate @ threshold
    - (the confidence threshold based on a target false positive rate)

    Example command:
    python evaluate.py --model_path=../data/models/single_nao_test.hdf5 --test_ds_path=../data/datasets/simulator/1_1/single_nao_val.record
    
    Author: David Kostka
    Date: 20.1.2021
'''
from absl import app, flags, logging
from absl.flags import FLAGS

import util.dataset as ds
import util.model as md

import sklearn.metrics as metrics
import numpy as np
import tensorflow as tf

import pickle
import json

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', '', 'Path to model')
flags.DEFINE_string('test_ds_path', '', 'Path to testset')

scores_file = 'scores.json'
plots_file = 'prc.json'

def normalize_pixels(image, label):
    return image / 255, label

def get_confidences(img, label):
    return label[4]

def calculate_threshold(fpr, threshold, target_false_positive_rate=0.01):
    optimal_idx = np.argmin(np.abs(fpr - target_false_positive_rate))

    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold, optimal_idx

def calculate_roc(model, dset):
    true_confidences = dset.map(get_confidences, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    true_confidences = list(true_confidences.as_numpy_iterator())

    pred = model.predict(dset.batch(1))

    fpr, tpr, threshold = metrics.roc_curve(true_confidences, pred[..., 4])
    roc_auc = metrics.auc(fpr, tpr)

    return fpr, tpr, threshold

# Für Multi-Objekt erkennung noch weitere metriken wie mAP@50, mAP@75, ... speichern
def main(argv):
    tf.get_logger().setLevel('ERROR')

    test_dset = ds.load_tfrecord_dataset(FLAGS.test_ds_path)
    test_dset = test_dset.map(normalize_pixels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dset_opt = test_dset.shuffle(buffer_size=512).batch(16).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    logging.info('Loaded Dataset')

    model = tf.keras.models.load_model(FLAGS.model_path, custom_objects={'nao_loss': md.nao_loss, 'bbox_mse': md.bbox_mse, 'conf_bce': md.conf_bce})
    logging.info('Loaded Model')

    logging.info('Calculating Metrics...')
    results = model.evaluate(test_dset_opt)

    logging.info('Calculating ROC Values...')
    fpr_list, tpr_list, threshold_list = calculate_roc(model, test_dset)
    threshold, threshold_idx = calculate_threshold(fpr_list, threshold_list, target_false_positive_rate=0.01)
    logging.info('Done!')

    print(results)
    print([fpr_list[threshold_idx], tpr_list[threshold_idx], threshold])

    with open(scores_file, 'w') as fd:
        json.dump({'loss': results[0], 'bbox_mse': results[1], 'conf_bce': results[2]}, fd)

    with open(plots_file, 'w') as fd:
        json.dump({'prc': [{
                'false_positive': str(p),
                'true_positive': str(r),
                'threshold': str(t)
            } for p, r, t in zip(fpr_list, tpr_list, threshold_list)
        ]}, fd)

if __name__ == '__main__':
    app.run(main)