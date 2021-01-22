import pandas as pd
import numpy as np
import os
from absl import app, flags, logging
from absl.flags import FLAGS

# python generate_csv.py --raw_images_path=D:/workspace/datasets/simulator/1_1/images --csv_path=D:/workspace/datasets/simulator/1_1/labels.csv
# TODO: Use config file like yaml

FLAGS = flags.FLAGS
flags.DEFINE_string('raw_images_path', '', 'Path to images')
flags.DEFINE_string('csv_path', '', 'Path to original CSV')
flags.DEFINE_string('output_folder', '', 'Path to TFRecord output folder')
flags.DEFINE_string('full_csv_name', 'total_with_negatives.csv', 'Name of generated clean labels CSV')
flags.DEFINE_string('train_csv_name', 'train.csv', 'Name of train split CSV')
flags.DEFINE_string('test_csv_name', 'val.csv', 'Name of test split CSV')

def read_csv(path):
    return pd.read_csv(path, sep=r'\s*,\s*', engine='python')

def split_csv(csv_path, output_path, train_name, test_name, test_split=0.2):
    labels = read_csv(csv_path)
    labels = labels.sample(frac=1).reset_index(drop=True)

    count = len(labels)
    split_index = int(count - count * test_split)

    labels[:split_index].to_csv(os.path.join(output_path, train_name), index=False)
    labels[split_index:].to_csv(os.path.join(output_path, test_name), index=False)

def create_clean_csv(csv_path, img_path, output_path, nr_neg_labels_weight=0.3):
    raw_labels = pd.read_csv(csv_path, sep=r'\s*,\s*', index_col=None, engine='python')
    raw_labels = raw_labels[['name', 'minX', 'minY', 'maxX', 'maxY']]

    raw_labels['classes'] = 1
    labels_img_names = raw_labels['name'].unique()
    all_img_names = os.listdir(img_path)
    no_nao_img_names = np.setdiff1d(all_img_names, labels_img_names)
    negative_labels = pd.DataFrame(no_nao_img_names, columns=['name'])
    negative_labels['classes'] = 0
    negative_labels = negative_labels.sample(frac=1).reset_index(drop=True)
    raw_labels = raw_labels.append(negative_labels[:int(len(negative_labels) * nr_neg_labels_weight)])
    raw_labels = raw_labels[raw_labels.name != 'name']

    labels = raw_labels.fillna(0).groupby('name').filter(lambda x: len(x) == 1)
    labels = labels.sample(frac=1).reset_index(drop=True)

    return labels.to_csv(output_path, index=False)

def main(argv):
    if not os.path.exists(FLAGS.output_folder):
        os.mkdir(FLAGS.output_folder)

    clean_csv_path = os.path.join(FLAGS.output_folder, FLAGS.full_csv_name)

    create_clean_csv(FLAGS.csv_path, FLAGS.raw_images_path, clean_csv_path)
    logging.info('Created clean CSV with false negatives in: ' + clean_csv_path)

    split_csv(clean_csv_path, FLAGS.output_folder, FLAGS.train_csv_name, FLAGS.test_csv_name)
    logging.info('Splitting clean CSV into: ' + FLAGS.train_csv_name + ', ' + FLAGS.test_csv_name)
    logging.info('Done creating CSVs!')

if __name__ == '__main__':
    app.run(main)