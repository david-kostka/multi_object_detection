import util.dataset as ds
import os
from absl import app, flags, logging
from absl.flags import FLAGS

# python generate_tfrecord.py --raw_images_path=D:/workspace/datasets/simulator/1_1/images/ --csv_path=../data/datasets/simulator/1_1/labels_with_negatives.csv
# TODO: Use config file like yaml

FLAGS = flags.FLAGS
flags.DEFINE_string('raw_images_path', '', 'Path to images')
flags.DEFINE_string('csv_path', '', 'Path to label CSV')
flags.DEFINE_string('output_folder', '', 'Path to TFRecord output folder')

tfrecord_train_name = 'train.record'
tfrecord_test_name = 'val.record'

def main(argv):
    if not os.path.exists(FLAGS.output_folder):
        os.mkdir(FLAGS.output_folder)

    tfrecord_train_path = os.path.join(FLAGS.output_folder, tfrecord_train_name)
    tfrecord_test_path = os.path.join(FLAGS.output_folder, tfrecord_test_name)

    logging.info('Creating trainingset record file...')
    ds.create_tfrecord_from_csv(FLAGS.csv_path, FLAGS.raw_images_path, tfrecord_train_path)
    logging.info('Created trainingset record file: ' + tfrecord_train_path)

    logging.info('Creating testset record file...')
    ds.create_tfrecord_from_csv(FLAGS.csv_path, FLAGS.raw_images_path, tfrecord_test_path)
    logging.info('Created testset record file: ' + tfrecord_test_path)
    logging.info('Done creating TFRecords!')

if __name__ == '__main__':
    app.run(main)