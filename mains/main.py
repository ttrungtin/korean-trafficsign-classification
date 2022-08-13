import sys
import argparse
import os
import tensorflow as tf

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from model.lanet import LaNet
from data_loader.data_loader import load_data_each
from utils.preprocessing import preprocess


def main(mode, data_dir, epochs, batch_size):
    # -------------------------------------------------------------------------------------------------
    # Parameters
    n_out = 76
    train_dir = data_dir + "train.bin"
    valid_dir = data_dir + "valid.bin"
    test_dir = data_dir + "test.bin"

    # -------------------------------------------------------------------------------------------------
    if mode == 'tr':
        # Load dataset
        x_train, y_train = load_data_each(train_dir)
        x_valid, y_valid = load_data_each(valid_dir)

        # Pre-processing data
        x_train_processed = preprocess(x_train)
        x_valid_processed = preprocess(x_valid)

        # Training
        model = LaNet(n_out)
        model.training(x_train_processed, y_train, x_valid_processed, y_valid, epochs, batch_size)

    elif mode == 't':
        # Load dataset
        x_test, y_test = load_data_each(test_dir)

        # Pre-processing data
        x_test_processed = preprocess(x_test)

        # Testing
        model = LaNet(n_out)
        with tf.Session() as sess:
            model.saver.restore(sess, ROOT_DIR + "/logs/LaNet")
            accuracy = model.evaluate(x_test_processed, y_test, batch_size)
            print("Test Accuracy = {:.1f}%".format(accuracy*100))


if __name__ == "__main__":
    # -------------------------------------------------------------------------------------------------
    # Initialize parser
    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------------------------------------------
    # Add parameters
    parser.add_argument('mode', help='select tr | t (train|test)')
    parser.add_argument('-d', '--data', help='dataset directory')
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int)

    # -------------------------------------------------------------------------------------------------
    # Parse the arguments
    args = parser.parse_args()

    # -------------------------------------------------------------------------------------------------
    # Check arguments
    if (args.mode == 'tr') and ((args.epochs is None) or (args.batch_size is None)):
        print('Missing EPOCHS or BATCH SIZE variables for TRAINING mode')
        sys.exit()
    elif (args.mode == 't') and (args.batch_size is None):
        print('Missing BATCH SIZE variables for TESTING mode')
        sys.exit()
    else:
        main(args.mode, args.data, args.epochs, args.batch_size)
