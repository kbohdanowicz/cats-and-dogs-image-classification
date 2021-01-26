import os

from data_processing import get_images_data, get_train_test_data
from cnn import run_cnn, cnn_predict
from knn import run_knn
import tensorflow as tf


def config_cnn():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def main():
    # Classifiers
    (raw_images, histogram, labels) = get_images_data(load_from_file=True)

    train_test_data_raw_images = get_train_test_data(raw_images, labels, ratio=0.25)
    train_test_data_histogram = get_train_test_data(histogram, labels, ratio=0.25)

    run_knn(train_test_data_raw_images, 'raw images', find_best=False)
    run_knn(train_test_data_raw_images, 'raw images', find_best=True)

    run_knn(train_test_data_histogram, 'histogram', find_best=False)
    run_knn(train_test_data_histogram, 'histogram', find_best=True)

    # CNN
    run_cnn(type=1)
    run_cnn(type=2)
    run_cnn(type=3)
    run_cnn(type=4)


def predict():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    cnn_predict('model_1_acc_0.5573999881744385.h5', 1)
    cnn_predict('model_2_acc_0.7954000234603882.h5', 2)
    cnn_predict('model_3_acc_0.6855999827384949.h5', 3)
    cnn_predict('model_4_acc_0.9264000058174133.h5', 4)


if __name__ == '__main__':
    config_cnn()
    main()
    predict()
