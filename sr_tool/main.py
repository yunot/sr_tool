import matplotlib.pyplot as plt
import tensorflow as tf

from utils import test

if __name__ == '__main__':
    # test.test_fsrgan_openvino(isDynamic=True)
    # tf.debugging.set_log_device_placement(True)
    # test.test_sesr_large('resources/0-lobby')
    test.test_sesr('resources/images')

