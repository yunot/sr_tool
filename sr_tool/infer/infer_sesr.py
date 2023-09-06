# @Author: yunotao :)
# @Date: 7/26/2023 :)
import os
import time
import numpy as np
import tensorflow as tf
from utils import sesr_utils
import cv2


def infer(model, img_dir, size=0):
    tf.debugging.set_log_device_placement(True)
    # Input Image Preparation
    image_lr_dir = img_dir
    # Get all image paths
    image_paths = [os.path.join(image_lr_dir, x) for x in os.listdir(image_lr_dir)]
    infer_time_list = []
    image_names_list = []
    for image_path in image_paths:
        # RGB to Y
        # IMAGE = imageio.imread(image_path, pilmode="RGB")
        IMAGE = cv2.imread(image_path, 1)
        IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)

        if size:
            IMAGE = cv2.resize(IMAGE, (size, size), interpolation=cv2.INTER_CUBIC)

        IMAGE = tf.convert_to_tensor(IMAGE)
        IMAGE = tf.cast(IMAGE, dtype=tf.float32)
        IMAGE_ycrcb = sesr_utils.rgb_to_ycbcr(IMAGE)
        IMAGE_y = IMAGE_ycrcb[..., 0:1] / 255
        IMAGE_cb = IMAGE_ycrcb[..., 1:2]
        IMAGE_cr = IMAGE_ycrcb[..., 2:3]

        out_cb, out_cr = tf.image.resize([IMAGE_cb, IMAGE_cr], method='nearest',
                                 size=[IMAGE_cb.shape[0]*2, IMAGE_cb.shape[1]*2])

        IMAGE_y = tf.reshape(IMAGE_y, shape=(1, IMAGE_y.shape[0], IMAGE_y.shape[1], 1))
        # Once the file is in the desired format, just do:
        #
        # Compute the upscaled image for a trained model

        start_time = time.perf_counter()
        model_out_y = model(IMAGE_y)
        # model_out_y.shape=(1, 532, 1172, 1), dtype=float32
        # print(model_out_y)
        model_out_y = model_out_y[0] * 255
        output = tf.concat([model_out_y, out_cb, out_cr], axis=2)

        IMAGE_rgb = sesr_utils.ycbcr_to_rgb(output)
        IMAGE_bgr = cv2.cvtColor(np.asarray(IMAGE_rgb), cv2.COLOR_RGB2BGR)

        process_time = time.perf_counter() - start_time
        infer_time_list.append(process_time)

        # output_dir = f'results/image_sr/image_sesr/lobby/lobby_{size}x{size}'
        output_dir = f'results/image_sr/image_sesr/minicap_540p'
        new_image_path = 'sesr_' + os.path.basename(image_path)
        if size:
            new_image_path = f'sesr_{size}x{size}_' + os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_dir, new_image_path), IMAGE_bgr)
        image_names_list.append(os.path.basename(image_path))
        # cv2.imwrite(f'comparison/{new_image_path}', IMAGE_bgr)


        print(process_time)

    return infer_time_list, image_names_list

if __name__ == '__main__':
    model_path = 'resources/models/SESR_m5_f16_x2_fs256_collapsedTraining_FP32'
    model = tf.saved_model.load(model_path)
    infer(model)