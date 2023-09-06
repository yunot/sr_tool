# @Author: yunotao :)
# @Date: 7/26/2023 :)
import os
import cv2
import numpy as np
import time



def infer(compiled_model, isDynamic):
    # Image Preparation
    if isDynamic:
        image_lr_dir = 'resources/image_lr'
    else:
        image_lr_dir = 'resources/image_lr_586'
    # Get all image paths
    image_paths = [os.path.join(image_lr_dir, x) for x in os.listdir(image_lr_dir)]

    # Infer Request Created
    infer_request = compiled_model.create_infer_request()
    # Run inference on lr_images
    inference_time_list = []
    for image_path in image_paths:
        # Read image
        low_res = cv2.imread(image_path, 1)
        # low_res = cv2.resize(low_res, (1280, 720), interpolation=cv2.INTER_CUBIC)

        # Convert to RGB (opencv uses BGR as default)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

        # Rescale to 0-1.
        low_res = low_res / 255.0
        low_res = np.expand_dims(low_res, axis=0)

        # Get super resolution image
        start_time = time.perf_counter()
        infer_request.start_async(low_res)
        infer_request.wait()
        sr_output = infer_request.get_output_tensor()
        end_time = time.perf_counter()

        sr = sr_output.data[0]
        # Rescale values in range 0-255
        sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

        # Convert back to BGR for opencv
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        inference_time = end_time - start_time
        print(f"inference time: {inference_time}")
        inference_time_list.append(inference_time)

        # Save the results:
        output_dir = 'results/image_sr/image_fsrgan/'
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), sr)
        new_image_path = 'fsrgan_' + os.path.basename(image_path)
        cv2.imwrite(f'comparison/{new_image_path}', sr)

    # print(inference_time_list)
    return inference_time_list


if __name__ == '__main__':
    ov_model = convert_model(saved_model_dir='resources/models/Fast_SRGAN/saved_model')
    infer()