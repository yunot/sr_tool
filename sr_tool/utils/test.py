# @Author: yunotao :)
# @Date: 7/26/2023 :)
import sys

import numpy as np
import tensorflow as tf
import pandas as pd
from infer import infer_fsrgan_openvino
from infer import infer_sesr
import time
from openvino.runtime import Core
from openvino.tools.mo import convert_model


def test_fsrgan_openvino(isDynamic: bool):
    tf.debugging.set_log_device_placement(True)
    # Model Preparation
    core = Core()
    start_time = time.perf_counter()
    ov_model = convert_model(saved_model_dir='resources/models/Fast_SRGAN/saved_model')
    print(f'convert_duration: {time.perf_counter() - start_time}')
    # dynamic input shape
    if isDynamic:
        ov_model.reshape([1, -1, -1, 3])
    else:
        ov_model.reshape([1, 266, 586, 3])

    # Model Compilation
    start_time = time.perf_counter()
    compiled_model = core.compile_model(ov_model)
    print(f'compile_duration: {time.perf_counter() - start_time}')

    # Testing!!!
    iteration = [
        10, 20, 40
    ]

    py = sys.version.split("(")[0].strip()
    version = f"py{py}_tf{tf.__version__}_numpy{np.__version__}"
    infer_time_dict = {}
    mean_infer_time = [0, 0, 0]

    for _ in range(2):
        # warm-up
        infer_fsrgan_openvino.infer(compiled_model, isDynamic)

    for index, it in enumerate(iteration):
        print(f"start infer iteration: {index}")
        infer_time = 0
        subDict = {}
        num = 0
        for _ in range(it):
            inference_per_time_list = infer_fsrgan_openvino.infer(compiled_model, isDynamic)
            num = len(inference_per_time_list)
            for infer_per_time in inference_per_time_list:
                infer_time += infer_per_time
                key = _ * num
                subDict[key] = infer_per_time
        infer_time_dict[index] = subDict
        infer_time /= (it * num)
        infer_time = format(infer_time, ".4f")
        print(f"iteration:{index}, mean_infer_time: {infer_time}")
        mean_infer_time[index] = infer_time

    infer_per_time_df = pd.DataFrame(infer_time_dict)
    infer_mean_time_df = pd.DataFrame({'50': {f"{version}": mean_infer_time[0]},
                                       '100': {f"{version}": mean_infer_time[1]},
                                       '200': {f"{version}": mean_infer_time[2]}
                                       })

    prefix = 'dynamic' if isDynamic else 'specific'
    infer_per_time_df.to_csv(
        f'results/inference_time/fsrgan/{prefix}_input_shape/DML_{prefix}_fsrgan_openvino_inference_per_time_py{py}_tf{tf.__version__}.csv')

    infer_mean_time_df.to_csv(
        f'results/inference_time/fsrgan/{prefix}_input_shape/DML_{prefix}_fsrgan_openvino_inference_mean_time_py{py}_tf{tf.__version__}.csv',
        index=True,
        index_label="env")


def test_sesr(img_dir):
    # tf.debugging.set_log_device_placement(True)
    # Model Preparation
    start_time = time.perf_counter()
    # model_path = 'resources/models/SESR_m5_f16_x2_fs256_collapsedTraining_FP32'
    model_path = 'resources/models/SESR_m5_FP32'
    model = tf.saved_model.load(model_path)
    print(f'sesr_model_load_duration: {time.perf_counter() - start_time}')

    iteration = [
        10, 20, 40
    ]

    py = sys.version.split("(")[0].strip()
    version = f"py{py}_tf{tf.__version__}_numpy{np.__version__}"
    infer_time_dict = {}
    mean_infer_time = [0, 0, 0]

    for _ in range(2):
        # warm-up
        infer_sesr.infer(model, img_dir)

    for index, it in enumerate(iteration):
        print(f"start infer iteration: {index}")
        infer_time = 0
        subDict = {}
        num = 0
        for _ in range(it):
            inference_per_time_list, image_names_list = infer_sesr.infer(model, img_dir)
            num = len(inference_per_time_list)
            for infer_per_time in inference_per_time_list:
                infer_time += infer_per_time
                key = _ * num
                subDict[key] = infer_per_time
        infer_time_dict[index] = subDict
        infer_time /= (it * num)
        infer_time = format(infer_time, ".4f")
        print(f"iteration:{index}, mean_infer_time: {infer_time}")
        mean_infer_time[index] = infer_time

    infer_per_time_df = pd.DataFrame(infer_time_dict)
    infer_mean_time_df = pd.DataFrame({'50': {f"{version}": mean_infer_time[0]},
                                       '100': {f"{version}": mean_infer_time[1]},
                                       '200': {f"{version}": mean_infer_time[2]}
                                       })

    infer_per_time_df.to_csv(
        f'results/inference_time/sesr/no_DML_sesr_minicap_540p_fp32_inference_per_time_py{py}_tf{tf.__version__}.csv')

    infer_mean_time_df.to_csv(
        f'results/inference_time/sesr/no_DML_sesr_minicap_540p_fp32_inference_mean_time_py{py}_tf{tf.__version__}.csv',
        index=True,
        index_label="env")


def test_sesr_1_folder(model, img_dir, size):
    py = sys.version.split("(")[0].strip()
    version = f"py{py}_tf{tf.__version__}_numpy{np.__version__}"

    infer_time = 0
    inference_per_time_list, image_names_list = infer_sesr.infer(model, img_dir, size)
    num = len(inference_per_time_list)
    for infer_per_time in inference_per_time_list:
        infer_time += infer_per_time
    infer_time /= num
    infer_time = format(infer_time, ".4f")
    print(f"mean_infer_time: {infer_time}")
    infer_time_float = float(infer_time) * 1000

    infer_per_time_df = pd.DataFrame({'image_name': image_names_list,
                                      'infer_time': inference_per_time_list})
    infer_per_time_df.insert(2, "mean_infer_time", np.nan)
    infer_per_time_df.at[0, "mean_infer_time"] = infer_time
    infer_per_time_df.to_csv(
        f'results/inference_time/sesr/lobby/no_DML_lobby_{size}x{size}_{round(infer_time_float)}ms_sesr_fp32_inference_per_time_py{py}_tf{tf.__version__}.csv')

    return infer_time_float


def test_sesr_large(img_dir, size=0):
    start_time = time.perf_counter()
    model_path = 'resources/models/SESR_m5_f16_x2_fs256_collapsedTraining_FP32'
    model = tf.saved_model.load(model_path)
    print(f'sesr_model_load_duration: {time.perf_counter() - start_time}')

    iteration = 10
    mean_time_list = []
    for _ in range(iteration):
        infer_mean_time = test_sesr_1_folder(model, img_dir, size)
        mean_time_list.append(infer_mean_time)
    mean_infer_time_df = pd.DataFrame(mean_time_list)
    mean_infer_time_df.to_csv(
        f'results/inference_time/sesr/lobby/avg_no_DML_lobby_{size}x{size}_sesr_fp32_inference_time.csv',
        index_label='iteration',
        header=['mean_infer_time'])
