from typing import Callable

import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
from scc4onnx import order_conversion

import os
import pathlib

def pytorch2onnx(*args, **kwargs):
    torch.onnx.export(*args, **kwargs)

def onnx2tensorflow(onnx_model_path: str,
                    input_op_names_and_order_dims: list = [0, 3, 2, 1],
                    *args, **kwargs):
    onnx_model = onnx.load(onnx_model_path)
    onnx_model = order_conversion(onnx_grap = onnx_model,
                                  input_op_names_and_order_dims = input_op_names_and_order_dims,
                                  *args, **kwargs)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(onnx_model_path.split('onnx') + 'h5')

def tf2tflite(tf_model_path: str,
              representative_data_gen: Callable):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint 8 (APIs added in r2.3)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model_quant = converter.convert()
    path, fname = os.path.split(tf_model_path)
    tflite_model_dir = pathlib.Path(path)
    tflite_model_dir.write_bytes('quant_' + fname.split('.h5') + '.tflite')