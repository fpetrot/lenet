#!/usr/bin/env python
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt
from typing import List, Callable, Optional
import time
import os
import pathlib
import sys
import re

if len(sys.argv) != 2:
    print(f'Usage {sys.argv[0]} (float|int8_t)')
    sys.exit(1)

if  sys.argv[1] == "int8_t" or sys.argv[1] == "float":
    rep = sys.argv[1]
else:
    print(f'Usage {sys.argv[0]} (float|int8_t)')
    sys.exit(1)


# récupération du data-set: i == image, l == label (catégorie)
(_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()

# on passe de 28x28 en 32x32
test_images = np.pad(array = test_images, pad_width = 2, mode = 'constant', constant_values = 0)
test_images = np.expand_dims(test_images, axis=-1)
test_images = test_images[2:test_images.shape[0] - 2]

model_filename = f'lenet5_models/model_{rep}.tflite' 
interpreter = tf.lite.Interpreter(model_path = model_filename)
interpreter.allocate_tensors()

# Dump the whole content of a numpy array
np.set_printoptions(threshold = np.inf)

if True:
    f = open(f'{rep}_parameters.h', 'wt')
    if rep == 'int8_t':
        f.write('#include <inttypes.h>\n')
    # Gives the tensors in what seems to be the correct order, alleluhia!
    ops = interpreter._get_ops_details()
    for op_index, op in enumerate(ops):
        for layer_idx in op['inputs']:
            layer = interpreter._get_tensor_details(layer_idx)
            # Seems to me that some 'layers' are intermediate results not needed yet
            if not re.match(r'sequential', layer['name']) or ';' in layer['name'] or 'Pool' in layer['name']:
                continue
            layer_name = layer["name"].split('/')
            if layer_name[2] == 'Conv2D':
                wtype = 'kernels'
            elif layer_name[2] == 'BiasAdd':
                wtype = 'biases'
            elif layer_name[2] == 'MatMul':
                wtype = 'weights'
            else :
                continue

            layer_shape = re.sub(' +', '][', re.sub('\[ +', '[', f'{layer["shape"]}'))
            t = f'{rep} {layer_name [1]}_{wtype}' + f'{layer_shape} =\n'
            s = np.array2string(interpreter.get_tensor(layer_idx), 80, separator=',')
            s = s.replace('[', '{')
            s = s.replace(']', '}')
            t += s + ';\n'
            f.write(t)
    f.close()

if False:
    all_layers_details = interpreter.get_tensor_details() 
    for layer in all_layers_details:
        print("//Index ", str(layer['index']))
        print("// Name ", layer['name'])
        print("// Shape ", layer['shape'])
        print("// Quantization ", layer['quantization_parameters'])
        print("// Tensor ")
        s = np.array2string(interpreter.get_tensor(layer['index']), 80, separator=',')
        s = s.replace('[', '{')
        s = s.replace(']', '}')
        print(s + ';')

# Change this to test a different image
test_image_index = 2

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
    global test_images

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file), experimental_preserve_all_tensors = True)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
   
    predictions = np.zeros((len(test_image_indices),), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        test_image = test_images[test_image_index]
        test_label = test_labels[test_image_index]
        
        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details['quantization']
            test_image = test_image / input_scale + input_zero_point
        
        test_image = np.expand_dims(test_image, axis=0).astype(input_details['dtype'])
        interpreter.set_tensor(input_details['index'], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])[0]
        
        predictions[i] = output.argmax()

    return predictions

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index):
    global test_labels

    predictions = run_tflite_model(tflite_file, [test_image_index])

    if False:
        plt.imshow(test_images[test_image_index])
        template = rep + " Model \n True:{true}, Predicted:{predict}"
        _ = plt.title(template.format(true= str(test_labels[test_image_index]), predict=str(predictions[0])))
        plt.grid(False)
    else:
        print(f"vrai: {test_labels[test_image_index]}, predit: {predictions}\n")


# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file):
    global test_images
    global test_labels

    test_image_indices = range(test_images.shape[0])
    predictions = run_tflite_model(tflite_file, test_image_indices)

    accuracy = (np.sum(test_labels == predictions) * 100) / len(test_images)

    print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (rep, accuracy, len(test_images)))


if False:
    test_model(f"lenet5_models/model_{rep}.tflite", test_image_index)
    all_layers_details = interpreter.get_tensor_details() 
    for layer in all_layers_details:
        print("//Index ", str(layer['index']))
        print("// Name ", layer['name'])
        print("// Shape ", layer['shape'])
        print("// Quantization ", layer['quantization_parameters'])
        print("// Tensor ")
        s = np.array2string(interpreter.get_tensor(layer['index']), 80, separator=',')
        s = s.replace('[', '{')
        s = s.replace(']', '}')
        print(s + ';')

if False:
    evaluate_model(f"lenet5_models/model_{rep}.tflite")
