#!/usr/bin/env python
# Construction d'un tableau C à partir d'images de data-sets
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

if len(sys.argv) != 2:
    print(f'Usage {sys.argv[0]} (float|uint8_t|int8_t)')
    sys.exit(1)

if  sys.argv[1] == "uint8_t" or sys.argv[1] == "float" or sys.argv[1] == "int8_t":
    rep = sys.argv[1]
else:
    print(f'Usage {sys.argv[0]} (float|uint8_t|int8_t)')
    sys.exit(1)

(_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()

# on passe de 28x28 en 32x32
test_images = np.pad(array = test_images, pad_width = 2, mode = 'constant', constant_values = 0)
# ca ajoute deux images vides devant et derrière, ...
# on ne garde que les 8 premières images dans un premier temps
test_images = test_images[0:10000]
# petite vérif qu'on récupère bien ce qu'il faut
# np.set_printoptions(threshold = np.inf)
# print(test_images)
# sys.exit(0)

# s doit être initialisée pour plus tard
s = ''

if rep == 'uint8_t' or rep == 'int8_t':
    s += '#include <inttypes.h>\n'
else:
    test_images = test_images.astype(float)
    # si on veut normaliser entre 0 et 1, mais bon, on ne veut pas
    # test_images = tf.image.convert_image_dtype(test_images, dtype=tf.float32, saturate=False)

s += f'{rep}' + ' test_mnist[][32][32][1] = {'
for l, j in enumerate(test_images):
    s += '\n{' + f'// label: {test_labels[l]}\n'
    for i in j:
        s += '{'
        for v in i:
            if rep == 'uint8_t':
                u = "0x{0:02x}".format(v)
            else:
                u = "{:f}".format(v)
            s += '{' + f'{u}' + '},'
        s += '}\n,'
    s += '},\n'
s += '};\n'

f = open(f'{rep}_images.h', 'wt')
f.write(s)
f.close()

