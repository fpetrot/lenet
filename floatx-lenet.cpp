/*
 * Making float-lenet.c a templated C++ file.
 * Now adding support for other FP formats, in particular those of
 * small size
 *
 * 2020-2024 (c) Frédéric Pétrot <frederic.petrot@univ-grenoble-alpes.fr>
 * SLS Team, TIMA Lab, Grenoble INP/UGA
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <cfloat>
#include "float_images.h"
#include "float_parameters.h"
#include "floatx.hpp"

using namespace std;

typedef flx::floatx<5,10> fp16;
typedef flx::floatx<5,10> bfloat16;
typedef flx::floatx<5,2> fp8e5m2;
typedef flx::floatx<4,3> fp8e4m3;

/* Dump tensors more or less as tensorflow does */
template<typename T, int channels, int tensor_size>
void dump_tensor(T input[tensor_size][tensor_size][channels])
{
#ifdef DUMP_TENSORS
    cout << "[";
    for (int i = 0; i < tensor_size; i++) {
        if (i != 0)
            cout << " ";
        cout << "[";
        for (int j = 0; j < tensor_size; j++) {
            if (j != 0)
                cout << "  ";
            cout << "[";
            for (int c = 0; c < channels; c++) {
                cout << input[i][j][c];
                if (c != channels - 1)
                    cout << " ";
            }
            cout << "]";
            if (j != tensor_size - 1) {
                cout << "\n";
            }
        }
        printf("]");
        if (i != tensor_size - 1) {
            cout << "\n";
        }
    }
    cout << "]\n";
#endif
}

/* Dump dense */
template<typename T, int tensor_size>
void dump_dense(T input)
{
#ifdef DUMP_TENSORS
    cout << "[";
    for (int i = 0; i < tensor_size; i++) {
        if (i == 0) {
            cout << "[";
        } else if (i % 8 == 0) {
            cout << "\n  ";
        } else {
            cout << " ";
        }
        cout << input[i];

        if (i == tensor_size - 1) {
            cout << "]";
        }
    }
    cout << "]\n";
#endif
}


#if 0
template<typename T>
static inline T max(T a, T b)
{
    return a > b ? a : b;
}
#endif

template<typename T>
static inline T relu(T a)
{
    return a < 0 ? 0 : a;
}

/* FIXME: Handle padding as it should */
template<typename T,
         int in_channels, int out_channels, int img_size, int kernel_size>
void conv2d(T input[img_size][img_size][in_channels],
            T kernel[out_channels][kernel_size][kernel_size][in_channels],
            T bias[out_channels],
            T output[img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                    [img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                    [out_channels])
{
    int fm_size = img_size - 2 * (kernel_size / 2) + !(kernel_size & 1);

    for (int o = 0; o < out_channels; o++) {
        for (int k = 0; k < fm_size; k++) {
            for (int l = 0; l < fm_size; l++) {
                float mac = 0;
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        for (int i = 0; i < in_channels; i++) {
                            mac += kernel[o][m][n][i] * input[k + m][l + n][i];
                        }
                    }
                }
                output[k][l][o] = relu(mac + bias[o]);
            }
        }
    }
}

template<typename T, int channels, int img_size, int stride_size>
void maxpool(T input[img_size][img_size][channels],
             T output[img_size / stride_size]
                     [img_size / stride_size]
                     [channels])
{
    for (int i = 0; i < channels; i++) {
        for (int j = 0; j < img_size; j += stride_size) {
            for (int k = 0; k < img_size; k += stride_size) {
                float v = -FLT_MAX;
                for (int m = 0; m < stride_size; m++) {
                    for (int n = 0; n < stride_size; n++) {
                        v = max(v, input[j + m][k + n][i]);
                    }
                }
                output[j / stride_size][k / stride_size][i] = v;
            }
        }
    }
}

template<typename T, int channels, int img_size, int stride_size>
void reshape(T input[img_size][img_size][channels],
             T output[(img_size * img_size * channels) / stride_size])
{
    for (int i = 0; i < img_size; i += stride_size) {
        for (int j = 0; j < img_size; j += stride_size) {
            for (int c = 0; c < channels; c++) {
                output[c + j * channels + i * channels * img_size] =
                    input[i][j][c];
            }
        }
    }
}

template<typename T, int inputs, int outputs>
void dense(T input[inputs],
           T weight[outputs][inputs],
           T bias[outputs],
           T output[outputs])
{
    for (int j = 0; j < outputs; j ++) {
        for (int i = 0; i < inputs; i ++) {
            output[j] += input[i] * weight[j][i];
        }
        output[j] = relu(output[j] + bias[j]);
    }
}


typedef float    fpnum;
#if 0
typedef fp16     fpnum;
typedef bfloat16 fpnum;
typedef fp8e5m2  fpnum;
typedef fp8e4m3  fpnum;
#endif

int main(int argc, char *argv[])
{
    if (argc != 2)
        return -1;
    int i = strtol(argv[1], NULL, 0);

    /* Convert float input into fpnum */
    fpnum fp_test_mnist[32][32][1];

    for (int j = 0; j < 31; j++) {
        for (int k = 0; k < 31; k++) {
            fp_test_mnist[j][k][0] = fpnum(test_mnist[i][j][k][0]);
        }
    }

    /* Input image 32x32, output image 28x28 */
    fpnum c1_out[28][28][6];
    conv2d<fpnum, 1, 6, 32, 5>(fp_test_mnist, C1_kernels, C1_biases, c1_out);
#if 0
    dump_tensor<fpnum, 6, 28>(c1_out);
    exit(0);
#endif
    fpnum s2_out[14][14][6];
    maxpool<fpnum, 6, 28, 2>(c1_out, s2_out);
#if 0
    dump_tensor<fpnum, 6, 14>(s2_out);
    exit(0);
#endif
    fpnum c3_out[10][10][16];
    conv2d<fpnum, 6, 16, 14, 5>(s2_out, C3_kernels, C3_biases, c3_out);
#if 0
    dump_tensor<fpnum, 16, 10>(c3_out);
    exit(0);
#endif
    fpnum s4_out[5][5][16];
    maxpool<fpnum, 16, 10, 2>(c3_out, s4_out);
#if 0
    dump_tensor<fpnum, 16, 5>(s4_out);
    exit(0);
#endif
    fpnum r_out[400];
    reshape<fpnum, 16, 5, 1>(s4_out, r_out);
#if 0
    dump_dense<fpnum, 400>(r_out);
    exit(0);
#endif
    fpnum f5_out[120];
    dense<fpnum, 400, 120>(r_out, F5_weights, F5_biases, f5_out);
#if 0
    dump_dense<fpnum, 120>(f5_out);
    exit(0);
#endif
    fpnum f6_out[84];
    dense<fpnum, 120, 84>(f5_out, F6_weights, F6_biases, f6_out);
#if 0
    dump_dense<fpnum, 84>(f6_out);
    exit(0);
#endif
    fpnum f7_out[10];
    dense<fpnum, 84, 10>(f6_out, F7_weights, F7_biases, f7_out);
#if 0
    dump_dense(10, f7_out);
    exit(0);
#endif

    fpnum v = -FLT_MAX;
    int rank = -1;
    for (size_t i = 0; i < sizeof(f7_out)/sizeof(*f7_out); i++) {
        if (v < f7_out[i]) {
            v = f7_out[i];
            rank = i;
        }
    }
    cout << "got a " << rank << endl;
    return 0;
}
