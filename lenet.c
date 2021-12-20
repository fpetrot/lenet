#include <inttypes.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "float_images.h"
#include "float_parameters.h"

/* Dump tensors more or less as tensorflow does */
void dump_tensor(int channels, int tensor_size,
                 double input[tensor_size][tensor_size][channels])
{
#ifdef DUMP_TENSORS
    printf("[");
    for (int i = 0; i < tensor_size; i++) {
        if (i != 0)
            printf(" ");
        printf("[");
        for (int j = 0; j < tensor_size; j++) {
            if (j != 0)
                printf("  ");
            printf("[");
            for (int c = 0; c < channels; c++) {
                printf("%5.5g", input[i][j][c]);
                if (c != channels - 1)
                    printf(" ");
            }
            printf("]");
            if (j != tensor_size - 1) {
                printf("\n");
            }
        }
        printf("]");
        if (i != tensor_size - 1) {
            printf("\n");
        }
    }
    printf("]\n");
#endif
}

/* Dump dense */
void dump_dense(int tensor_size,
                double input[tensor_size])
{
#ifdef DUMP_TENSORS
    printf("[");
    for (int i = 0; i < tensor_size; i++) {
        if (i == 0) {
            printf("[");
        } else if (i % 8 == 0) {
            printf("\n  ");
        } else {
            printf(" ");
        }
        printf("%5.5g", input[i]);

        if (i == tensor_size - 1) {
            printf("]");
        }
    }
    printf("]\n");
#endif
}


static inline double max(double a, double b)
{
    return a > b ? a : b;
}

static inline double relu(double a)
{
    return a < 0 ? 0 : a;
}

/* FIXME: Handle padding as it should */
/* C99 makes my day! */
void conv2d(int in_channels, int out_channels,
            int img_size, int kernel_size,
            double input[img_size][img_size][in_channels],
            double kernel[out_channels][kernel_size][kernel_size][in_channels],
            double bias[out_channels],
            double output[img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                         [img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                         [out_channels])
{
    int fm_size = img_size - 2 * (kernel_size / 2) + !(kernel_size & 1);

    /*
     * We must zero the output values before entering the outer loop as values
     * are supposed to accumulate.
     */
    for (int o = 0; o < out_channels; o++) {
        for (int k = 0; k < fm_size; k++) {
            for (int l = 0; l < fm_size; l++) {
                output[k][l][o] = 0.0;
            }
        }
    }

    for (int o = 0; o < out_channels; o++) {
        for (int i = 0; i < in_channels; i++) {
            for (int k = 0; k < fm_size; k++) {
                for (int l = 0; l < fm_size; l++) {
                    for (int m = 0; m < kernel_size; m++) {
                        for (int n = 0; n < kernel_size; n++) {
                            output[k][l][o] += kernel[o][m][n][i] * input[k + m][l + n][i];
                        }
                    }
                }
            }
        }
    }

    /* Apply bias and relu on all output channels */
    for (int o = 0; o < out_channels; o++) {
        for (int k = 0; k < fm_size; k++) {
            for (int l = 0; l < fm_size; l++) {
                output[k][l][o] = relu(output[k][l][o] + bias[o]);
            }
        }
    }
}

void maxpool(int channels,
             int img_size,
             int stride_size,
             double input[img_size][img_size][channels],
             double output[img_size / stride_size]
                         [img_size / stride_size]
                         [channels])
{
    for (int i = 0; i < channels; i++) {
        for (int j = 0; j < img_size; j += stride_size) {
            for (int k = 0; k < img_size; k += stride_size) {
                double v = -FLT_MAX;
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

void reshape(int channels,
             int img_size,
             int stride_size,
             double input[img_size][img_size][channels],
             double output[(img_size * img_size * channels) / stride_size])
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

void dense(int inputs,
           int outputs,
           double input[inputs],
           double weight[outputs][inputs],
           double bias[outputs],
           double output[outputs])
{
    for (int j = 0; j < outputs; j ++) {
        for (int i = 0; i < inputs; i ++) {
            output[j] += input[i] * weight[j][i];
        }
        output[j] = relu(output[j] + bias[j]);
    }
}


int main(void)
{
    /* Input image 32x32, output image 28x28 */
    double c1_out[28][28][6];
    conv2d(1, 6, 32, 5, test_mnist[0], C1_kernels, C1_biases, c1_out);
#if 0
    dump_tensor(6, 28, c1_out);
    exit(0);
#endif
    double s2_out[14][14][6];
    maxpool(6, 28, 2, c1_out, s2_out);
#if 0
    dump_tensor(6, 14, s2_out);
    exit(0);
#endif
    double c3_out[10][10][16];
    conv2d(6, 16, 14, 5, s2_out, C3_kernels, C3_biases, c3_out);
#if 0
    dump_tensor(16, 10, c3_out);
    exit(0);
#endif
    double s4_out[5][5][16];
    maxpool(16, 10, 2, c3_out, s4_out);
#if 0
    dump_tensor(16, 5, s4_out);
    exit(0);
#endif
    double r_out[400];
    reshape(16, 5, 1, s4_out, r_out);
#if 0
    dump_dense(400, r_out);
    exit(0);
#endif
    double f5_out[120];
    dense(400, 120, r_out, F5_weights, F5_biases, f5_out);
#if 0
    dump_dense(120, f5_out);
    exit(0);
#endif
    double f6_out[84];
    dense(120, 84, f5_out, F6_weights, F6_biases, f6_out);
#if 0
    dump_dense(84, f6_out);
    exit(0);
#endif
    double f7_out[10];
    dense(84, 10, f6_out, F7_weights, F7_biases, f7_out);
#if 0
    dump_dense(10, f7_out);
    exit(0);
#endif

    double v = -FLT_MAX;
    int rank = -1;
    for (int i = 0; i < sizeof(f7_out)/sizeof(*f7_out); i++) {
        printf("%g ", f7_out[i]);
        if (v < f7_out[i]) {
            v = f7_out[i];
            rank = i;
            printf(" hit!");
        }
        printf("\n");
    }
    printf("got a %d\n", rank);
    return 0;
}
