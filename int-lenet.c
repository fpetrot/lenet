#include <inttypes.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "int8_t_images.h"
#include "int8_t_parameters.h"

/* Dump tensors more or less as tensorflow does */
void dump_tensor(int channels, int tensor_size,
                 int8_t input[tensor_size][tensor_size][channels])
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
                printf("%3d", input[i][j][c]);
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
                int8_t input[tensor_size])
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


static inline int8_t max(int8_t a, int8_t b)
{
    return a > b ? a : b;
}

static inline int8_t relu(int32_t a)
{
    return a < 0 ? 0 : a;
}

/*
 * There are several linear algebra tricks used by tensorflow lite when it
 * targets software, in particular it does not use the zero points in
 * the inner-most loop, but for hw implementation, this is both simple and
 * fast, so let's go for it.
 * Note that by construction tflite guaranties for convolutions that the weights
 * tensors have a zero_point of zero.
 */

/* FIXME: Handle padding as it should */
/* C99 makes my day! */
void conv2d(int in_channels, int out_channels,
            int img_size, int kernel_size,
            int8_t input[img_size][img_size][in_channels],
            int8_t input_zp,
            int8_t kernel[out_channels][kernel_size][kernel_size][in_channels],
            int16_t bias[out_channels],
            int32_8_t m0_s[out_channels],
            int8_t output[img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                         [img_size - 2 * (kernel_size / 2) + !(kernel_size & 1)]
                         [out_channels],
            int8_t output_zp)
{
    int fm_size = img_size - 2 * (kernel_size / 2) + !(kernel_size & 1);

    for (int k = 0; k < fm_size; k++) {
        for (int l = 0; l < fm_size; l++) {
            for (int o = 0; o < out_channels; o++) {
                int32_t accu = 0;
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        for (int i = 0; i < in_channels; i++) {
                            accu += kernel[o][m][n][i]
                                    * (input[k + m][l + n][i] - input_zp);
                        }
                    }
                }
                accu += bias[o];
                accu = output_zp + (accu * m0_s[o].mult) >> m0_s[o].shift;
                output[k][l][o] = relu(accu);
            }
        }
    }
}

void maxpool(int channels,
             int img_size,
             int stride_size,
             int8_t input[img_size][img_size][channels],
             int8_t output[img_size / stride_size]
                          [img_size / stride_size]
                          [channels])
{
    for (int i = 0; i < channels; i++) {
        for (int j = 0; j < img_size; j += stride_size) {
            for (int k = 0; k < img_size; k += stride_size) {
                int8_t v = SCHAR_MIN;
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
             int8_t input[img_size][img_size][channels],
             int8_t output[(img_size * img_size * channels) / stride_size])
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
           int8_t input[inputs],
           int8_t input_zp,
           int8_t weight[outputs][inputs],
           int16_t bias[outputs],
           int32_8_t m0_s,
           int8_t output[outputs],
           int8_t output_zp)
{
    for (int j = 0; j < outputs; j ++) {
        int32_t accu = 0;
        for (int i = 0; i < inputs; i ++) {
            accu += (input[i] - input_zp) * weight[j][i];
        }
        accu += bias[j];
        accu = output_zp + (accu * m0_s.mult) >> m0_s.shift;
        accu += output_zp;
        output[j] = relu(accu);
    }
}

int main(void)
{
    int8_t c1_in[32][32];
    /* Input image 32x32, output image 28x28 */
    int8_t c1_out[28][28][6];
    conv2d(1, 6, 32, 5,
           test_mnist[0], C1_zero_points_in[0],
           C1_kernels, C1_biases, C1_m0_s,
           c1_out, C1_zero_points_out[0]);
#if 0
    dump_tensor(6, 28, c1_out);
    exit(0);
#endif
    int8_t s2_out[14][14][6];
    maxpool(6, 28, 2, c1_out, s2_out);
#if 0
    dump_tensor(6, 14, s2_out);
    exit(0);
#endif
    int8_t c3_out[10][10][16];
    conv2d(6, 16, 14, 5, s2_out, C3_zero_points_in[0],
           C3_kernels, C3_biases, C3_m0_s,
           c3_out, C3_zero_points_out[0]);
#if 0
    dump_tensor(16, 10, c3_out);
    exit(0);
#endif
    int8_t s4_out[5][5][16];
    maxpool(16, 10, 2, c3_out, s4_out);
#if 0
    dump_tensor(16, 5, s4_out);
    exit(0);
#endif
    int8_t r_out[400];
    reshape(16, 5, 1, s4_out, r_out);
#if 0
    dump_dense(400, r_out);
    exit(0);
#endif
    int8_t f5_out[120];
    dense(400, 120,
          r_out, F5_zero_points_in[0],
          F5_weights, F5_biases, F5_m0_s[0],
          f5_out, F5_zero_points_out[0]);
#if 0
    dump_dense(120, f5_out);
    exit(0);
#endif
    int8_t f6_out[84];
    dense(120, 84,
          f5_out, F6_zero_points_in[0],
          F6_weights, F6_biases, F6_m0_s[0],
          f6_out, F6_zero_points_out[0]);
#if 0
    dump_dense(84, f6_out);
    exit(0);
#endif
    int8_t f7_out[10];
    dense(84, 10,
          f6_out, F7_zero_points_in[0],
          F7_weights, F7_biases, F7_m0_s[0],
          f7_out, F7_zero_points_out[0]);
#if 0
    dump_dense(10, f7_out);
    exit(0);
#endif

    int8_t v = SCHAR_MIN;
    int rank = -1;
    for (int i = 0; i < sizeof(f7_out)/sizeof(*f7_out); i++) {
        printf("%d ", f7_out[i]);
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
