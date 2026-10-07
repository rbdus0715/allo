"""Match kernels with respective schedules."""

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .systolic import (
    systolic,
    packed_systolic,
    packed_int8xint8_systolic,
    schedule_systolic,
)

from .gemv import (
    int8xint8_mat_vec,
    schedule_int8xint8_mat_vec,
)

from .nn import (
    linear2d,
    linear3d,
    schedule_linear2d,
    schedule_linear3d,
    mx_matmul,
    schedule_mx_matmul,
    mx_gemm_q,
    mx_gemm_wq,
    mx_gemm_ff,
    mx_matmul_q,
    mx_matmul_wq,
    mx_matmul_ff,
    schedule_mx_gemm,
    mx_linear2d,
    schedule_mx_linear2d,
    mx_linear3d,
    schedule_mx_linear3d,
    relu2d,
    relu4d,
    schedule_relu2d,
    schedule_relu4d,
    softmax,
    schedule_softmax,
    layer_norm,
    schedule_layernorm,
    GeLU,
    schedule_gelu,
    conv2d,
    schedule_conv2d,
    maxpool2d,
    schedule_maxpool2d,
    avgpool2d,
    schedule_avgpool2d,
    batchnorm2d,
    schedule_batchnorm2d,
    relu3d,
    schedule_relu3d,
    repeat_batch3d,
    schedule_repeat_batch3d,
    batchnorm1d_2d,
    schedule_batchnorm1d_2d,
    batchnorm1d_3d,
    schedule_batchnorm1d_3d,
    log_softmax,
    schedule_log_softmax,
    concat,
    schedule_concat,
)

KERNEL2SCHEDULE = {}

KERNEL2SCHEDULE.update(
    {
        systolic: schedule_systolic,
        packed_systolic: schedule_systolic,
        packed_int8xint8_systolic: schedule_systolic,
    }
)

KERNEL2SCHEDULE[int8xint8_mat_vec] = schedule_int8xint8_mat_vec

KERNEL2SCHEDULE.update(
    {
        linear2d: schedule_linear2d,
        linear3d: schedule_linear3d,
        relu2d: schedule_relu2d,
        relu4d: schedule_relu4d,
        softmax: schedule_softmax,
        layer_norm: schedule_layernorm,
        GeLU: schedule_gelu,
        conv2d: schedule_conv2d,
        maxpool2d: schedule_maxpool2d,
        avgpool2d: schedule_avgpool2d,
        batchnorm2d: schedule_batchnorm2d,
        relu3d: schedule_relu3d,
        repeat_batch3d: schedule_repeat_batch3d,
        batchnorm1d_2d: schedule_batchnorm1d_2d,
        batchnorm1d_3d: schedule_batchnorm1d_3d,
        log_softmax: schedule_log_softmax,
        concat: schedule_concat,
        mx_matmul: schedule_mx_matmul,
        mx_gemm_q: schedule_mx_gemm,
        mx_gemm_wq: schedule_mx_gemm,
        mx_gemm_ff: schedule_mx_gemm,
        mx_matmul_q: schedule_mx_gemm,
        mx_matmul_wq: schedule_mx_gemm,
        mx_matmul_ff: schedule_mx_gemm,
    }
)

KERNEL2SCHEDULE.update(
    {
        mx_matmul: schedule_mx_matmul,
        mx_linear2d: schedule_mx_linear2d,
        mx_linear3d: schedule_mx_linear3d,
    }
)
