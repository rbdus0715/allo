
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
using namespace std;

static const int ALLO_E4M3_BIAS = 7;
static const int ALLO_E8M0_BIAS = 127;

static inline float allo_pow2_int(int exp) { return ldexp(1.0f, exp); }

static inline float allo_decode_e4m3(uint8_t u8) {
  int sign = (u8 >> 7) & 1;
  int exp = (u8 >> 3) & 15;
  int mant = u8 & 7;
  float val = 0.0f;
  if (exp == 15 && mant == 7) {
    val = 0.0f;
  } else if (exp == 0) {
    val = float(mant) / 8.0f * allo_pow2_int(1 - ALLO_E4M3_BIAS);
  } else {
    val = (1.0f + float(mant) / 8.0f) * allo_pow2_int(exp - ALLO_E4M3_BIAS);
  }
  return sign ? -val : val;
}

static inline uint8_t allo_encode_e4m3(float f) {
  if (f == 0.0f)
    return 0;
  int sign = (f < 0.0f) ? 1 : 0;
  float abs_f = sign ? -f : f;
  int unbiased_exp = -20;
  for (int e = -20; e < 16; ++e) {
    if (abs_f >= allo_pow2_int(e))
      unbiased_exp = e;
  }
  int exp_field = unbiased_exp + ALLO_E4M3_BIAS;
  int mant = 0;
  if (exp_field <= 0) {
    exp_field = 0;
    mant = int(abs_f / allo_pow2_int(1 - ALLO_E4M3_BIAS) * 8.0f + 0.5f);
  } else {
    mant = int((abs_f / allo_pow2_int(unbiased_exp) - 1.0f) * 8.0f + 0.5f);
    if (mant == 8) {
      mant = 0;
      exp_field += 1;
    }
  }
  if (exp_field >= 15) {
    exp_field = 14;
    mant = 7;
  }
  return uint8_t((sign << 7) | (exp_field << 3) | mant);
}

static inline float allo_decode_e8m0(uint8_t u8) {
  if (u8 == 0 || u8 == 255)
    return 0.0f;
  return allo_pow2_int(int(u8) - ALLO_E8M0_BIAS);
}

static inline uint8_t allo_encode_e8m0(float scale) {
  if (scale <= 0.0f)
    return 0;
  int result = 254;
  for (int e = 1; e < 255; ++e) {
    if (result == 254 && allo_pow2_int(e - ALLO_E8M0_BIAS) >= scale)
      result = e;
  }
  return uint8_t(result);
}

static inline void allo_decode_mxfp8_block(int bs, uint8_t scale, uint8_t *data,
                                           float *out) {
  float s = allo_decode_e8m0(scale);
  for (int i = 0; i < bs; ++i)
    out[i] = allo_decode_e4m3(data[i]) * s;
}

static inline void allo_encode_mxfp8_block(int bs, float *data, uint8_t *scale_out,
                                           uint8_t *data_out) {
  float max_val = 0.0f;
  for (int i = 0; i < bs; ++i) {
    float av = data[i] < 0.0f ? -data[i] : data[i];
    if (av > max_val)
      max_val = av;
  }
  scale_out[0] = allo_encode_e8m0(max_val);
  float s = allo_decode_e8m0(scale_out[0]);
  for (int i = 0; i < bs; ++i) {
    float scaled = (s != 0.0f) ? data[i] / s : data[i];
    data_out[i] = allo_encode_e4m3(scaled);
  }
}

static inline void allo_block_add_mxfp8(int bs, uint8_t scale_a, uint8_t *data_a,
                                        uint8_t scale_b, uint8_t *data_b,
                                        uint8_t *scale_out, uint8_t *data_out) {
  float sa = allo_decode_e8m0(scale_a);
  float sb = allo_decode_e8m0(scale_b);
  float buf[32];
  for (int i = 0; i < bs; ++i)
    buf[i] = allo_decode_e4m3(data_a[i]) * sa + allo_decode_e4m3(data_b[i]) * sb;
  allo_encode_mxfp8_block(bs, buf, scale_out, data_out);
}

static inline void allo_block_matmul_mxfp8(int bs, uint8_t scale_a, uint8_t *data_a,
                                           uint8_t scale_b, uint8_t *data_b,
                                           uint8_t *scale_out, uint8_t *data_out) {
  float sa = allo_decode_e8m0(scale_a);
  float sb = allo_decode_e8m0(scale_b);
  float acc = 0.0f;
  for (int i = 0; i < bs; ++i)
    acc += allo_decode_e4m3(data_a[i]) * sa * allo_decode_e4m3(data_b[i]) * sb;
  float out[1] = {acc};
  allo_encode_mxfp8_block(1, out, scale_out, data_out);
}
/// This is top function.
void native_block_add(
  uint8_t v0,
  uint8_t v1[32],
  uint8_t v2,
  uint8_t v3[32],
  uint8_t v4[1],
  uint8_t v5[32]
) {	// L2
  allo_block_add_mxfp8(32, v0, v1, v2, v3, v4, v5);
}

