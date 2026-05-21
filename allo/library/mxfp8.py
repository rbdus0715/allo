import math
import numpy as np

MXFP8_BLOCK_SIZE = 32
E4M3_BIAS = 7
E8M0_BIAS = 127

def decode_e8m0(u8: int) -> float:
    u8 = int(u8) & 0xFF
    if u8 == 0 or u8 == 0xFF:
        return float("nan")
    return float(2.0 ** (u8 - E8M0_BIAS))

def encode_e8m0(scale: float) -> int:
    if scale <= 0:
        return 0
    exp = int(math.ceil(math.log2(scale))) + E8M0_BIAS
    return max(1, min(254, exp))

def decode_e4m3(u8: int) -> float:
    u8 = int(u8) & 0xFF
    sign = (u8 >> 7) & 1
    exp = (u8 >> 3) & 0xF
    mant = u8 & 0x7
    
    if exp == 15 and mant == 7:
        return float("nan")
        
    if exp == 0:
        val = mant / 8.0 * 2.0 ** (1 - E4M3_BIAS)
    else:
        val = (1.0 + mant / 8.0) * 2.0 ** (exp - E4M3_BIAS)
    return -val if sign else val

def encode_e4m3(f: float) -> int:
    if math.isnan(f):
        return 0x7F
    if f == 0:
        return 0
        
    sign = 1 if f < 0 else 0
    abs_f = abs(f)
    
    exp = int(math.floor(math.log2(abs_f))) + E4M3_BIAS
    
    if exp <= 0:
        exp = 0
        mant = int(round((abs_f / 2.0**(1 - E4M3_BIAS)) * 8))
    else:
        mant = int(round((abs_f / 2.0**(exp - E4M3_BIAS) - 1.0) * 8))
        if mant == 8:
            mant = 0
            exp += 1
            
    if exp >= 15:
        exp = 14
        mant = 7
        
    return (sign << 7) | (exp << 3) | mant

def decode_block(scale_u8: int, data: np.ndarray) -> np.ndarray:
    s = decode_e8m0(scale_u8)
    if math.isnan(s):
        return np.full(len(data), np.nan, dtype=np.float32)
    return np.array([decode_e4m3(x) * s for x in data], dtype=np.float32)

def encode_block(data: np.ndarray) -> tuple[int, np.ndarray]:
    max_val = np.max(np.abs(data))
    scale_u8 = encode_e8m0(max_val)
    s = decode_e8m0(scale_u8)
    
    scaled_data = data / s if s != 0 else data
    encoded = np.array([encode_e4m3(x) for x in scaled_data], dtype=np.uint8)
    return scale_u8, encoded

def block_add(scale1: int, data1: np.ndarray, scale2: int, data2: np.ndarray) -> tuple[int, np.ndarray]:
    arr1 = decode_block(scale1, data1)
    arr2 = decode_block(scale2, data2)
    
    arr_sum = arr1 + arr2
    
    return encode_block(arr_sum)
