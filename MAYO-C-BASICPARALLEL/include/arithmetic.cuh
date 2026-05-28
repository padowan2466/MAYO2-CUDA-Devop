#ifndef SIMPLE_ARITHMETIC_H
#define SIMPLE_ARITHMETIC_H

#include <stdint.h>

static __device__ __forceinline__
uint32_t mul_table_dev(uint8_t b)
{
    uint32_t x = ((uint32_t)b) * 0x08040201u;

    uint32_t high_nibble_mask = 0xf0f0f0f0u;
    uint32_t high_half = x & high_nibble_mask;

    return x ^ (high_half >> 4) ^ (high_half >> 3);
}


static __device__ __forceinline__
uint64_t gf16_vec_mul(uint64_t in, uint8_t a)
{
    uint32_t tab = mul_table_dev(a);

    const uint64_t lsb_mask = 0x1111111111111111ULL;

    return  ( in       & lsb_mask) * ( tab        & 0xff)
          ^ ((in >> 1) & lsb_mask) * ((tab >> 8)  & 0xf)
          ^ ((in >> 2) & lsb_mask) * ((tab >> 16) & 0xf)
          ^ ((in >> 3) & lsb_mask) * ((tab >> 24) & 0xf);
}

#endif