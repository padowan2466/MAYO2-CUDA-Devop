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

__device__ __forceinline__
uint8_t gf16_mul_scalar(uint8_t a, uint8_t b)
{
    uint8_t r = 0;

    a &= 0x0F;
    b &= 0x0F;

    while (b) {
        if (b & 1) {
            r ^= a;
        }

        b >>= 1;

        uint8_t carry = a & 0x08;
        a <<= 1;

        if (carry) {
            a ^= 0x03;   
        }

        a &= 0x0F;
    }

    return r & 0x0F;
}

__device__ __forceinline__
uint8_t mul_f(uint8_t a, uint8_t b)
{
    return gf16_mul_scalar(a, b);
}

__device__ __forceinline__
void transpose_16x16_nibbles_gpu(uint64_t *A)
{
    uint64_t in[16];
    uint64_t out[16];

    for (int i = 0; i < 16; i++) {
        in[i] = A[i];
        out[i] = 0;
    }

    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            uint64_t nibble = (in[r] >> (4 * c)) & 0xFULL;
            out[c] |= nibble << (4 * r);
        }
    }

    for (int i = 0; i < 16; i++) {
        A[i] = out[i];
    }
}

#endif