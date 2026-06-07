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

static __device__ __forceinline__
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

static __device__ __forceinline__
uint8_t mul_f(uint8_t a, uint8_t b)
{
    return gf16_mul_scalar(a, b);
}

static __device__ __forceinline__
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

static __device__ __forceinline__
uint64_t ct_compare_64_device(uint64_t a, uint64_t b)
{
    uint64_t x = a ^ b;

    x |= x >> 32;
    x |= x >> 16;
    x |= x >> 8;
    x |= x >> 4;
    x |= x >> 2;
    x |= x >> 1;

    return (uint64_t)(0ULL - (x & 1ULL));
}

static __device__ __forceinline__
uint8_t ct_compare_8_device(uint8_t a, uint8_t b)
{
    uint8_t x = a ^ b;

    x |= x >> 4;
    x |= x >> 2;
    x |= x >> 1;

    return (uint8_t)(0 - (x & 1));
}

static __device__ __forceinline__
uint64_t ct_64_is_greater_than_device(uint64_t a, uint64_t b)
{
    return (uint64_t)(0ULL - (uint64_t)(a > b));
}

static __device__ __forceinline__
uint8_t inverse_f_device(uint8_t a)
{
    a &= 0x0F;

    if (a == 0) {
        return 0;
    }

    for (uint8_t x = 1; x < 16; x++) {
        if (mul_f(a, x) == 1) {
            return x;
        }
    }

    return 0;
}

static __device__ __forceinline__
uint8_t m_extract_element_device(const uint64_t *row, int idx)
{
    return (row[idx / 16] >> (4 * (idx % 16))) & 0x0F;
}

static __device__ __forceinline__
void vec_mul_add_u64_device(int row_len,
                            const uint64_t *in,
                            uint8_t a,
                            uint64_t *acc)
{
    for (int i = 0; i < row_len; i++) {
        acc[i] ^= gf16_vec_mul(in[i], a);
    }
}

static __device__ __forceinline__
uint64_t mul_fx8_device(uint8_t u, uint64_t x)
{
    return gf16_vec_mul(x, u);
}

static __device__ __forceinline__
void ef_pack_m_vec_device(const unsigned char *in,
                          uint64_t *out,
                          int ncols)
{
    int row_len = (ncols + 15) / 16;

    for (int i = 0; i < row_len; i++) {
        out[i] = 0;
    }

    for (int i = 0; i < ncols; i++) {
        out[i / 16] |= ((uint64_t)(in[i] & 0x0F)) << (4 * (i % 16));
    }
}

static __device__ __forceinline__
void ef_unpack_m_vec_device(int row_len,
                            const uint64_t *in,
                            unsigned char *out)
{
    for (int i = 0; i < row_len * 16; i++) {
        out[i] = (in[i / 16] >> (4 * (i % 16))) & 0x0F;
    }
}

#endif