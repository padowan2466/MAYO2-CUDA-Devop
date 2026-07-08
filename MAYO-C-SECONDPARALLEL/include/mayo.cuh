#ifndef MAYO_H
#define MAYO_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#include "fips202.cuh"
// #include "aes_c.h"
#include "arithmetic.cuh"
#include <cuda_runtime.h>

/**
 * Status codes
 */
#define MAYO_OK 0
#define MAYO_ERR 1

#define THREADS (256)


#define BATCH (64)




#define F_TAIL_64                                                              \
  { 8, 0, 2, 8 }

__device__ __constant__ unsigned char d_f_tail_64[4] = {8, 0, 2, 8};

#define F_TAIL_LEN 4

/*
 MAYO2 - PARAMETERS
*/
#define MAYO_2_name "MAYO_2"
#define MAYO_2_n 81
#define MAYO_2_m 64
#define MAYO_2_m_vec_limbs 4
#define MAYO_2_o 17
#define MAYO_2_v (MAYO_2_n - MAYO_2_o)
#define MAYO_2_A_cols (MAYO_2_k * MAYO_2_o + 1)
#define MAYO_2_k 4
#define MAYO_2_q 16
#define MAYO_2_m_bytes 32
#define MAYO_2_O_bytes 544
#define MAYO_2_v_bytes 32
#define MAYO_2_r_bytes 34
#define MAYO_2_P1_bytes 66560
#define MAYO_2_P2_bytes 34816
#define MAYO_2_P3_bytes 4896
#define MAYO_2_csk_bytes 24
#define MAYO_2_cpk_bytes 4912
#define MAYO_2_sig_bytes 186
#define MAYO_2_f_tail F_TAIL_64
#define MAYO_2_f_tail_arr f_tail_64
#define MAYO_2_salt_bytes 24
#define MAYO_2_digest_bytes 32
#define MAYO_2_pk_seed_bytes 16
#define MAYO_2_sk_seed_bytes 24




#define M_VEC_LIMBS_MAX MAYO_2_m_vec_limbs

#define P1_LIMBS_MAX (MAYO_2_v * (MAYO_2_v + 1) / 2 * MAYO_2_m_vec_limbs)
#define P2_LIMBS_MAX (MAYO_2_v * MAYO_2_o * MAYO_2_m_vec_limbs)
#define P3_LIMBS_MAX (MAYO_2_o * (MAYO_2_o + 1) / 2 * MAYO_2_m_vec_limbs)

#define EPK_LIMBS (P1_LIMBS_MAX + P2_LIMBS_MAX + P3_LIMBS_MAX)


typedef struct sk_t {
    uint64_t p[P1_LIMBS_MAX + P2_LIMBS_MAX];
    uint8_t O[MAYO_2_v*MAYO_2_o];
} sk_t;


/* Functions */
int mayo2_sign(unsigned char *sm,
              size_t *smlen, const unsigned char *m,
              size_t mlen, const unsigned char *csk);


#endif