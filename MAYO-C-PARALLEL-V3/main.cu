#include <stdio.h>

#include "cuda_kernel.cuh"
#include "inputs.cuh"
#include "parameters.cuh"
#include "mayo.cuh"
#include <stdlib.h>

#define PRINT_KEYS 0



#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

int main()
{
    int res = MAYO_OK;

    unsigned char msg[32] = {0x0e};

    size_t msglen = 32;
    size_t smlen = MAYO_2_sig_bytes + msglen;

    unsigned char *sig = (unsigned char *)calloc(smlen, 1);
    if (sig == NULL) {
        printf("Error: no se pudo reservar sig\n");
        return -1;
    }

    unsigned char seed[16] = {
        0x51, 0x97, 0xcd, 0xad,
        0x64, 0x7e, 0xe6, 0x17,
        0x2a, 0x0a, 0x11, 0x8a,
        0x77, 0x24, 0xd8, 0x6b
    };

    unsigned char *seed_pk = NULL;
    uint64_t *P = NULL;

    const int p1_vecs = MAYO_2_v * (MAYO_2_v + 1) / 2;
    const int p2_vecs = MAYO_2_v * MAYO_2_o;
    const int total_vecs = p1_vecs + p2_vecs;

    const int packed_bytes = MAYO_2_P1_bytes + MAYO_2_P2_bytes;

    const int unpacked_bytes =
        total_vecs * (MAYO_2_m_vec_limbs * sizeof(uint64_t));

    cudaError_t err;

    err = cudaMallocHost((void **)&P, unpacked_bytes);
    if (err != cudaSuccess) {
        printf("Error cudaMallocHost P: %s\n", cudaGetErrorString(err));
        free(sig);
        return -1;
    }

    err = cudaMallocHost((void **)&seed_pk, MAYO_2_pk_seed_bytes);
    if (err != cudaSuccess) {
        printf("Error cudaMallocHost seed_pk: %s\n", cudaGetErrorString(err));
        cudaFreeHost(P);
        free(sig);
        return -1;
    }

    if (MAYO_2_pk_seed_bytes > sizeof(seed)) {
        printf("Error: seed tiene %lu bytes, pero MAYO_2_pk_seed_bytes = %d\n",
               sizeof(seed), MAYO_2_pk_seed_bytes);

        cudaFreeHost(seed_pk);
        cudaFreeHost(P);
        free(sig);
        return -1;
    }

    memcpy(seed_pk, seed, MAYO_2_pk_seed_bytes);

    memset(P, 0, unpacked_bytes);

    AES_128_CTR(
        (unsigned char *)P,
        packed_bytes,
        seed_pk,
        MAYO_2_pk_seed_bytes
    );

    printf("AES_128_CTR finished\n");

    printf("First 32 bytes:\n");
    for (int i = 0; i < packed_bytes && i < packed_bytes; i++) {
        printf("%02x, ", ((unsigned char *)P)[i]);
    }
    printf("\n");


    cudaFreeHost(seed_pk);
    cudaFreeHost(P);
    free(sig);

    return res;
}