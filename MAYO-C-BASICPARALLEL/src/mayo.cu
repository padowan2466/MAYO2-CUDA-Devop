#include <stdio.h>

#include "mayo.cuh"

/* Definitions */
int mayo2_sign_signature(unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const unsigned char *csk);

int mayo_expand_sk(const unsigned char *csk);



/* Functions */
int mayo2_sign(unsigned char *sm,
              size_t *smlen, const unsigned char *m,
              size_t mlen, const unsigned char *csk)
{
    int ret = MAYO_OK;
    const int param_sig_bytes = MAYO_2_sig_bytes;
    size_t siglen;

    memmove(sm + param_sig_bytes, m, mlen);
    ret = mayo2_sign_signature(sm, &siglen, sm + param_sig_bytes, mlen, csk);

    return ret;
    

}

int mayo2_sign_signature(unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const unsigned char *csk)
{
    int ret = MAYO_OK;
    ret = mayo_expand_sk(csk);
    

    return ret;

}

int mayo_expand_sk(const unsigned char *csk)
{
    int ret = MAYO_OK;
    

    // unsigned char S[MAYO_2_pk_seed_bytes + MAYO_2_O_bytes];
    unsigned char *h_S, *h_seed_sk;
    unsigned char *d_S, *d_seed_sk;



    cudaMallocHost((void**)&h_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * sizeof(uint8_t));
    cudaMallocHost((void**)&h_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t));

    cudaMalloc((void**)&d_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * sizeof(uint8_t));
    cudaMalloc((void**)&d_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t));

    for(int i = 0; i < MAYO_2_sk_seed_bytes; i++)
    {
        h_seed_sk[i] = csk[i];
    }

    cudaMemcpy(d_seed_sk, h_seed_sk, MAYO_2_sk_seed_bytes, cudaMemcpyHostToDevice);
   

    

    shake256<<<1,25>>>(d_S, MAYO_2_pk_seed_bytes + MAYO_2_O_bytes,  d_seed_sk, MAYO_2_sk_seed_bytes);
    cudaMemcpy(h_S, d_S, MAYO_2_pk_seed_bytes + MAYO_2_O_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("FIRST SHAKE256:\r\n");
    for(int i = 0; i < MAYO_2_pk_seed_bytes + MAYO_2_O_bytes; i++)
    {
        printf("%02x, ", h_S[i]);
    }
    printf("\r\n");

    cudaFree(d_S);
    cudaFree(d_seed_sk);

    cudaFreeHost(h_S);
    cudaFreeHost(h_seed_sk);

    return ret;
}


