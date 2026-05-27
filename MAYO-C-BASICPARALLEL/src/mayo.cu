#include <stdio.h>

#include "mayo.cuh"

/* Definitions */
int mayo2_sign_signature(unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const unsigned char *csk);

int mayo_expand_sk(const unsigned char *csk, sk_t *sk);

__global__ void decode(const unsigned char *m, unsigned char *mdec, int *mdeclen);


void printElement(unsigned char* element, int n, char* title)
{
  printf("%s\r\n",title);
  for(int i = 0; i < n; i++)
  {
    printf("%02x, ",element[i]);
  }
  printf("\r\n");
}


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
    alignas(32) sk_t sk; 
    ret = mayo_expand_sk(csk, &sk);
    

    return ret;

}

int mayo_expand_sk(const unsigned char *csk, sk_t *sk)
{
    int ret = MAYO_OK;
    

    unsigned char *h_S, *h_seed_sk;
    unsigned char *d_S, *d_seed_sk;

    unsigned char *h_O = sk->O;
    unsigned char *d_O;
    int *h_mdeclen;
    int *d_mdeclen;



    /* First SHAKE 256*/
    cudaMallocHost((void**)&h_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * sizeof(uint8_t));
    cudaMallocHost((void**)&h_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t));

    cudaMalloc((void**)&d_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * sizeof(uint8_t));
    cudaMalloc((void**)&d_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t));

    /* DECODE */
    cudaMallocHost((void**)&h_mdeclen, sizeof(int));
    cudaMalloc((void**)&d_O, MAYO_2_v * MAYO_2_o * sizeof(unsigned char));
    cudaMalloc((void**)&d_mdeclen, sizeof(int));
    

    for(int i = 0; i < MAYO_2_sk_seed_bytes; i++)
    {
        h_seed_sk[i] = csk[i];
    }
    cudaMemcpy(d_seed_sk, h_seed_sk, MAYO_2_sk_seed_bytes, cudaMemcpyHostToDevice);

    *h_mdeclen = MAYO_2_v * MAYO_2_o;
    cudaMemcpy(d_mdeclen, h_mdeclen, sizeof(int), cudaMemcpyHostToDevice);
   

    
    
    shake256<<<1,25>>>(d_S, MAYO_2_pk_seed_bytes + MAYO_2_O_bytes,  d_seed_sk, MAYO_2_sk_seed_bytes);
    // cudaMemcpy(h_S, d_S, MAYO_2_pk_seed_bytes + MAYO_2_O_bytes, cudaMemcpyDeviceToHost);
    int blocks = ((((MAYO_2_o * MAYO_2_v) + 1)/2) + 255)/256;
    decode<<<blocks, 256>>>(d_S + MAYO_2_pk_seed_bytes, d_O, d_mdeclen);
    printf("After Kernel \r\n");
    cudaMemcpy(h_O, d_O, MAYO_2_v * MAYO_2_o * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printElement(h_O, MAYO_2_v * MAYO_2_o, "O");

    // printf("FIRST SHAKE256:\r\n");
    // for(int i = 0; i < MAYO_2_pk_seed_bytes + MAYO_2_O_bytes; i++)
    // {
    //     printf("%02x, ", h_S[i]);
    // }
    // printf("\r\n");

    cudaFree(d_S);
    cudaFree(d_seed_sk);
    cudaFree(d_mdeclen);
    cudaFree(d_O);

    cudaFreeHost(h_S);
    cudaFreeHost(h_seed_sk);
    cudaFree(h_mdeclen);
    cudaFreeHost(h_O);

    return ret;
}

__global__ void decode(const unsigned char *m, unsigned char *mdec, int *mdeclen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int nBytes = (*mdeclen + 1)/2;

    if (idx < nBytes) {
        unsigned char byte = m[idx];

        int out0 = 2 * idx;
        int out1 = 2 * idx + 1;

        if (out0 < *mdeclen) {
            mdec[out0] = byte & 0x0F;
        }

        if (out1 < *mdeclen) {
            mdec[out1] = byte >> 4;
        }
    }
}


