#include <stdio.h>

#include "mayo.cuh"

/* Definitions */
int mayo2_sign_signature(unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const unsigned char *csk);

int mayo_expand_sk(const unsigned char *csk, sk_t *sk);


__global__ void decode(const unsigned char *m, unsigned char *mdec, int *mdeclen);

__global__ void unpack_m_vecs(const unsigned char *in,
                                     uint64_t *out,
                                     int *vecs);


__global__ void P1P1t_times_O(const uint64_t* __restrict__ P1,
                                    const unsigned char* __restrict__ O,
                                    uint64_t* __restrict__ acc);
                                     
__device__ __forceinline__ int upper_triangular_index(int r, 
                                                      int c, 
                                                      int param_v);

void mayo_secure_clear(void *mem, 
                       size_t size); 

void printElement(unsigned char* element, int n, const char* title)
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

    float gpu_t; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const unsigned char *seed_sk;

    unsigned char *h_tmp, *d_tmp;//[DIGEST_BYTES_MAX + SALT_BYTES_MAX + SK_SEED_BYTES_MAX + 1];
    unsigned char *d_m;

    cudaMallocHost((void**)&h_tmp,(MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1)*sizeof(unsigned char));

    cudaMalloc((void**)&d_tmp,(MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1)*sizeof(unsigned char));
    cudaMalloc((void**)&d_m, mlen);

    alignas(32) sk_t sk;
    ret = mayo_expand_sk(csk, &sk);

    seed_sk = csk;

    printf("\nmsg\n");
    for (size_t i = 0; i < mlen; i++) {
    printf("0x%02x, ", m[i]);
    }
    printf("\r\n");

    cudaMemcpy(d_m, m,  mlen, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    shake256<<<1,25>>>(d_tmp, MAYO_2_digest_bytes, d_m, mlen);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_t, start, stop);    
    printf("Time elapsed %.6f ms\n", gpu_t);
    cudaMemcpy(h_tmp, d_tmp,  MAYO_2_digest_bytes, cudaMemcpyDeviceToHost);

    printElement(h_tmp, (MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1), "tmp:");

    // printElement(sk.O,
    //          MAYO_2_v * MAYO_2_o,
    //          "O After mayo_expand_sk:");

//   printElement((unsigned char *)sk.p,
//              P1_LIMBS_MAX * sizeof(uint64_t),
//              "P1 after mayo_expand_sk:");

//   printElement((unsigned char *)(sk.p + P1_LIMBS_MAX),
//              P2_LIMBS_MAX * sizeof(uint64_t),
//              "L/P2 after mayo_expand_sk:");
//              printf("%d %d\r\n",P2_LIMBS_MAX, P1_LIMBS_MAX);
    

    return ret;

}

int mayo_expand_sk(const unsigned char *csk, sk_t *sk)
{
    int ret = MAYO_OK;
    

    unsigned char *h_S, *h_seed_sk;
    unsigned char *d_S, *d_seed_sk;

    int blocks = ((((MAYO_2_o * MAYO_2_v) + 1)/2) + THREADS-1)/THREADS;

    unsigned char *h_O = sk->O;
    unsigned char *d_O;
    int *h_mdeclen;
    int *d_mdeclen;

    uint64_t *h_P = sk->p;
    unsigned char *h_seed_pk;
    int *h_vecs;
    uint64_t *d_P;
    int *d_vecs;

    const int m_vec_limbs = MAYO_2_m_vec_limbs;

    const int p1_vecs = MAYO_2_v * (MAYO_2_v + 1) / 2;
    const int p2_vecs = MAYO_2_v * MAYO_2_o;
    const int total_vecs = p1_vecs + p2_vecs;

    const int packed_bytes = MAYO_2_P1_bytes + MAYO_2_P2_bytes;
    const int unpacked_bytes = total_vecs * (MAYO_2_m_vec_limbs * sizeof(uint64_t));

    unsigned char *d_P_bytes;
    uint64_t *d_P_limbs;

    /* First SHAKE 256*/
    cudaMallocHost((void**)&h_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * sizeof(uint8_t));
    cudaMallocHost((void**)&h_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t));

    cudaMalloc((void**)&d_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * sizeof(uint8_t));
    cudaMalloc((void**)&d_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t));

    /* DECODE */
    cudaMallocHost((void**)&h_mdeclen, sizeof(int));
    cudaMalloc((void**)&d_O, MAYO_2_v * MAYO_2_o * sizeof(unsigned char));
    cudaMalloc((void**)&d_mdeclen, sizeof(int));

    /* Expand P1P2 */
    cudaMallocHost((void**)&h_vecs, sizeof(int));
    cudaMalloc((void**)&d_P, (P1_LIMBS_MAX + P2_LIMBS_MAX)*sizeof(uint64_t));
    cudaMalloc((void**)&d_vecs, sizeof(int));


    cudaMalloc((void**)&d_P_bytes, packed_bytes);
    cudaMalloc((void**)&d_P_limbs, unpacked_bytes);

    int threadsPerBlock = THREADS;
    int total_bytes = total_vecs * (MAYO_2_m_vec_limbs * sizeof(uint64_t));



    

    for(int i = 0; i < MAYO_2_sk_seed_bytes; i++)
    {
        h_seed_sk[i] = csk[i];
    }
    cudaMemcpy(d_seed_sk, h_seed_sk, MAYO_2_sk_seed_bytes, cudaMemcpyHostToDevice);

    *h_mdeclen = MAYO_2_v * MAYO_2_o;
    cudaMemcpy(d_mdeclen, h_mdeclen, sizeof(int), cudaMemcpyHostToDevice);
   

    float gpu_t; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    /***************** SHAKE256 **********************/
    shake256<<<1,25>>>(d_S, MAYO_2_pk_seed_bytes + MAYO_2_O_bytes,  d_seed_sk, MAYO_2_sk_seed_bytes);
    cudaMemcpy(h_S, d_S, MAYO_2_pk_seed_bytes + MAYO_2_O_bytes, cudaMemcpyDeviceToHost);
    h_seed_pk = h_S;
    /************************************************/
    
    /***************** DECODE ***********************/
    decode<<<blocks, THREADS>>>(d_S + MAYO_2_pk_seed_bytes, d_O, d_mdeclen);
    cudaMemcpy(h_O, d_O, MAYO_2_v * MAYO_2_o * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    /************************************************ */

    /******************* Expand P1 P2 **************************/
    AES_128_CTR((unsigned char *)h_P,
                packed_bytes,
                h_seed_pk,
                MAYO_2_pk_seed_bytes);

    cudaMemcpy(d_P_bytes,
            (unsigned char *)h_P,
            packed_bytes,
            cudaMemcpyHostToDevice);

    *h_vecs = total_vecs;
    cudaMemcpy(d_vecs, h_vecs, sizeof(int), cudaMemcpyHostToDevice);

    blocks = (total_bytes + threadsPerBlock - 1) / threadsPerBlock;

    unpack_m_vecs<<<blocks, threadsPerBlock>>>(
        d_P_bytes,
        d_P_limbs,
        d_vecs
    );
    cudaDeviceSynchronize();

    /**********************************************************/


    /******************** P1P1t_times_O *************************/
    // compute L_i = (P1 + P1^t)*O + P2
    blocks = ((MAYO_2_v * MAYO_2_o) + 511) / 512;


    P1P1t_times_O<<<blocks, THREADS>>>(
    d_P_limbs,                  /* P1 */
    d_O,
    d_P_limbs + P1_LIMBS_MAX    /* P2 */
    );
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_P,
           d_P_limbs,
           (P1_LIMBS_MAX + P2_LIMBS_MAX) * sizeof(uint64_t),
           cudaMemcpyDeviceToHost);

    /*********************************************************/

    mayo_secure_clear(h_S,MAYO_2_pk_seed_bytes + MAYO_2_O_bytes);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_t, start, stop);    
    printf("Time elapsed %.6f ms\n", gpu_t);




    cudaFree(d_S);
    cudaFree(d_seed_sk);
    cudaFree(d_mdeclen);
    cudaFree(d_O);
    cudaFree(d_P_bytes);
    cudaFree(d_P_limbs);

    cudaFreeHost(h_S);
    cudaFreeHost(h_seed_sk);
    cudaFreeHost(h_mdeclen);
    cudaFreeHost(h_O);

    // cudaFreeHost(h_P2_tmp);

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



__global__ void unpack_m_vecs(const unsigned char *in,
                              uint64_t *out,
                              int *vecs)
{
    int m_vec_limbs = (MAYO_2_m + 15) / 16;

    int in_bytes_per_vec  = MAYO_2_m/ 2;
    int out_bytes_per_vec = m_vec_limbs * sizeof(uint64_t);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int total_bytes = *vecs * out_bytes_per_vec;

    if (tid < total_bytes) 
    {
        int vec_index = tid / out_bytes_per_vec;
        int byte_index = tid % out_bytes_per_vec;

        unsigned char *out_bytes = (unsigned char *)out;

        if (byte_index < in_bytes_per_vec) 
        {
            out_bytes[vec_index * out_bytes_per_vec + byte_index] = in[vec_index * in_bytes_per_vec + byte_index];
        } 
        else 
        {
            out_bytes[vec_index * out_bytes_per_vec + byte_index] = 0;
        }
    }
}

__global__
void P1P1t_times_O(const uint64_t* __restrict__ P1,
                          const unsigned char* __restrict__ O,
                          uint64_t* __restrict__ acc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int total_outputs = MAYO_2_v * MAYO_2_o;

    if (tid >= total_outputs) {
        return;
    }

    int row = tid / MAYO_2_o;
    int k   = tid % MAYO_2_o;

    uint64_t local[M_VEC_LIMBS_MAX];

    for (int i = 0; i < M_VEC_LIMBS_MAX; i++) {
        local[i] = 0;
    }

    for (int col = 0; col < MAYO_2_v; col++) {

        if (col == row) {
            continue;
        }

        int r;
        int c;

        if (row < col) {
            r = row;
            c = col;
        } else {
            r = col;
            c = row;
        }

        int p1_index = upper_triangular_index(r, c, MAYO_2_v);

        const uint64_t* P1_entry = P1 + p1_index * MAYO_2_m_vec_limbs;

        uint8_t a = O[col * MAYO_2_o + k];

        for (int i = 0; i < MAYO_2_m_vec_limbs; i++) {
            local[i] ^= gf16_vec_mul(P1_entry[i], a);
        }
    }

    uint64_t* acc_entry = acc + (row * MAYO_2_o + k) * MAYO_2_m_vec_limbs;

    for (int i = 0; i < MAYO_2_m_vec_limbs; i++) {
        acc_entry[i] ^= local[i];
    }
}

__device__ __forceinline__
int upper_triangular_index(int r, 
                           int c, 
                           int param_v)
{
    return r * param_v - (r * (r - 1)) / 2 + (c - r);
}

void mayo_secure_clear(void *mem, size_t size) 
{
    typedef void *(*memset_t)(void *, int, size_t);
    static volatile memset_t memset_func = memset;
    memset_func(mem, 0, size);
}
