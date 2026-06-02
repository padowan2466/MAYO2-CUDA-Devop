#include <stdio.h>

#include "mayo.cuh"

/* Definitions */
int mayo2_sign_signature(unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const unsigned char *csk);

int mayo_expand_sk(const unsigned char *csk, sk_t *sk);


__global__ void decode(const unsigned char *m,
                       unsigned char *mdec,
                       int mdeclen,
                       int blocks_per_sig,
                       int input_stride,
                       int output_stride);

__global__ void unpack_m_vecs(const unsigned char *in,
                              uint64_t *out,
                              int total_vecs);


__global__ void P1P1t_times_O(const uint64_t* __restrict__ P1,
                                    const unsigned char* __restrict__ O,
                                    uint64_t* __restrict__ acc);
                                     
__device__ __forceinline__ int upper_triangular_index(int r, 
                                                      int c, 
                                                      int param_v);

void mayo_secure_clear(void *mem, 
                       size_t size); 


__global__
void mul_add_mat_x_m_mat_batch_kernel(int m_vec_limbs,
                                      const unsigned char *mat,   
                                      const uint64_t *bs_mat,
                                      uint64_t *acc,
                                      int mat_rows,              
                                      int mat_cols,              
                                      int bs_mat_cols,            
                                      int mat_stride,             
                                      int bs_mat_stride,          
                                      int acc_stride              
);

void printElement(unsigned char* element, int n, const char* title)
{
  printf("\r\n");
  printf("%s\r\n",title);
  for(int i = 0; i < n; i++)
  {
    printf("%02x, ",element[i]);
  }
  printf("\r\n\n");
}

void printBatch(unsigned char* element, int n, int nBatch, const char* title)
{
    for(int i = 0;  i<nBatch;  i++)
    {
        printf("Batch %d\r\n", i);
        printElement(element + i*n, n, title);
    }
}

void printFirstAndLast(unsigned char* element, int n, int nBatch, const char* title)
{   
    printf("%s\r\n", title);
    for(int i = 0;  i<nBatch;  i++)
    {
        printf("Batch %d\r\n", i);
        printf("First: %02x\r\n",*(element + i*n));
        printf("First: %02x\r\n",*(element + (i+1)*n - 1));
    }
}

void print_sk_batch_first_last(sk_t *sk, int nBatch)
{
    for (int i = 0; i < nBatch; i++)
    {
        printf("Batch %d\r\n", i);

        printf("O first: %02x\r\n", sk[i].O[0]);
        printf("O last : %02x\r\n", sk[i].O[MAYO_2_v * MAYO_2_o - 1]);

        printf("p first: %02llx\r\n", (unsigned long long)sk[i].p[0]);
        printf("p last : %02llx\r\n",
               (unsigned long long)sk[i].p[P1_LIMBS_MAX + P2_LIMBS_MAX - 1]);

        printf("\r\n");
    }
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

    unsigned char *h_msg;
    unsigned char *h_tmp, *d_tmp;//[DIGEST_BYTES_MAX + SALT_BYTES_MAX + SK_SEED_BYTES_MAX + 1];
    unsigned char *d_m;

    unsigned char *h_salt, *d_salt;//[SALT_BYTES_MAX];

    unsigned char *ctrbyte;

    unsigned char *h_tenc, *d_tenc;

    unsigned char *h_t, *d_t;

    unsigned char *h_V, *d_V; //[K_MAX * V_BYTES_MAX + R_BYTES_MAX]

    unsigned char *h_digest;

    unsigned char *d_Vdec, *h_Vdec;

    uint64_t *d_VL, *h_VL;

    uint64_t *h_P_batch, *d_P_batch;
    int p1_vecs = MAYO_2_v * (MAYO_2_v + 1) / 2;
    int p2_vecs = MAYO_2_v * MAYO_2_o;
    int p1_limbs = p1_vecs * MAYO_2_m_vec_limbs;
    int p2_limbs = p2_vecs * MAYO_2_m_vec_limbs;
    int p_limbs_per_sig = p1_limbs + p2_limbs;

    cudaMallocHost((void**)&h_msg, mlen*BATCH);
    cudaMallocHost((void**)&h_tmp,(MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1)*BATCH);
    cudaMallocHost((void**)&h_salt, (MAYO_2_salt_bytes)*BATCH);
    cudaMallocHost((void**)&h_tenc, (MAYO_2_m_bytes)*BATCH);
    cudaMallocHost((void**)&h_t, (MAYO_2_m)*BATCH);
    cudaMallocHost((void**)&h_V, (MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes)*BATCH);
    cudaMallocHost((void**)&h_digest, MAYO_2_digest_bytes * BATCH);
    cudaMallocHost((void**)&h_Vdec, MAYO_2_v*MAYO_2_k*BATCH);
    cudaMallocHost((void**)&h_VL, MAYO_2_k * MAYO_2_o * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);
    cudaMallocHost((void **)&h_P_batch, p_limbs_per_sig * sizeof(uint64_t) * BATCH);


    cudaMalloc((void**)&d_tmp,(MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1)*BATCH);
    cudaMalloc((void**)&d_m, mlen*BATCH);
    cudaMalloc((void**)&d_salt, MAYO_2_salt_bytes*BATCH);
    cudaMalloc((void**)&d_tenc, MAYO_2_m_bytes*BATCH);
    cudaMalloc((void**)&d_t, MAYO_2_m*BATCH);
    cudaMalloc((void**)&d_V, (MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes)*BATCH);
    cudaMalloc((void**)&d_Vdec, MAYO_2_v*MAYO_2_k*BATCH);
    cudaMalloc((void**)&d_VL, MAYO_2_k * MAYO_2_o * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);
    cudaMalloc((void **)&d_P_batch, p_limbs_per_sig * sizeof(uint64_t) * BATCH);


    alignas(32) sk_t sk[BATCH];
    ret = mayo_expand_sk(csk, sk);
    // print_sk_batch_first_last(sk, BATCH);

    for (int i = 0; i < BATCH; i++)
    {
        memcpy(h_P_batch + i * p_limbs_per_sig,
               sk[i].p,
               p_limbs_per_sig * sizeof(uint64_t));
    }

    cudaMemcpy(d_P_batch,
               h_P_batch,
               p_limbs_per_sig * sizeof(uint64_t) * BATCH,
               cudaMemcpyHostToDevice);

    uint64_t *d_L = d_P_batch + p1_limbs;



    // seed_sk = csk;

    printf("\nmsg\n");
    for (size_t i = 0; i < mlen; i++) {
    printf("0x%02x, ", m[i]);
    }
    printf("\r\n");


    for(int i=0;i<BATCH;i++)
    {
        for(int j=0;j<mlen;j++) h_msg[j + i*mlen] = m[j];
    }

    // printBatch(h_msg, mlen, BATCH, "MSG:");

    cudaMemcpy(d_m, h_msg,  mlen*BATCH, cudaMemcpyHostToDevice);

    // cudaEventRecord(start);
    shake256<<<BATCH,25>>>(d_tmp, MAYO_2_digest_bytes, d_m, mlen, 0);


    cudaMemcpy(h_digest,
            d_tmp,
            MAYO_2_digest_bytes * BATCH,
            cudaMemcpyDeviceToHost);

    int tmp_bytes =
        MAYO_2_digest_bytes +
        MAYO_2_salt_bytes +
        MAYO_2_sk_seed_bytes +
        1;

    for (int j = 0; j < BATCH; j++)
    {
        unsigned char *tmp_j = h_tmp + j * tmp_bytes;

        memcpy(tmp_j,
            h_digest + j * MAYO_2_digest_bytes,
            MAYO_2_digest_bytes);

        for (int i = 0; i < MAYO_2_salt_bytes; i++)
        {
            tmp_j[MAYO_2_digest_bytes + i] = 1;
        }

        memcpy(tmp_j + MAYO_2_digest_bytes + MAYO_2_salt_bytes,
            csk,
            MAYO_2_sk_seed_bytes);

        tmp_j[MAYO_2_digest_bytes +
            MAYO_2_salt_bytes +
            MAYO_2_sk_seed_bytes] = 0;
    }

    printBatch(h_tmp, tmp_bytes, BATCH, "tmp:");

    cudaMemcpy(d_tmp, h_tmp, (MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1)*BATCH, cudaMemcpyHostToDevice);
 
    shake256<<<BATCH, 25>>>(d_salt, MAYO_2_salt_bytes, d_tmp, MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes,
                            MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1);

    cudaMemcpy(h_salt, d_salt, MAYO_2_salt_bytes*BATCH, cudaMemcpyDeviceToHost);
    // printBatch(h_salt, MAYO_2_salt_bytes, BATCH, "Salt:");

    

    for (int i = 0; i < BATCH; i++)
    {
        unsigned char *tmp_i = h_tmp + i * tmp_bytes;
        unsigned char *salt_i = h_salt + i * MAYO_2_salt_bytes;

        memcpy(tmp_i + MAYO_2_digest_bytes,
            salt_i,
            MAYO_2_salt_bytes);
    }

    // printBatch(h_tmp, tmp_bytes, BATCH, "TMP:");
    cudaMemcpy(d_tmp, h_tmp, tmp_bytes*BATCH, cudaMemcpyHostToDevice);

    shake256<<<BATCH, 25>>>(d_tenc, MAYO_2_m_bytes, d_tmp, MAYO_2_digest_bytes + MAYO_2_salt_bytes, tmp_bytes);

    // cudaMemcpy(h_tenc, d_tenc, MAYO_2_m_bytes*BATCH, cudaMemcpyDeviceToHost);
    // printBatch(h_tenc, MAYO_2_m_bytes, BATCH, "Tenc:");
    int blocks = (((MAYO_2_m  + 1)/2) + THREADS-1)/THREADS;
    decode<<<blocks*BATCH, THREADS>>>(d_tenc, d_t, MAYO_2_m, blocks,MAYO_2_m_bytes, MAYO_2_m);
    cudaMemcpy(h_t, d_t, MAYO_2_m*BATCH, cudaMemcpyDeviceToHost);
    // printBatch(h_t, MAYO_2_m, BATCH, "T:");

    shake256<<<BATCH, 25>>>(d_V, MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes, d_tmp,
             tmp_bytes, 0);

    cudaMemcpy(h_V, d_V, (MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes)*BATCH, cudaMemcpyDeviceToHost);
    // printBatch(h_V, MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes, BATCH, "V:");

    int V_blocks = (((MAYO_2_v + 1) / 2) + THREADS - 1) / THREADS;

    int V_bytes_per_batch = MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes;
    int Vdec_per_batch    = MAYO_2_k * MAYO_2_v;

    for (int i = 0; i < MAYO_2_k; i++)
    {
        decode<<<V_blocks * BATCH, THREADS>>>(
            d_V + i * MAYO_2_v_bytes,
            d_Vdec + i * MAYO_2_v,
            MAYO_2_v,
            V_blocks,
            V_bytes_per_batch,
            Vdec_per_batch
        );
    }

    cudaMemcpy(h_Vdec,
           d_Vdec,
           Vdec_per_batch * BATCH,
           cudaMemcpyDeviceToHost);

    // printBatch(h_Vdec, Vdec_per_batch, BATCH, "Vdec:");

    cudaMemset(d_VL,
           0,
           MAYO_2_k * MAYO_2_o * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);

    int VL_limbs_per_batch = MAYO_2_k * MAYO_2_o * MAYO_2_m_vec_limbs;

    int mat_rows = MAYO_2_k;
    int mat_cols = MAYO_2_v;
    int bs_mat_cols = MAYO_2_o;

    int mat_stride = MAYO_2_k * MAYO_2_v;
    int bs_mat_stride = p_limbs_per_sig;
    int acc_stride = VL_limbs_per_batch;

    int total_VL = BATCH * mat_rows * bs_mat_cols * MAYO_2_m_vec_limbs;
    int blocks_VL = (total_VL + THREADS - 1) / THREADS;

    mul_add_mat_x_m_mat_batch_kernel<<<blocks_VL, THREADS>>>(
        MAYO_2_m_vec_limbs,
        d_Vdec,
        d_L,
        d_VL,
        mat_rows,
        mat_cols,
        bs_mat_cols,
        mat_stride,
        bs_mat_stride,
        acc_stride
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_VL, d_VL, VL_limbs_per_batch * sizeof(uint64_t) * BATCH, cudaMemcpyDeviceToHost);

    printBatch((unsigned char *)h_VL, VL_limbs_per_batch * sizeof(uint64_t), BATCH, "VL:");

    

    cudaFreeHost(h_msg);
    cudaFreeHost(h_tmp);
    cudaFreeHost(h_salt);
    cudaFreeHost(h_tenc);
    cudaFreeHost(h_t);
    cudaFreeHost(h_V);
    cudaFreeHost(h_digest);
    cudaFreeHost(h_Vdec);
    cudaFreeHost(h_VL);
    cudaFreeHost(h_P_batch);

    cudaFree(d_tmp);
    cudaFree(d_m);
    cudaFree(d_salt);
    cudaFree(d_tenc);
    cudaFree(d_t);
    cudaFree(d_V);
    cudaFree(d_Vdec);
    cudaFree(d_VL);
    cudaFree(d_P_batch);


    return ret;

}

int mayo_expand_sk(const unsigned char *csk, sk_t *sk)
{
    int ret = MAYO_OK;
    

    unsigned char *h_S, *h_seed_sk;
    unsigned char *d_S, *d_seed_sk;


    unsigned char *h_O; //= sk->O;
    unsigned char *d_O;
    int mdeclen;

    uint64_t *h_P; //= sk->p;

    const int m_vec_limbs = MAYO_2_m_vec_limbs;

    const int p1_vecs = MAYO_2_v * (MAYO_2_v + 1) / 2;
    const int p2_vecs = MAYO_2_v * MAYO_2_o;
    const int total_vecs = p1_vecs + p2_vecs;

    const int packed_bytes = MAYO_2_P1_bytes + MAYO_2_P2_bytes;
    const int unpacked_bytes = total_vecs * (MAYO_2_m_vec_limbs * sizeof(uint64_t));

    unsigned char *d_P_bytes;
    uint64_t *d_P_limbs;

    /* First SHAKE 256*/
    cudaMallocHost((void**)&h_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * sizeof(uint8_t)*BATCH);
    cudaMallocHost((void**)&h_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t)*BATCH);

    cudaMalloc((void**)&d_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * sizeof(uint8_t)*BATCH);
    cudaMalloc((void**)&d_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t)*BATCH);

    /* DECODE */
    cudaMallocHost((void**)&h_O, MAYO_2_v * MAYO_2_o * sizeof(unsigned char)*BATCH);
    cudaMalloc((void**)&d_O, MAYO_2_v * MAYO_2_o * sizeof(unsigned char)*BATCH);

    /* Expand P1P2 */
    cudaMallocHost((void**)&h_P, unpacked_bytes * BATCH);



    cudaMalloc((void**)&d_P_bytes, packed_bytes*BATCH);
    cudaMalloc((void**)&d_P_limbs, unpacked_bytes*BATCH);

    int threadsPerBlock = THREADS;



    
    for(int j = 0; j< BATCH; j++)
    {
        for(int i = 0; i < MAYO_2_sk_seed_bytes; i++)
        {
            h_seed_sk[i+ j*MAYO_2_sk_seed_bytes] = csk[i];
        }
    }
    cudaMemcpy(d_seed_sk, h_seed_sk, MAYO_2_sk_seed_bytes*BATCH, cudaMemcpyHostToDevice);

    mdeclen = MAYO_2_v * MAYO_2_o;
   

    float gpu_t; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    /***************** SHAKE256 **********************/
    shake256<<<BATCH,25>>>(d_S, MAYO_2_pk_seed_bytes + MAYO_2_O_bytes,  d_seed_sk, MAYO_2_sk_seed_bytes, 0);
    cudaMemcpy(h_S, d_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes)*BATCH, cudaMemcpyDeviceToHost);
    /************************************************/
    
    /***************** DECODE ***********************/
    int in_bytes_per_sig = (mdeclen + 1) / 2;
    int blocks_per_sig = (in_bytes_per_sig + THREADS - 1) / THREADS;
    decode<<<blocks_per_sig * BATCH, THREADS>>>(
        d_S + MAYO_2_pk_seed_bytes,
        d_O,
        mdeclen,
        blocks_per_sig,
        MAYO_2_pk_seed_bytes + MAYO_2_O_bytes,
        mdeclen
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_O,
            d_O,
            MAYO_2_v * MAYO_2_o * sizeof(unsigned char) * BATCH,
            cudaMemcpyDeviceToHost);
    // /************************************************ */

    // /******************* Expand P1 P2 **************************/
    for (int i = 0; i < BATCH; i++)
    {
        AES_128_CTR((unsigned char *)h_P + i * packed_bytes,
                    packed_bytes,
                    h_S + i * (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes),
                    MAYO_2_pk_seed_bytes);
    }
    // printBatch((unsigned char*)h_P, packed_bytes, BATCH, "P:");
    // printFirstAndLast((unsigned char*)h_P, packed_bytes, BATCH, "P:");

    cudaMemcpy(d_P_bytes,
           (unsigned char *)h_P,
           packed_bytes * BATCH,
           cudaMemcpyHostToDevice);

    int total_bytes_all =
        BATCH * total_vecs * MAYO_2_m_vec_limbs * sizeof(uint64_t);

    int blocks_unpack =
        (total_bytes_all + threadsPerBlock - 1) / threadsPerBlock;

    unpack_m_vecs<<<blocks_unpack, threadsPerBlock>>>(
        d_P_bytes,
        d_P_limbs,
        total_vecs
    );
    // cudaMemcpy(h_P_limbs, d_P_limbs, unpacked_bytes*BATCH, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // printFirstAndLast((unsigned char *)h_P_limbs, unpacked_bytes, BATCH, "P_limbs:");


    // /**********************************************************/


    // /******************** P1P1t_times_O *************************/
    // compute L_i = (P1 + P1^t)*O + P2
    int total_outputs = BATCH * MAYO_2_v * MAYO_2_o;

    int blocks_P1P1t =
        (total_outputs + THREADS - 1) / THREADS;


    int p1_limbs = p1_vecs * MAYO_2_m_vec_limbs;

    P1P1t_times_O<<<blocks_P1P1t, THREADS>>>(
        d_P_limbs,
        d_O,
        d_P_limbs + p1_limbs
    );

    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_P,
           d_P_limbs,
           unpacked_bytes * BATCH,
           cudaMemcpyDeviceToHost);
    

    // /*********************************************************/

    mayo_secure_clear(h_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * BATCH);

    for (int i = 0; i < BATCH; i++)
    {
        memcpy(sk[i].O,
            h_O + i * (MAYO_2_v * MAYO_2_o),
            MAYO_2_v * MAYO_2_o * sizeof(unsigned char));

        memcpy(sk[i].p,
            (unsigned char *)h_P + i * unpacked_bytes,
            unpacked_bytes);
    }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&gpu_t, start, stop);    
    // printf("Time elapsed %.6f ms\n", gpu_t);




    cudaFree(d_S);
    cudaFree(d_seed_sk);
    cudaFree(d_O);
    cudaFree(d_P_bytes);
    cudaFree(d_P_limbs);

    cudaFreeHost(h_S);
    cudaFreeHost(h_seed_sk);
    cudaFreeHost(h_O);
    cudaFreeHost(h_P);

    // cudaFreeHost(h_P2_tmp);

    return ret;
}

__global__ void decode(const unsigned char *m,
                       unsigned char *mdec,
                       int mdeclen,
                       int blocks_per_sig,
                       int input_stride,
                       int output_stride)
{
    int sig_id = blockIdx.x / blocks_per_sig;

    int local_block = blockIdx.x % blocks_per_sig;
    int local_idx = local_block * blockDim.x + threadIdx.x;

    int packed_bytes = (mdeclen + 1) / 2;

    if (local_idx >= packed_bytes) {
        return;
    }

    const unsigned char *m_sig =
        m + sig_id * input_stride;

    unsigned char *mdec_sig =
        mdec + sig_id * output_stride;

    unsigned char byte = m_sig[local_idx];

    int out0 = 2 * local_idx;
    int out1 = 2 * local_idx + 1;

    if (out0 < mdeclen) {
        mdec_sig[out0] = byte & 0x0F;
    }

    if (out1 < mdeclen) {
        mdec_sig[out1] = byte >> 4;
    }
}



__global__ void unpack_m_vecs(const unsigned char *in,
                              uint64_t *out,
                              int total_vecs)
{
    int m_vec_limbs = (MAYO_2_m + 15) / 16;

    int in_bytes_per_vec  = MAYO_2_m / 2;
    int out_bytes_per_vec = m_vec_limbs * sizeof(uint64_t);

    int in_bytes_per_sig  = total_vecs * in_bytes_per_vec;
    int out_bytes_per_sig = total_vecs * out_bytes_per_vec;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int total_bytes = BATCH * out_bytes_per_sig;

    if (tid >= total_bytes) {
        return;
    }

    int sig_id = tid / out_bytes_per_sig;
    int local_tid = tid % out_bytes_per_sig;

    int vec_index = local_tid / out_bytes_per_vec;
    int byte_index = local_tid % out_bytes_per_vec;

    unsigned char *out_bytes = (unsigned char *)out;

    if (byte_index < in_bytes_per_vec) {
        out_bytes[sig_id * out_bytes_per_sig +
                  vec_index * out_bytes_per_vec +
                  byte_index] =
            in[sig_id * in_bytes_per_sig +
               vec_index * in_bytes_per_vec +
               byte_index];
    } else {
        out_bytes[sig_id * out_bytes_per_sig +
                  vec_index * out_bytes_per_vec +
                  byte_index] = 0;
    }
}

__global__
void P1P1t_times_O(const uint64_t* __restrict__ P1,
                   const unsigned char* __restrict__ O,
                   uint64_t* __restrict__ acc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int outputs_per_sig = MAYO_2_v * MAYO_2_o;
    int total_outputs = BATCH * outputs_per_sig;

    if (tid >= total_outputs) {
        return;
    }

    int sig_id = tid / outputs_per_sig;
    int local_tid = tid % outputs_per_sig;

    int row = local_tid / MAYO_2_o;
    int k   = local_tid % MAYO_2_o;

    int p_limbs_per_sig = P1_LIMBS_MAX + P2_LIMBS_MAX;
    int o_bytes_per_sig = MAYO_2_v * MAYO_2_o;

    const uint64_t *P1_sig =
        P1 + sig_id * p_limbs_per_sig;

    const unsigned char *O_sig =
        O + sig_id * o_bytes_per_sig;

    uint64_t *acc_sig =
        acc + sig_id * p_limbs_per_sig;

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

        const uint64_t *P1_entry =
            P1_sig + p1_index * MAYO_2_m_vec_limbs;

        uint8_t a = O_sig[col * MAYO_2_o + k];

        for (int i = 0; i < MAYO_2_m_vec_limbs; i++) {
            local[i] ^= gf16_vec_mul(P1_entry[i], a);
        }
    }

    uint64_t *acc_entry =
        acc_sig + (row * MAYO_2_o + k) * MAYO_2_m_vec_limbs;

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

__global__
void mul_add_mat_x_m_mat_batch_kernel(int m_vec_limbs,
                                      const unsigned char *mat,   
                                      const uint64_t *bs_mat,
                                      uint64_t *acc,
                                      int mat_rows,              
                                      int mat_cols,              
                                      int bs_mat_cols,            
                                      int mat_stride,             
                                      int bs_mat_stride,          
                                      int acc_stride              
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int outputs_per_batch = mat_rows * bs_mat_cols * m_vec_limbs;
    int total = BATCH * outputs_per_batch;

    if (tid >= total) {
        return;
    }

    int batch = tid / outputs_per_batch;
    int local = tid % outputs_per_batch;

    int limb = local % m_vec_limbs;

    int tmp = local / m_vec_limbs;
    int k = tmp % bs_mat_cols;
    int r = tmp / bs_mat_cols;

    const unsigned char *mat_b =
        mat + batch * mat_stride;

    const uint64_t *bs_mat_b =
        bs_mat + batch * bs_mat_stride;

    uint64_t *acc_b =
        acc + batch * acc_stride;

    uint64_t sum =
        acc_b[m_vec_limbs * (r * bs_mat_cols + k) + limb];

    for (int c = 0; c < mat_cols; c++) {
        uint8_t a = mat_b[r * mat_cols + c];

        uint64_t b =
            bs_mat_b[m_vec_limbs * (c * bs_mat_cols + k) + limb];

        sum ^= gf16_vec_mul(b, a);
    }

    acc_b[m_vec_limbs * (r * bs_mat_cols + k) + limb] = sum;
}