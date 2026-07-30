#include <stdio.h>

#include "mayo.cuh"
#include "aes_cu.cuh"

#define DEBUG_PRINT 0
#define PRINT_RESULT 0

#define P1O_TILE_C   16
#define P1O_THREADS 128

#define MATMUL_TILE_C   16
#define MATMUL_THREADS 288

/* Definitions */
int mayo2_sign_signature(unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const unsigned char *csk);

int mayo_expand_sk(const unsigned char *csk, sk_t *sk);

static void encode(const unsigned char *m, unsigned char *menc, int mlen) {
  int i;
  for (i = 0; i < mlen / 2; ++i, m += 2) {
    menc[i] = (*m) | (*(m + 1) << 4);
  }

  if (mlen % 2 == 1) {
    menc[i] = (*m);
  }
}



__global__ void decode(const unsigned char *m,
                       unsigned char *mdec,
                       int mdeclen,
                       int blocks_per_sig,
                       int input_stride,
                       int output_stride);

__global__ void unpack_m_vecs(const unsigned char *in,
                              uint64_t *out,
                              int total_vecs);


__global__
void P1P1t_times_O_tiled(
    const uint64_t *__restrict__ P1,
    const unsigned char *__restrict__ O,
    uint64_t *__restrict__ acc,
    int p_limbs_per_sig
);
                                     
__device__ __forceinline__ int upper_triangular_index(int r, 
                                                      int c, 
                                                      int param_v);

void mayo_secure_clear(void *mem, 
                       size_t size); 


__global__
void mul_add_mat_x_m_mat_tiled(
    int m_vec_limbs,
    const unsigned char *__restrict__ mat,
    const uint64_t *__restrict__ bs_mat,
    uint64_t *__restrict__ acc,
    int mat_rows,
    int mat_cols,
    int bs_mat_cols,
    int mat_stride,
    int bs_mat_stride,
    int acc_stride
);

__global__
void P1_times_Vt_tiled(
    const uint64_t *P1,
    const unsigned char *Vdec,
    uint64_t *Pv,
    int p_limbs_per_sig,
    int Vdec_stride,
    int Pv_stride);


__global__
void compute_rhs_finegrain(const uint64_t *vPv,
                                  const unsigned char *t,
                                  unsigned char *y);

__global__
void compute_A_build(const uint64_t *VtL,
                            uint64_t *A_work);

__global__
void compute_A_transpose_cooperative(
    uint64_t *A_work,
    int total_tiles
);

__global__
void compute_A_reduce(uint64_t *A_work);

__global__
void compute_A_decode(const uint64_t *A_work,
                             unsigned char *A_out);

__global__
void zero_last_column_A(unsigned char *A);

__global__
void sample_solution_prepare(const unsigned char *r,
                             unsigned char *x,
                             unsigned char *A,
                             const unsigned char *y,
                             unsigned char *Ar);

__device__
void EF_device(unsigned char *A);

__global__
void sample_solution_finish(unsigned char *A,
                            unsigned char *x,
                            unsigned char *sol_found);

__global__
void build_s(const unsigned char *O,
                    const unsigned char *Vdec,
                    const unsigned char *x,
                    unsigned char *s);

__global__
void pack_sk_kernel(sk_t *d_sk,
                    const unsigned char *d_O,
                    const uint64_t *d_P_limbs);

__global__
void build_tmp_for_salt_kernel(unsigned char *d_tmp,
                               const unsigned char *d_digest,
                               const unsigned char *d_seed_sk);

__global__
void insert_salt_in_tmp_kernel(unsigned char *d_tmp,
                               const unsigned char *d_salt);

__global__
void set_ctr_in_tmp_kernel(unsigned char *d_tmp, int ctr_group);

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
    if (ret != MAYO_OK || siglen != (size_t)param_sig_bytes) {
        memset(sm, 0, siglen + mlen);
        return MAYO_ERR;
    }

    *smlen = siglen + mlen;
    return ret;
}

int mayo2_sign_signature(unsigned char *sig,
              size_t *siglen, const unsigned char *m,
              size_t mlen, const unsigned char *csk)
{
    float throughput[REPEATS+1];
    int ret;
    for(int rep = 0;rep <= REPEATS; rep++)
    {
    ret = MAYO_OK;

    /* EXPAND SK */
    unsigned char *h_seed_sk;
    unsigned char *d_S, *d_seed_sk;


    unsigned char *d_O;
    int mdeclen;


    const int m_vec_limbs = MAYO_2_m_vec_limbs;

    const int p1_vecs = MAYO_2_v * (MAYO_2_v + 1) / 2;
    const int p2_vecs = MAYO_2_v * MAYO_2_o;
    const int total_vecs = p1_vecs + p2_vecs;

    const int packed_bytes = MAYO_2_P1_bytes + MAYO_2_P2_bytes;
    const int unpacked_bytes = total_vecs * (MAYO_2_m_vec_limbs * sizeof(uint64_t));

   






    /************************************************ */

    float gpu_t; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const unsigned char *seed_sk;

    unsigned char *h_msg;
    unsigned char *h_tmp, *d_tmp;
    unsigned char *d_m;

    unsigned char *h_salt, *d_salt;

    unsigned char *ctrbyte;

    unsigned char *h_tenc, *d_tenc;

    unsigned char *h_t, *d_t;

    unsigned char *h_V, *d_V;

    unsigned char *h_digest, *d_digest;

    unsigned char *d_Vdec, *h_Vdec;

    uint64_t *d_VL, *h_VL;

    unsigned char *h_y, *d_y;

    int p1_limbs = p1_vecs * MAYO_2_m_vec_limbs;
    int p2_limbs = p2_vecs * MAYO_2_m_vec_limbs;
    int p_limbs_per_sig = p1_limbs + p2_limbs;

    unsigned char *h_A, *d_A;
    uint64_t *d_Pv, *h_Pv;

    uint64_t *d_A_work;
    unsigned char *d_A_out;
    unsigned char *h_A_out;

    unsigned char *h_x, *d_x;
    unsigned char *h_Ar, *d_Ar;

    int pairs = (MAYO_2_k + 1) * MAYO_2_k / 2;
    int A_width = ((MAYO_2_o * MAYO_2_k + 15) / 16) * 16;
    int A_work_limbs_per_batch =
        A_width * ((MAYO_2_m + pairs + 15) / 16);

    int A_out_bytes_per_batch =
        MAYO_2_m * MAYO_2_A_cols;

    unsigned char *h_r, *d_r;
    int r_per_batch = MAYO_2_k * MAYO_2_o;

    unsigned char *h_sol_found, *d_sol_found;

    int ko = MAYO_2_k * MAYO_2_o;
    int x_stride = MAYO_2_k * MAYO_2_n;

    unsigned char *h_s, *d_s;

    int s_stride = MAYO_2_k * MAYO_2_n;
    int O_stride = MAYO_2_v * MAYO_2_o;

    // int total_bytes_all =
    //     BATCH * total_vecs * MAYO_2_m_vec_limbs * sizeof(uint64_t);

    // int blocks_unpack =
    //     (total_bytes_all + THREADS - 1) / THREADS;

    int in_bytes_per_sig = (MAYO_2_v * MAYO_2_o + 1) / 2;
    int blocks_per_sig = (in_bytes_per_sig + THREADS - 1) / THREADS;

    int total_outputs = BATCH * MAYO_2_v * MAYO_2_o;

    // int blocks_P1P1t =
    //     (total_outputs + THREADS - 1) / THREADS;

    int O_elems = MAYO_2_v * MAYO_2_o;
    int P_elems = P1_LIMBS_MAX + P2_LIMBS_MAX;

    int total = BATCH * (O_elems + P_elems);
    int blocks_pack = (total + THREADS - 1) / THREADS;

    /* AES */
    unsigned char *d_P_packed_batch;
    uint64_t *d_P_unpacked_batch;
    uint64_t *d_rkeys_batch = NULL;
    


    


        /**** EXPAND SK ***/
        cudaMallocHost((void**)&h_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t)*BATCH);
        



        cudaMalloc((void**)&d_S, (MAYO_2_pk_seed_bytes + MAYO_2_O_bytes) * sizeof(uint8_t)*BATCH);
        cudaMalloc((void**)&d_seed_sk, MAYO_2_sk_seed_bytes * sizeof(uint8_t)*BATCH);
        cudaMalloc((void**)&d_O, MAYO_2_v * MAYO_2_o * sizeof(unsigned char)*BATCH);
        
        /******************************************************************************* */







        cudaMallocHost((void**)&h_msg, mlen*BATCH);
        cudaMallocHost((void**)&h_tmp,(MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1)*BATCH);
        cudaMallocHost((void**)&h_salt, (MAYO_2_salt_bytes)*BATCH);
        cudaMallocHost((void**)&h_tenc, (MAYO_2_m_bytes)*BATCH);
        cudaMallocHost((void**)&h_t, (MAYO_2_m)*BATCH);
        cudaMallocHost((void**)&h_V, (MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes)*BATCH);
        cudaMallocHost((void**)&h_digest, MAYO_2_digest_bytes * BATCH);
        cudaMallocHost((void**)&h_Vdec, MAYO_2_v*MAYO_2_k*BATCH);
        cudaMallocHost((void**)&h_VL, MAYO_2_k * MAYO_2_o * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);
        cudaMallocHost((void**)&h_Pv, MAYO_2_v * MAYO_2_k * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);
        cudaMallocHost((void**)&h_A,  MAYO_2_k * MAYO_2_k * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);
        cudaMallocHost((void**)&h_y, MAYO_2_m * BATCH);
        cudaMallocHost((void **)&h_A_out, A_out_bytes_per_batch * BATCH);
        cudaMallocHost((void**)&h_r, MAYO_2_k * MAYO_2_o * BATCH);
        cudaMallocHost((void**)&h_x,  x_stride * BATCH);
        cudaMallocHost((void**)&h_Ar, MAYO_2_m * BATCH);
        cudaMallocHost((void**)&h_sol_found, BATCH);
        cudaMallocHost((void**)&h_s, s_stride * BATCH);


        cudaMalloc((void**)&d_tmp,(MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1)*BATCH);
        cudaMalloc((void**)&d_digest, MAYO_2_digest_bytes * BATCH);
        cudaMalloc((void**)&d_m, mlen*BATCH);
        cudaMalloc((void**)&d_salt, MAYO_2_salt_bytes*BATCH);
        cudaMalloc((void**)&d_tenc, MAYO_2_m_bytes*BATCH);
        cudaMalloc((void**)&d_t, MAYO_2_m*BATCH);
        cudaMalloc((void**)&d_V, (MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes)*BATCH);
        cudaMalloc((void**)&d_Vdec, MAYO_2_v*MAYO_2_k*BATCH);
        cudaMalloc((void**)&d_VL, MAYO_2_k * MAYO_2_o * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);
        cudaMalloc((void**)&d_Pv, MAYO_2_v * MAYO_2_k * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);
        cudaMalloc((void**)&d_A,  MAYO_2_k * MAYO_2_k * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);
        cudaMalloc((void**)&d_y, MAYO_2_m * BATCH);
        cudaMalloc((void **)&d_A_work, A_work_limbs_per_batch * sizeof(uint64_t) * BATCH);
        cudaMalloc((void **)&d_A_out, A_out_bytes_per_batch * BATCH);
        cudaMalloc((void**)&d_r, MAYO_2_k * MAYO_2_o * BATCH);
        cudaMalloc((void**)&d_x,  x_stride * BATCH);
        cudaMalloc((void**)&d_Ar, MAYO_2_m * BATCH);
        cudaMalloc((void**)&d_sol_found, BATCH);
        cudaMalloc((void**)&d_s, s_stride * BATCH);


        /* AES */
        

        cudaMalloc((void **)&d_P_packed_batch,
                BATCH * packed_bytes);

        cudaMalloc((void **)&d_P_unpacked_batch,
                BATCH * unpacked_bytes);

        cudaMalloc((void **)&d_rkeys_batch,
                BATCH * AES128_SK_EXP_WORDS * sizeof(uint64_t));
        

    
        



        

        // alignas(32) sk_t sk[BATCH];

        // ret = mayo_expand_sk(csk, sk);
        
        /***************************************************************************************/
        /***************************************************************************************/
        /**************************************MAYO_EXPAND_SK***********************************/
        /***************************************************************************************/
        /***************************************************************************************/

        /* CSK */
        for (int j = 0; j < BATCH; j++)
        {
            for (int i = 0; i < MAYO_2_sk_seed_bytes; i++)
            {
                h_seed_sk[i + j * MAYO_2_sk_seed_bytes] = csk[i];
            }
        }

        cudaMemcpy(d_seed_sk,
                h_seed_sk,
                MAYO_2_sk_seed_bytes * BATCH,
                cudaMemcpyHostToDevice);

        for(int i=0;i<BATCH;i++)
        {
            for(int j=0;j<mlen;j++) h_msg[j + i*mlen] = m[j];
        }


        cudaMemcpy(d_m, h_msg,  mlen*BATCH, cudaMemcpyHostToDevice);


        cudaMemset(d_A_work,
                    0,
                    A_work_limbs_per_batch * sizeof(uint64_t) * BATCH);

        cudaMemset(d_A_out,
                    0,
                    A_out_bytes_per_batch * BATCH);

        cudaMemset(d_x, 0, x_stride * BATCH);
        cudaMemset(d_Ar, 0, MAYO_2_m * BATCH);
        cudaMemset(d_sol_found, 0, BATCH);

        cudaEventRecord(start);


        /* SHAKE256: seed_sk -> seed_pk || O_bytes */
        shake256<<<BATCH,25>>>(
            d_S,
            MAYO_2_pk_seed_bytes + MAYO_2_O_bytes,
            d_seed_sk,
            MAYO_2_sk_seed_bytes,
            0
        );


        /*  seed_pk for AES */
        // cudaMemcpy2D(
        //     h_seed_pk,
        //     MAYO_2_pk_seed_bytes,
        //     d_S,
        //     MAYO_2_pk_seed_bytes + MAYO_2_O_bytes,
        //     MAYO_2_pk_seed_bytes,
        //     BATCH,
        //     cudaMemcpyDeviceToHost
        // );

        /* Decode O in GPU */
        decode<<<blocks_per_sig * BATCH, THREADS>>>(
            d_S + MAYO_2_pk_seed_bytes,
            d_O,
            MAYO_2_v * MAYO_2_o,
            blocks_per_sig,
            MAYO_2_pk_seed_bytes + MAYO_2_O_bytes,
            MAYO_2_v * MAYO_2_o
        );


        const int S_stride = MAYO_2_pk_seed_bytes + MAYO_2_O_bytes;

        int threads_key = 128;
        int blocks_key = (BATCH + threads_key - 1) / threads_key;

        aes128_key_expand_batch<<<blocks_key, threads_key>>>(
            d_S,
            S_stride,
            d_rkeys_batch,
            BATCH
        );

        const int groups_per_batch = (packed_bytes + 63) / 64;
        const int total_groups = BATCH * groups_per_batch;

        int blocks_aes = (total_groups + THREADS - 1) / THREADS;

        aes128_ctr4x_batch<<<blocks_aes, THREADS>>>(
            d_P_packed_batch,
            packed_bytes,
            d_rkeys_batch,
            groups_per_batch,
            BATCH
        );


        int total_unpack_threads = BATCH * total_vecs;
        int blocks_unpack = (total_unpack_threads + THREADS - 1) / THREADS;

        unpack_m_vecs_batch<<<blocks_unpack, THREADS>>>(
            d_P_packed_batch,
            d_P_unpacked_batch,
            packed_bytes,
            total_vecs,
            BATCH
        );

        

        /* CPU AES */
        // for (int i = 0; i < BATCH; i++)
        // {
        //     AES_128_CTR((unsigned char *)h_P + i * packed_bytes,
        //                 packed_bytes,
        //                 h_seed_pk + i * MAYO_2_pk_seed_bytes,
        //                 MAYO_2_pk_seed_bytes);
        // }
        


        
        uint64_t *d_P_limbs =
        d_P_unpacked_batch;

    /*
    * Un bloque procesa una fila de una firma.
    */
    int blocks_P1P1t =
        BATCH * MAYO_2_v;

    /*
    * L = (P1 + P1^T)O + P2
    */
    P1P1t_times_O_tiled
        <<<blocks_P1P1t, P1O_THREADS>>>(
            d_P_limbs,
            d_O,
            d_P_limbs + p1_limbs,
            p_limbs_per_sig
        );

    cudaError_t p1o_error =
        cudaGetLastError();

    if (p1o_error != cudaSuccess)
    {
        printf(
            "Error P1P1t_times_O_tiled: %s\n",
            cudaGetErrorString(p1o_error)
        );
    }

    /* Pointers */
    uint64_t *d_P_batch_alias =
        d_P_limbs;

    unsigned char *d_O_batch_alias =
        d_O;

    uint64_t *d_L =
        d_P_limbs + p1_limbs;
        
        // cudaDeviceSynchronize();
        
        // cudaMemcpy(h_P,
        //        d_P_limbs,
        //        unpacked_bytes * BATCH,
        //        cudaMemcpyDeviceToHost);


        // for (int i = 0; i < BATCH; i++)
        // {
        //     memcpy(d_sk[i].O,
        //         h_O + i * (MAYO_2_v * MAYO_2_o),
        //         MAYO_2_v * MAYO_2_o * sizeof(unsigned char));

        //     memcpy(d_sk[i].p,
        //         (unsigned char *)h_P + i * unpacked_bytes,
        //         unpacked_bytes);
        // }

        

        /***************************************************************************************/
        /***************************************************************************************/
        /***************************************************************************************/
        /***************************************************************************************/
        /***************************************************************************************/

        // #if DEBUG_PRINT
        // printf("GPU O first: %02x\r\n", sk[0].O[0]);
        // printf("GPU O last : %02x\r\n", sk[0].O[MAYO_2_v * MAYO_2_o - 1]);
        // printf("GPU p first: %016llx\r\n", (unsigned long long)sk[0].p[0]);
        // printf("GPU p last : %016llx\r\n", (unsigned long long)sk[0].p[p_limbs_per_sig - 1]);
        // #endif

        // cudaMemcpy(
        // sk,
        // d_sk,
        // BATCH * sizeof(sk_t),
        // cudaMemcpyDeviceToHost
        // );

        // for (int i = 0; i < BATCH; i++)
        // {
        //     memcpy(h_P_batch + i * p_limbs_per_sig,
        //            sk[i].p,
        //            p_limbs_per_sig * sizeof(uint64_t));
        //     memcpy(h_O_batch + i * MAYO_2_v * MAYO_2_o,
        //         sk[i].O,
        //         MAYO_2_v * MAYO_2_o);
        // }

        // cudaMemcpy(d_P_batch,
        //            h_P_batch,
        //            p_limbs_per_sig * sizeof(uint64_t) * BATCH,
        //            cudaMemcpyHostToDevice);

        // cudaMemcpy(d_O_batch,
        //        h_O_batch,
        //        MAYO_2_v * MAYO_2_o * BATCH,
        //        cudaMemcpyHostToDevice);

        // uint64_t *d_L = d_P_batch + p1_limbs;



        #if DEBUG_PRINT
        printf("\nmsg\n");
        for (size_t i = 0; i < mlen; i++) {
        printf("0x%02x, ", m[i]);
        }
        printf("\r\n");
        #endif


        

        

        


        shake256<<<BATCH,25>>>(d_digest, MAYO_2_digest_bytes, d_m, mlen, 0);

        // cudaMemcpy(h_digest,
        //         d_tmp,
        //         MAYO_2_digest_bytes * BATCH,
        //         cudaMemcpyDeviceToHost);


        int tmp_bytes =
            MAYO_2_digest_bytes +
            MAYO_2_salt_bytes +
            MAYO_2_sk_seed_bytes +
            1;

        int total_tmp = BATCH * tmp_bytes;
        int blocks_tmp = (total_tmp + THREADS - 1) / THREADS;
        // for (int j = 0; j < BATCH; j++)
        // {
        //     unsigned char *tmp_j = h_tmp + j * tmp_bytes;

        //     memcpy(tmp_j,
        //         h_digest + j * MAYO_2_digest_bytes,
        //         MAYO_2_digest_bytes);

        //     for (int i = 0; i < MAYO_2_salt_bytes; i++)
        //     {
        //         tmp_j[MAYO_2_digest_bytes + i] = 1;
        //     }

        //     memcpy(tmp_j + MAYO_2_digest_bytes + MAYO_2_salt_bytes,
        //         csk,
        //         MAYO_2_sk_seed_bytes);

        //     tmp_j[MAYO_2_digest_bytes +
        //         MAYO_2_salt_bytes +
        //         MAYO_2_sk_seed_bytes] = 0;
        // }
        build_tmp_for_salt_kernel<<<blocks_tmp, THREADS>>>(
        d_tmp,
        d_digest,
        d_seed_sk
        );

        #if DEBUG_PRINT
        printBatch(h_tmp, tmp_bytes, BATCH, "tmp:");
        #endif

        // cudaMemcpy(d_tmp, h_tmp, (MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1)*BATCH, cudaMemcpyHostToDevice);
    
        shake256<<<BATCH, 25>>>(d_salt, MAYO_2_salt_bytes, d_tmp, MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes,
                                MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes + 1);

        int total_salt = BATCH * MAYO_2_salt_bytes;
        int blocks_salt = (total_salt + THREADS - 1) / THREADS;

        insert_salt_in_tmp_kernel<<<blocks_salt, THREADS>>>(
            d_tmp,
            d_salt
        );

        // cudaMemcpy(h_salt, d_salt, MAYO_2_salt_bytes*BATCH, cudaMemcpyDeviceToHost);

        

        // for (int i = 0; i < BATCH; i++)
        // {
        //     unsigned char *tmp_i = h_tmp + i * tmp_bytes;
        //     unsigned char *salt_i = h_salt + i * MAYO_2_salt_bytes;

        //     memcpy(tmp_i + MAYO_2_digest_bytes,
        //         salt_i,
        //         MAYO_2_salt_bytes);
        // }

        // cudaMemcpy(d_tmp, h_tmp, tmp_bytes*BATCH, cudaMemcpyHostToDevice);

        shake256<<<BATCH, 25>>>(d_tenc, MAYO_2_m_bytes, d_tmp, MAYO_2_digest_bytes + MAYO_2_salt_bytes, tmp_bytes);

        int blocks = (((MAYO_2_m  + 1)/2) + THREADS-1)/THREADS;
        decode<<<blocks*BATCH, THREADS>>>(d_tenc, d_t, MAYO_2_m, blocks,MAYO_2_m_bytes, MAYO_2_m);

        int sol_found = 0;
        int sol_batch = -1;
        int sol_ctr = -1;


        for (int ctr = 0; ctr <= 255; ctr++)
        {
            int blocks_ctr = (BATCH + THREADS - 1) / THREADS;

            set_ctr_in_tmp_kernel<<<blocks_ctr, THREADS>>>(
                d_tmp,
                ctr
            );
            // for (int b = 0; b < BATCH; b++)
            // {
            //     int current_ctr = ctr * BATCH + b;
            //     h_tmp[b * tmp_bytes + tmp_bytes - 1] = (unsigned char)(current_ctr <= 255 ? current_ctr : 255);
            // }

            // cudaMemcpy(d_tmp,
            //         h_tmp,
            //         tmp_bytes * BATCH,
            //         cudaMemcpyHostToDevice);

            
            shake256<<<BATCH, 25>>>(
                d_V,
                MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes,
                d_tmp,
                tmp_bytes,
                tmp_bytes
            );

            
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

        
            cudaMemset(
            d_VL,
            0,
            MAYO_2_k *
            MAYO_2_o *
            MAYO_2_m_vec_limbs *
            sizeof(uint64_t) *
            BATCH
            );

        mul_add_mat_x_m_mat_tiled
        <<<BATCH, MATMUL_THREADS>>>(
            MAYO_2_m_vec_limbs,
            d_Vdec,
            d_L,
            d_VL,
            MAYO_2_k,
            MAYO_2_v,
            MAYO_2_o,
            MAYO_2_k * MAYO_2_v,
            p_limbs_per_sig,
            MAYO_2_k *
                MAYO_2_o *
                MAYO_2_m_vec_limbs
        );

        /* P1 × V^T con tiling */

        constexpr int P1_ROWS_PER_BLOCK = 8;

        constexpr int P1_THREADS_PER_BLOCK =
            P1_ROWS_PER_BLOCK *
            MAYO_2_k *
            MAYO_2_m_vec_limbs;

        dim3 blocks_Pv(
            (MAYO_2_v + P1_ROWS_PER_BLOCK - 1) /
                P1_ROWS_PER_BLOCK,
            BATCH
        );

        dim3 threads_Pv(
            P1_THREADS_PER_BLOCK
        );

        P1_times_Vt_tiled<<<blocks_Pv, threads_Pv>>>(
            d_P_batch_alias,
            d_Vdec,
            d_Pv,
            p_limbs_per_sig,
            MAYO_2_k * MAYO_2_v,
            MAYO_2_v *
                MAYO_2_k *
                MAYO_2_m_vec_limbs
        );

            
            cudaMemset(d_A,
                    0,
                    MAYO_2_k * MAYO_2_k * MAYO_2_m_vec_limbs * sizeof(uint64_t) * BATCH);

            mul_add_mat_x_m_mat_tiled
        <<<BATCH, MATMUL_THREADS>>>(
            MAYO_2_m_vec_limbs,
            d_Vdec,
            d_Pv,
            (uint64_t *)d_A,
            MAYO_2_k,
            MAYO_2_v,
            MAYO_2_k,
            MAYO_2_k * MAYO_2_v,
            MAYO_2_v *
                MAYO_2_k *
                MAYO_2_m_vec_limbs,
            MAYO_2_k *
                MAYO_2_k *
                MAYO_2_m_vec_limbs
        );

            
            compute_rhs_finegrain<<<BATCH, MAYO_2_m_vec_limbs>>>(
                (uint64_t *)d_A,
                d_t,
                d_y
            );

            
            

            int total_build =
        BATCH *
        pairs *
        MAYO_2_o *
        MAYO_2_m_vec_limbs *
        2;

    int blocks_build =
        (total_build + THREADS - 1) /
        THREADS;

    compute_A_build<<<blocks_build, THREADS>>>(
        d_VL,
        d_A_work
    );


    /* Transposición cooperativa 16x16 */

        constexpr int TRANSPOSE_THREADS = 32;
        constexpr int TRANSPOSE_TILES_PER_BLOCK = 2;

        int transpose_tiles_per_batch =
            A_work_limbs_per_batch / 16;

        int total_transpose_tiles =
            BATCH * transpose_tiles_per_batch;

        int blocks_transpose =
            (
                total_transpose_tiles +
                TRANSPOSE_TILES_PER_BLOCK - 1
            ) / TRANSPOSE_TILES_PER_BLOCK;

        compute_A_transpose_cooperative
            <<<blocks_transpose, TRANSPOSE_THREADS>>>(
                d_A_work,
                total_transpose_tiles
            );


        int total_reduce =
            BATCH *
            (A_width / 16) *
            pairs;

        int blocks_reduce =
            (total_reduce + THREADS - 1) /
            THREADS;

        compute_A_reduce<<<blocks_reduce, THREADS>>>(
            d_A_work
        );

            int total_decode =
                BATCH *
                MAYO_2_m *
                (MAYO_2_A_cols - 1);

            int blocks_decode = (total_decode + THREADS - 1) / THREADS;

            compute_A_decode<<<blocks_decode, THREADS>>>(
                d_A_work,
                d_A_out
            );

            
            int total_zero = BATCH * MAYO_2_m;
            int blocks_zero = (total_zero + THREADS - 1) / THREADS;

            zero_last_column_A<<<blocks_zero, THREADS>>>(
                d_A_out
            );

            
            int r_blocks = (((MAYO_2_k * MAYO_2_o + 1) / 2) + THREADS - 1) / THREADS;

            decode<<<r_blocks * BATCH, THREADS>>>(
                d_V + MAYO_2_k * MAYO_2_v_bytes,
                d_r,
                MAYO_2_k * MAYO_2_o,
                r_blocks,
                MAYO_2_k * MAYO_2_v_bytes + MAYO_2_r_bytes,
                MAYO_2_k * MAYO_2_o
            );

            
            int total_prepare = BATCH * MAYO_2_m;

            if (BATCH * ko > total_prepare) {
                total_prepare = BATCH * ko;
            }

            int blocks_prepare = (total_prepare + THREADS - 1) / THREADS;

            

            sample_solution_prepare<<<blocks_prepare, THREADS>>>(
                d_r,
                d_x,
                d_A_out,
                d_y,
                d_Ar
            );

            
            sample_solution_finish<<<BATCH, 1>>>(
                d_A_out,
                d_x,
                d_sol_found
            );

            cudaDeviceSynchronize();

            cudaMemcpy(h_sol_found,
                    d_sol_found,
                    BATCH,
                    cudaMemcpyDeviceToHost);

            for (int b = 0; b < BATCH; b++)
            {
                #if DEBUG_PRINT
                printf("ctr %d | Batch %d sample_solution return: %d\r\n",
                    ctr,
                    b,
                    h_sol_found[b]);
                #endif

                if (h_sol_found[b])
                {
                    sol_found = 1;
                    sol_batch = b;
                    sol_ctr = ctr;
                    break;
                }
            }

            if (sol_found)
            {
                break;
            }

            
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_t, start, stop);

        // printf("GPU mayo2_sign_signature total time: %.6f ms\n", gpu_t);
        // printf("GPU mayo2_sign_signature time per batch: %.6f ms\n", gpu_t/BATCH);
        printf("GPU mayo2_sign_signature time per batch: %.6f ms\n", gpu_t/BATCH);
        throughput[rep]= BATCH/gpu_t;


        if (!sol_found)
        {
            ret = MAYO_ERR;
        }
        else
        {   
            #if DEBUG_PRINT
            printf(
    "Solution found at ctr %d, batch %d\r\n",
    sol_ctr,
    sol_batch
);

            cudaMemcpy(h_x,
                    d_x,
                    x_stride * BATCH,
                    cudaMemcpyDeviceToHost);

            printElement(h_x + sol_batch * x_stride,
                        ko,
                        "x solution:");
            #endif

            int total_s = BATCH * MAYO_2_k * MAYO_2_n;
            int blocks_s = (total_s + THREADS - 1) / THREADS;

            build_s<<<blocks_s, THREADS>>>(
                d_O_batch_alias,
                d_Vdec,
                d_x,
                d_s
            );

            cudaDeviceSynchronize();

            cudaMemcpy(h_s,
                d_s,
                s_stride * BATCH,
                cudaMemcpyDeviceToHost);

            #if PRINT_RESULT
            printBatch(
            h_s,
            MAYO_2_k * MAYO_2_n,
            BATCH,
            "s:"
            );
            #endif

            encode(h_s + sol_batch * s_stride, sig, MAYO_2_k * MAYO_2_n);

            cudaMemcpy(sig + MAYO_2_sig_bytes - MAYO_2_salt_bytes,
                d_salt + sol_batch * MAYO_2_salt_bytes,
                MAYO_2_salt_bytes,
                cudaMemcpyDeviceToHost);

            *siglen = MAYO_2_sig_bytes;
        }


        
        

        cudaFreeHost(h_seed_sk);
        cudaFreeHost(h_msg);
        cudaFreeHost(h_tmp);
        cudaFreeHost(h_salt);
        cudaFreeHost(h_tenc);
        cudaFreeHost(h_t);
        cudaFreeHost(h_V);
        cudaFreeHost(h_digest);
        cudaFreeHost(h_Vdec);
        cudaFreeHost(h_VL);
        cudaFreeHost(h_Pv);
        cudaFreeHost(h_y);
        cudaFreeHost(h_A);
        cudaFreeHost(h_A_out);
        cudaFreeHost(h_r);
        cudaFreeHost(h_x);
        cudaFreeHost(h_Ar);
        cudaFreeHost(h_sol_found);
        cudaFreeHost(h_s);
        

        

        cudaFree(d_S);
        cudaFree(d_seed_sk);
        cudaFree(d_O);
        cudaFree(d_tmp);
        cudaFree(d_m);
        cudaFree(d_salt);
        cudaFree(d_tenc);
        cudaFree(d_t);
        cudaFree(d_V);
        cudaFree(d_Vdec);
        cudaFree(d_VL);
        cudaFree(d_Pv);
        cudaFree(d_A);
        cudaFree(d_y);
        cudaFree(d_A_work);
        cudaFree(d_A_out);
        cudaFree(d_r);
        cudaFree(d_x);
        cudaFree(d_Ar);
        cudaFree(d_sol_found);
        cudaFree(d_s);
        cudaFree(d_digest);
        cudaFree(d_P_packed_batch);
        cudaFree(d_P_unpacked_batch);
        cudaFree(d_rkeys_batch);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    float sumThput = 0;
    for(int i = 0; i<REPEATS; i++)
    {
        printf("Throughput %d: %.6f\r\n", i+1,1000.0/throughput[i+1]);
        sumThput += 1000.0/throughput[i+1];
    }

    printf("Avg Throughput: %.6f\r\n", sumThput/REPEATS);




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
void P1P1t_times_O_tiled(
    const uint64_t *__restrict__ P1,
    const unsigned char *__restrict__ O,
    uint64_t *__restrict__ acc,
    int p_limbs_per_sig)
{
    constexpr int OUTPUTS_PER_ROW =
        MAYO_2_o * MAYO_2_m_vec_limbs;

    
    int block_id = blockIdx.x;

    int batch = block_id / MAYO_2_v;
    int row   = block_id % MAYO_2_v;

    
    if (batch >= BATCH) {
        return;
    }

    int output_id = threadIdx.x;

    bool active =
        output_id < OUTPUTS_PER_ROW;

    int k = 0;
    int limb = 0;

    if (active) {
        limb = output_id % MAYO_2_m_vec_limbs;
        k    = output_id / MAYO_2_m_vec_limbs;
    }

    
    __shared__ uint64_t sP1
        [P1O_TILE_C]
        [MAYO_2_m_vec_limbs];

    
    __shared__ unsigned char sO
        [P1O_TILE_C]
        [MAYO_2_o];

    const uint64_t *P1_batch =
        P1 + batch * p_limbs_per_sig;

    const unsigned char *O_batch =
        O + batch * MAYO_2_v * MAYO_2_o;

    
    uint64_t *acc_batch =
        acc + batch * p_limbs_per_sig;

    uint64_t sum = 0;

    for (int c_base = 0;
         c_base < MAYO_2_v;
         c_base += P1O_TILE_C)
    {
        
        constexpr int P1_TILE_ELEMENTS =
            P1O_TILE_C * MAYO_2_m_vec_limbs;

        for (int index = threadIdx.x;
             index < P1_TILE_ELEMENTS;
             index += blockDim.x)
        {
            int tile_col =
                index / MAYO_2_m_vec_limbs;

            int load_limb =
                index % MAYO_2_m_vec_limbs;

            int col =
                c_base + tile_col;

            uint64_t value = 0;

            
            if (col < MAYO_2_v &&
                col != row)
            {
                int triangular_row;
                int triangular_col;

                if (row < col) {
                    triangular_row = row;
                    triangular_col = col;
                } else {
                    triangular_row = col;
                    triangular_col = row;
                }

                int p1_index =
                    upper_triangular_index(
                        triangular_row,
                        triangular_col,
                        MAYO_2_v
                    );

                value =
                    P1_batch[
                        p1_index *
                        MAYO_2_m_vec_limbs +
                        load_limb
                    ];
            }

            sP1[tile_col][load_limb] =
                value;
        }

        
        constexpr int O_TILE_ELEMENTS =
            P1O_TILE_C * MAYO_2_o;

        for (int index = threadIdx.x;
             index < O_TILE_ELEMENTS;
             index += blockDim.x)
        {
            int tile_col =
                index / MAYO_2_o;

            int load_k =
                index % MAYO_2_o;

            int col =
                c_base + tile_col;

            unsigned char value = 0;

            if (col < MAYO_2_v) {
                value =
                    O_batch[
                        col * MAYO_2_o +
                        load_k
                    ];
            }

            sO[tile_col][load_k] =
                value;
        }

        
        __syncthreads();

        
        if (active)
        {
            #pragma unroll
            for (int tile_col = 0;
                 tile_col < P1O_TILE_C;
                 tile_col++)
            {
                uint64_t p1_value =
                    sP1[tile_col][limb];

                unsigned char o_value =
                    sO[tile_col][k];

                sum ^=
                    gf16_vec_mul(
                        p1_value,
                        o_value
                    );
            }
        }

        
        __syncthreads();
    }

    if (active)
    {
        int output_index =
            (row * MAYO_2_o + k) *
            MAYO_2_m_vec_limbs +
            limb;

        
        acc_batch[output_index] ^=
            sum;
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
void mul_add_mat_x_m_mat_tiled(
    int m_vec_limbs,
    const unsigned char *__restrict__ mat,
    const uint64_t *__restrict__ bs_mat,
    uint64_t *__restrict__ acc,
    int mat_rows,
    int mat_cols,
    int bs_mat_cols,
    int mat_stride,
    int bs_mat_stride,
    int acc_stride)
{
    
    int batch = blockIdx.x;
    int tx = threadIdx.x;

    if (batch >= BATCH) {
        return;
    }

    
    __shared__ unsigned char sMat
        [MAYO_2_k]
        [MATMUL_TILE_C];

    __shared__ uint64_t sBs
        [MATMUL_TILE_C]
        [MAYO_2_o]
        [MAYO_2_m_vec_limbs];

    const unsigned char *mat_b =
        mat + batch * mat_stride;

    const uint64_t *bs_mat_b =
        bs_mat + batch * bs_mat_stride;

    uint64_t *acc_b =
        acc + batch * acc_stride;

    
    int outputs_per_batch =
        mat_rows *
        bs_mat_cols *
        m_vec_limbs;

    bool active =
        tx < outputs_per_batch;

    int r = 0;
    int k = 0;
    int limb = 0;
    int output_index = 0;

    if (active)
    {
        output_index = tx;

        limb =
            output_index % m_vec_limbs;

        int temp =
            output_index / m_vec_limbs;

        k =
            temp % bs_mat_cols;

        r =
            temp / bs_mat_cols;
    }

    
    uint64_t sum = 0;

    if (active)
    {
        sum =
            acc_b[
                m_vec_limbs *
                (r * bs_mat_cols + k) +
                limb
            ];
    }

    
    for (int c_base = 0;
         c_base < mat_cols;
         c_base += MATMUL_TILE_C)
    {
        
        int total_mat_elements =
            mat_rows * MATMUL_TILE_C;

        for (int index = tx;
             index < total_mat_elements;
             index += blockDim.x)
        {
            int load_r =
                index / MATMUL_TILE_C;

            int tile_c =
                index % MATMUL_TILE_C;

            int global_c =
                c_base + tile_c;

            unsigned char value = 0;

            if (global_c < mat_cols)
            {
                value =
                    mat_b[
                        load_r * mat_cols +
                        global_c
                    ];
            }

            sMat[load_r][tile_c] =
                value;
        }

        
        int total_bs_elements =
            MATMUL_TILE_C *
            bs_mat_cols *
            m_vec_limbs;

        for (int index = tx;
             index < total_bs_elements;
             index += blockDim.x)
        {
            int load_limb =
                index % m_vec_limbs;

            int temp =
                index / m_vec_limbs;

            int load_k =
                temp % bs_mat_cols;

            int tile_c =
                temp / bs_mat_cols;

            int global_c =
                c_base + tile_c;

            uint64_t value = 0;

            if (global_c < mat_cols)
            {
                value =
                    bs_mat_b[
                        m_vec_limbs *
                        (
                            global_c *
                            bs_mat_cols +
                            load_k
                        ) +
                        load_limb
                    ];
            }

            sBs
                [tile_c]
                [load_k]
                [load_limb] =
                value;
        }

        
        __syncthreads();

        
        if (active)
        {
            #pragma unroll
            for (int tile_c = 0;
                 tile_c < MATMUL_TILE_C;
                 tile_c++)
            {
                unsigned char a =
                    sMat[r][tile_c];

                uint64_t b =
                    sBs[tile_c][k][limb];

                sum ^=
                    gf16_vec_mul(b, a);
            }
        }

        
        __syncthreads();
    }

    if (active)
    {
        acc_b[
            m_vec_limbs *
            (r * bs_mat_cols + k) +
            limb
        ] = sum;
    }
}
#define TILE_C 16
#define ROWS_PER_BLOCK 8

__global__
void P1_times_Vt_tiled(
    const uint64_t *P1,
    const unsigned char *Vdec,
    uint64_t *Pv,
    int p_limbs_per_sig,
    int Vdec_stride,
    int Pv_stride)
{
    constexpr int OUTPUTS_PER_ROW =
        MAYO_2_k * MAYO_2_m_vec_limbs;

   
    __shared__ uint64_t sP1
        [ROWS_PER_BLOCK]
        [TILE_C]
        [MAYO_2_m_vec_limbs];

    __shared__ unsigned char sV
        [MAYO_2_k]
        [TILE_C];

    int tx = threadIdx.x;

    
    int row_base = blockIdx.x * ROWS_PER_BLOCK;
    int batch = blockIdx.y;

    
    int local_row = tx / OUTPUTS_PER_ROW;
    int output_in_row = tx % OUTPUTS_PER_ROW;

    int k =
        output_in_row / MAYO_2_m_vec_limbs;

    int limb =
        output_in_row % MAYO_2_m_vec_limbs;

    int r = row_base + local_row;

    bool valid_row = r < MAYO_2_v;

    const uint64_t *P1_b =
        P1 + batch * p_limbs_per_sig;

    const unsigned char *Vdec_b =
        Vdec + batch * Vdec_stride;

    uint64_t *Pv_b =
        Pv + batch * Pv_stride;

    uint64_t sum = 0;

    
    for (int c_base = row_base;
         c_base < MAYO_2_v;
         c_base += TILE_C)
    {
        
        int total_v_elements =
            MAYO_2_k * TILE_C;

        for (int index = tx;
             index < total_v_elements;
             index += blockDim.x)
        {
            int load_k = index / TILE_C;
            int tile_c = index % TILE_C;

            int global_c = c_base + tile_c;

            if (global_c < MAYO_2_v)
            {
                sV[load_k][tile_c] =
                    Vdec_b[
                        load_k * MAYO_2_v +
                        global_c
                    ];
            }
            else
            {
                sV[load_k][tile_c] = 0;
            }
        }

        
        int total_p1_elements =
            ROWS_PER_BLOCK *
            TILE_C *
            MAYO_2_m_vec_limbs;

        for (int index = tx;
             index < total_p1_elements;
             index += blockDim.x)
        {
            
            int load_limb =
                index % MAYO_2_m_vec_limbs;

            int temp =
                index / MAYO_2_m_vec_limbs;

            int tile_c =
                temp % TILE_C;

            int load_local_row =
                temp / TILE_C;

            int global_r =
                row_base + load_local_row;

            int global_c =
                c_base + tile_c;

            
            if (global_r < MAYO_2_v &&
                global_c < MAYO_2_v &&
                global_c >= global_r)
            {
                int p1_index =
                    upper_triangular_index(
                        global_r,
                        global_c,
                        MAYO_2_v
                    );

                sP1
                    [load_local_row]
                    [tile_c]
                    [load_limb] =
                    P1_b[
                        p1_index *
                        MAYO_2_m_vec_limbs +
                        load_limb
                    ];
            }
            else
            {
                sP1
                    [load_local_row]
                    [tile_c]
                    [load_limb] = 0;
            }
        }

        
        __syncthreads();

        
        if (valid_row)
        {
            #pragma unroll
            for (int tile_c = 0;
                 tile_c < TILE_C;
                 tile_c++)
            {
                uint64_t p1_val =
                    sP1
                        [local_row]
                        [tile_c]
                        [limb];

                unsigned char v =
                    sV[k][tile_c];

                sum ^=
                    gf16_vec_mul(
                        p1_val,
                        v
                    );
            }
        }

        
        __syncthreads();
    }

    if (valid_row)
    {
        Pv_b[
            (r * MAYO_2_k + k) *
            MAYO_2_m_vec_limbs +
            limb
        ] = sum;
    }
}
__global__
void compute_rhs_finegrain(const uint64_t *vPv,
                                  const unsigned char *t,
                                  unsigned char *y)                              
{
    int batch = blockIdx.x;
    int tid = threadIdx.x;   

    const int param_k = MAYO_2_k;
    const int param_m = MAYO_2_m;
    const int m_vec_limbs = MAYO_2_m_vec_limbs; 

    if (tid >= m_vec_limbs) {
        return;
    }

    const int top_pos = ((param_m - 1) % 16) * 4;

    const uint64_t *vPv_b =
        vPv + batch * param_k * param_k * m_vec_limbs;

    const unsigned char *t_b =
        t + batch * param_m;

    unsigned char *y_b =
        y + batch * param_m;

    __shared__ uint64_t temp[M_VEC_LIMBS_MAX];
    __shared__ unsigned char top;

    temp[tid] = 0;

    __syncthreads();

    for (int i = param_k - 1; i >= 0; i--) {
        for (int j = i; j < param_k; j++) {

           
            if (tid == m_vec_limbs - 1) {
                top = (unsigned char)((temp[m_vec_limbs - 1] >> top_pos) & 0xF);
            }

            __syncthreads();

            
            uint64_t old_limb = temp[tid];
            uint64_t carry_in = 0;

            if (tid > 0) {
                carry_in = temp[tid - 1] >> 60;
            }

            __syncthreads();

            temp[tid] = (old_limb << 4) ^ carry_in;

            __syncthreads();

            
            if (tid == 0) 
            {
                unsigned char *temp_bytes = (unsigned char *)temp;

                for (int jj = 0; jj < F_TAIL_LEN; jj++) {
                    unsigned char val = mul_f(top, d_f_tail_64[jj]);

                    if ((jj & 1) == 0) {
                        temp_bytes[jj / 2] ^= val;
                    } else {
                        temp_bytes[jj / 2] ^= val << 4;
                    }
                }
            }

            __syncthreads();

        
            uint64_t a =
                vPv_b[(i * param_k + j) * m_vec_limbs + tid];

            uint64_t b = 0;

            if (i != j) {
                b = vPv_b[(j * param_k + i) * m_vec_limbs + tid];
            }

            temp[tid] ^= a ^ b;

            __syncthreads();
        }
    }


    for (int idx = tid; idx < param_m; idx += m_vec_limbs) {
        unsigned char *temp_bytes = (unsigned char *)temp;

        unsigned char temp_i =
            (temp_bytes[idx / 2] >> (4 * (idx & 1))) & 0xF;

        y_b[idx] = t_b[idx] ^ temp_i;
    }
}


__global__
void compute_A_build(const uint64_t *VtL,
                            uint64_t *A_work)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int param_k = MAYO_2_k;
    const int param_o = MAYO_2_o;
    const int m_vec_limbs = MAYO_2_m_vec_limbs;

    const int A_width = ((MAYO_2_o * MAYO_2_k + 15) / 16) * 16;
    const int pairs = (MAYO_2_k + 1) * MAYO_2_k / 2;

    int total = BATCH * pairs * MAYO_2_o * m_vec_limbs * 2;

    if (tid >= total) {
        return;
    }

    int local = tid;

    int side = local % 2;
    local /= 2;

    int limb = local % m_vec_limbs;
    local /= m_vec_limbs;

    int c = local % param_o;
    local /= param_o;

    int pair_id = local % pairs;
    int batch = local / pairs;

    int i_found = 0;
    int j_found = 0;
    int count = 0;

    for (int i = 0; i < param_k; i++) {
        for (int j = param_k - 1; j >= i; j--) {
            if (count == pair_id) {
                i_found = i;
                j_found = j;
            }
            count++;
        }
    }

    int i = i_found;
    int j = j_found;

    if (side == 1 && i == j) {
        return;
    }

    int bits_to_shift = (4 * pair_id) % 64;
    int words_to_shift = (4 * pair_id) / 64;

    const uint64_t *VtL_b =
        VtL + batch * param_k * param_o * m_vec_limbs;

    uint64_t *A_b =
        A_work + batch * A_width *
        ((MAYO_2_m + pairs + 15) / 16);

    int source_row;
    int target_col;

    if (side == 0) {
        source_row = j;
        target_col = param_o * i + c;
    } else {
        source_row = i;
        target_col = param_o * j + c;
    }

    uint64_t value =
        VtL_b[(source_row * param_o + c) * m_vec_limbs + limb];

    int pos =
        target_col + (limb + words_to_shift) * A_width;

    atomicXor((unsigned long long *)&A_b[pos],
              (unsigned long long)(value << bits_to_shift));

    if (bits_to_shift > 0) {
        int pos_hi =
            target_col + (limb + words_to_shift + 1) * A_width;

        atomicXor((unsigned long long *)&A_b[pos_hi],
                  (unsigned long long)(value >> (64 - bits_to_shift)));
    }
}

__global__
void compute_A_transpose_cooperative(
    uint64_t *A_work,
    int total_tiles)
{
    
    constexpr int TILE_SIZE = 16;
    constexpr int TILES_PER_BLOCK = 2;

    int group = threadIdx.x / TILE_SIZE;  // 0 o 1
    int lane  = threadIdx.x % TILE_SIZE;  // 0...15

    int tile_id =
        blockIdx.x * TILES_PER_BLOCK + group;

    
    __shared__ uint64_t shared_rows
        [TILES_PER_BLOCK]
        [TILE_SIZE];

    uint64_t input_row = 0;

    if (tile_id < total_tiles)
    {
        input_row =
            A_work[tile_id * TILE_SIZE + lane];
    }

    
    shared_rows[group][lane] = input_row;

    __syncthreads();

    
    if (tile_id >= total_tiles) {
        return;
    }

    uint64_t output_row = 0;

    
    #pragma unroll
    for (int input_r = 0;
         input_r < TILE_SIZE;
         input_r++)
    {
        uint64_t nibble =
            (
                shared_rows[group][input_r]
                >> (4 * lane)
            ) & 0xFULL;

        output_row |=
            nibble << (4 * input_r);
    }

   
    A_work[tile_id * TILE_SIZE + lane] =
        output_row;
}

__global__
void compute_A_reduce(uint64_t *A_work)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int A_width = ((MAYO_2_o * MAYO_2_k + 15) / 16) * 16;
    const int pairs = (MAYO_2_k + 1) * MAYO_2_k / 2;

    int c_blocks = A_width / 16;

    int total = BATCH * c_blocks * pairs;

    if (tid >= total) {
        return;
    }

    int local = tid;

    int pair_row = local % pairs;
    local /= pairs;

    int c_block = local % c_blocks;
    int batch = local / c_blocks;

    int c = c_block * 16;
    int r = MAYO_2_m + pair_row;

    uint64_t *A_b =
        A_work + batch * A_width *
        ((MAYO_2_m + pairs + 15) / 16);

    uint64_t low_bit_in_nibble = 0x1111111111111111ULL;

    size_t pos =
        (r / 16) * A_width + c + (r % 16);

    uint64_t word = A_b[pos];

    uint64_t t0 = word & low_bit_in_nibble;
    uint64_t t1 = (word >> 1) & low_bit_in_nibble;
    uint64_t t2 = (word >> 2) & low_bit_in_nibble;
    uint64_t t3 = (word >> 3) & low_bit_in_nibble;

    for (int t = 0; t < F_TAIL_LEN; t++) {
        unsigned char f = d_f_tail_64[t];

        unsigned char tab0 = mul_f(f, 1);
        unsigned char tab1 = mul_f(f, 2);
        unsigned char tab2 = mul_f(f, 4);
        unsigned char tab3 = mul_f(f, 8);

        uint64_t value =
            t0 * tab0 ^
            t1 * tab1 ^
            t2 * tab2 ^
            t3 * tab3;

        int rr = r + t - MAYO_2_m;

        size_t dst =
            (rr / 16) * A_width + c + (rr % 16);

        atomicXor((unsigned long long *)&A_b[dst],
                  (unsigned long long)value);
    }
}

__global__
void compute_A_decode(const uint64_t *A_work,
                             unsigned char *A_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int A_width = ((MAYO_2_o * MAYO_2_k + 15) / 16) * 16;
    const int A_cols = MAYO_2_k * MAYO_2_o + 1;

    int useful_cols = A_cols - 1;

    int total = BATCH * MAYO_2_m * useful_cols;

    if (tid >= total) {
        return;
    }

    int local = tid;

    int col = local % useful_cols;
    local /= useful_cols;

    int row = local % MAYO_2_m;
    int batch = local / MAYO_2_m;

    const uint64_t *A_b =
        A_work + batch * A_width *
        ((MAYO_2_m + ((MAYO_2_k + 1) * MAYO_2_k / 2) + 15) / 16);

    unsigned char *A_out_b =
        A_out + batch * MAYO_2_m * A_cols;

    int c_base = (col / 16) * 16;
    int c_off = col % 16;

    int row_block = (row / 16) * 16;
    int row_off = row % 16;

    size_t pos =
        row_block * (A_width / 16) + c_base + row_off;

    uint64_t word = A_b[pos];

    A_out_b[row * A_cols + col] =
        (word >> (4 * c_off)) & 0xF;
}

__global__
void zero_last_column_A(unsigned char *A)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int total = BATCH * MAYO_2_m;

    if (tid >= total) {
        return;
    }

    int batch = tid / MAYO_2_m;
    int row   = tid % MAYO_2_m;

    int A_cols = MAYO_2_k * MAYO_2_o + 1;
    int A_bytes_per_batch = MAYO_2_m * A_cols;

    unsigned char *A_b = A + batch * A_bytes_per_batch;

    A_b[row * A_cols + (A_cols - 1)] = 0;
}

__global__
void sample_solution_prepare(const unsigned char *r,
                             unsigned char *x,
                             unsigned char *A,
                             const unsigned char *y,
                             unsigned char *Ar)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int ko = MAYO_2_k * MAYO_2_o;
    const int m = MAYO_2_m;
    const int A_cols = MAYO_2_k * MAYO_2_o + 1;
    const int x_stride = MAYO_2_k * MAYO_2_n;

    int total_x = BATCH * ko;
    int total_rows = BATCH * m;

    if (tid < total_x) {
        int batch = tid / ko;
        int idx   = tid % ko;

        x[batch * x_stride + idx] = r[batch * ko + idx];
    }

    if (tid < total_rows) {
        int batch = tid / m;
        int row   = tid % m;

        unsigned char *A_b =
            A + batch * m * A_cols;

        const unsigned char *r_b =
            r + batch * ko;

        const unsigned char *y_b =
            y + batch * m;

        unsigned char *Ar_b =
            Ar + batch * m;

        A_b[row * A_cols + (A_cols - 1)] = 0;

        unsigned char acc = 0;

        for (int col = 0; col < ko; col++) {
            unsigned char a = A_b[row * A_cols + col] & 0x0F;
            unsigned char b = r_b[col] & 0x0F;

            acc ^= mul_f(a, b);
        }

        Ar_b[row] = acc;

        A_b[row * A_cols + (A_cols - 1)] =
            (y_b[row] ^ acc) & 0x0F;
    }
}

__device__
void EF_device(unsigned char *A)
{
    const int nrows = MAYO_2_m;
    const int ncols = MAYO_2_A_cols;
    const int row_len = (ncols + 15) / 16;

    uint64_t pivot_row[(MAYO_2_k * MAYO_2_o + 1 + 15) / 16];
    uint64_t pivot_row2[(MAYO_2_k * MAYO_2_o + 1 + 15) / 16];

    uint64_t packed_A[((MAYO_2_k * MAYO_2_o + 1 + 15) / 16) * MAYO_2_m];

    for (int i = 0; i < row_len * nrows; i++) {
        packed_A[i] = 0;
    }

    for (int i = 0; i < nrows; i++) {
        ef_pack_m_vec_device(A + i * ncols,
                             packed_A + i * row_len,
                             ncols);
    }

    int pivot_row_idx = 0;

    for (int pivot_col = 0; pivot_col < ncols; pivot_col++) {
        int pivot_row_lower_bound = pivot_col + nrows - ncols;

        if (pivot_row_lower_bound < 0) {
            pivot_row_lower_bound = 0;
        }

        int pivot_row_upper_bound = pivot_col;

        if (pivot_row_upper_bound > nrows - 1) {
            pivot_row_upper_bound = nrows - 1;
        }

        for (int i = 0; i < row_len; i++) {
            pivot_row[i] = 0;
            pivot_row2[i] = 0;
        }

        unsigned char pivot = 0;
        uint64_t pivot_is_zero = 0xFFFFFFFFFFFFFFFFULL;

        int row_search_upper = pivot_row_upper_bound + 32;

        if (row_search_upper > nrows - 1) {
            row_search_upper = nrows - 1;
        }

        for (int row = pivot_row_lower_bound; row <= row_search_upper; row++) {
            uint64_t is_pivot_row =
                ~ct_compare_64_device((uint64_t)row,
                                      (uint64_t)pivot_row_idx);

            uint64_t below_pivot_row =
                ct_64_is_greater_than_device((uint64_t)row,
                                             (uint64_t)pivot_row_idx);

            for (int j = 0; j < row_len; j++) {
                pivot_row[j] ^=
                    (is_pivot_row | (below_pivot_row & pivot_is_zero)) &
                    packed_A[row * row_len + j];
            }

            pivot = m_extract_element_device(pivot_row, pivot_col);

            pivot_is_zero =
                ~ct_compare_64_device((uint64_t)pivot, 0);
        }

        unsigned char inverse = inverse_f_device(pivot);

        vec_mul_add_u64_device(row_len,
                               pivot_row,
                               inverse,
                               pivot_row2);

        for (int row = pivot_row_lower_bound; row <= pivot_row_upper_bound; row++) {
            uint64_t do_copy =
                ~ct_compare_64_device((uint64_t)row,
                                      (uint64_t)pivot_row_idx) &
                ~pivot_is_zero;

            uint64_t do_not_copy = ~do_copy;

            for (int col = 0; col < row_len; col++) {
                packed_A[row * row_len + col] =
                    (do_not_copy & packed_A[row * row_len + col]) ^
                    (do_copy & pivot_row2[col]);
            }
        }

        for (int row = pivot_row_lower_bound; row < nrows; row++) {
            unsigned char below_pivot =
                (unsigned char)(row > pivot_row_idx);

            unsigned char elt_to_elim =
                m_extract_element_device(packed_A + row * row_len,
                                         pivot_col);

            vec_mul_add_u64_device(row_len,
                                   pivot_row2,
                                   below_pivot * elt_to_elim,
                                   packed_A + row * row_len);
        }

        if (pivot_is_zero == 0) {
            pivot_row_idx++;
        }
    }

    unsigned char temp[MAYO_2_k * MAYO_2_o + 1 + 15];

    for (int i = 0; i < nrows; i++) {
        ef_unpack_m_vec_device(row_len,
                               packed_A + i * row_len,
                               temp);

        for (int j = 0; j < ncols; j++) {
            A[i * ncols + j] = temp[j];
        }
    }
}

__global__
void sample_solution_finish(unsigned char *A,
                            unsigned char *x,
                            unsigned char *sol_found)
{
    int batch = blockIdx.x;

    const int m = MAYO_2_m;
    const int ko = MAYO_2_k * MAYO_2_o;
    const int A_cols = MAYO_2_k * MAYO_2_o + 1;
    const int x_stride = MAYO_2_k * MAYO_2_n;

    unsigned char *A_b =
        A + batch * m * A_cols;

    unsigned char *x_b =
        x + batch * x_stride;

    EF_device(A_b);

    unsigned char full_rank = 0;

    for (int i = 0; i < A_cols - 1; i++) {
        full_rank |= A_b[(m - 1) * A_cols + i];
    }

    if (full_rank == 0) {
        sol_found[batch] = 0;
        return;
    }

    for (int row = m - 1; row >= 0; row--) {
        unsigned char finished = 0;

        int col_upper_bound = row + (32 / (m - row));

        if (col_upper_bound > ko) {
            col_upper_bound = ko;
        }

        for (int col = row; col <= col_upper_bound; col++) {
            unsigned char correct_column =
                ct_compare_8_device(A_b[row * A_cols + col], 0) &
                ~finished;

            unsigned char u =
                correct_column & A_b[row * A_cols + A_cols - 1];

            x_b[col] ^= u;

            for (int i = 0; i < row; i += 8) {
                uint64_t tmp =
                    ((uint64_t)A_b[(i    ) * A_cols + col] <<  0) ^
                    ((uint64_t)A_b[(i + 1) * A_cols + col] <<  8) ^
                    ((uint64_t)A_b[(i + 2) * A_cols + col] << 16) ^
                    ((uint64_t)A_b[(i + 3) * A_cols + col] << 24) ^
                    ((uint64_t)A_b[(i + 4) * A_cols + col] << 32) ^
                    ((uint64_t)A_b[(i + 5) * A_cols + col] << 40) ^
                    ((uint64_t)A_b[(i + 6) * A_cols + col] << 48) ^
                    ((uint64_t)A_b[(i + 7) * A_cols + col] << 56);

                tmp = mul_fx8_device(u, tmp);

                A_b[(i    ) * A_cols + A_cols - 1] ^= (tmp      ) & 0xF;
                A_b[(i + 1) * A_cols + A_cols - 1] ^= (tmp >>  8) & 0xF;
                A_b[(i + 2) * A_cols + A_cols - 1] ^= (tmp >> 16) & 0xF;
                A_b[(i + 3) * A_cols + A_cols - 1] ^= (tmp >> 24) & 0xF;
                A_b[(i + 4) * A_cols + A_cols - 1] ^= (tmp >> 32) & 0xF;
                A_b[(i + 5) * A_cols + A_cols - 1] ^= (tmp >> 40) & 0xF;
                A_b[(i + 6) * A_cols + A_cols - 1] ^= (tmp >> 48) & 0xF;
                A_b[(i + 7) * A_cols + A_cols - 1] ^= (tmp >> 56) & 0xF;
            }

            finished |= correct_column;
        }
    }

    sol_found[batch] = 1;
}

__global__
void build_s(const unsigned char *O,
             const unsigned char *Vdec,
             const unsigned char *x,
             unsigned char *s)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int k = MAYO_2_k;
    const int n = MAYO_2_n;
    const int o = MAYO_2_o;
    const int v = MAYO_2_v;

    const int O_stride    = MAYO_2_v * MAYO_2_o;
    const int Vdec_stride = MAYO_2_k * MAYO_2_v;
    const int x_stride    = MAYO_2_k * MAYO_2_n;
    const int s_stride    = MAYO_2_k * MAYO_2_n;

    int total = BATCH * k * n;

    if (tid >= total) {
        return;
    }

    int batch = tid / (k * n);
    int local = tid % (k * n);

    int i   = local / n;   
    int pos = local % n;   

    const unsigned char *O_b =
        O + batch * O_stride;

    const unsigned char *Vdec_b =
        Vdec + batch * Vdec_stride;

    const unsigned char *x_b =
        x + batch * x_stride;

    unsigned char *s_b =
        s + batch * s_stride;

    const unsigned char *x_i =
        x_b + i * o;

    if (pos < v) {
        unsigned char Ox = 0;

        for (int j = 0; j < o; j++) {
            unsigned char a = O_b[pos * o + j] & 0x0F;
            unsigned char b = x_i[j] & 0x0F;

            Ox ^= mul_f(a, b);
        }

        unsigned char vi = Vdec_b[i * v + pos] & 0x0F;

        s_b[i * n + pos] = (vi ^ Ox) & 0x0F;
    } else {
        int oil_pos = pos - v;

        s_b[i * n + pos] = x_i[oil_pos] & 0x0F;
    }
}

__global__
void pack_sk_kernel(sk_t *d_sk,
                    const unsigned char *d_O,
                    const uint64_t *d_P_limbs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int O_elems = MAYO_2_v * MAYO_2_o;
    const int P_elems = P1_LIMBS_MAX + P2_LIMBS_MAX;

    const int elems_per_sig = O_elems + P_elems;
    const int total_elems = BATCH * elems_per_sig;

    if (tid >= total_elems) {
        return;
    }

    int sig_id = tid / elems_per_sig;
    int local  = tid % elems_per_sig;

    if (local < O_elems) {
        d_sk[sig_id].O[local] =
            d_O[sig_id * O_elems + local];
    } else {
        int p_idx = local - O_elems;

        d_sk[sig_id].p[p_idx] =
            d_P_limbs[sig_id * P_elems + p_idx];
    }
}


__global__
void build_tmp_for_salt_kernel(unsigned char *d_tmp,
                               const unsigned char *d_digest,
                               const unsigned char *d_seed_sk)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int tmp_bytes =
        MAYO_2_digest_bytes +
        MAYO_2_salt_bytes +
        MAYO_2_sk_seed_bytes +
        1;

    int total = BATCH * tmp_bytes;

    if (tid >= total) return;

    int batch_id = tid / tmp_bytes;
    int local    = tid % tmp_bytes;

    unsigned char value = 0;

    if (local < MAYO_2_digest_bytes) {
        value = d_digest[batch_id * MAYO_2_digest_bytes + local];
    }
    else if (local < MAYO_2_digest_bytes + MAYO_2_salt_bytes) {
        value = 1;
    }
    else if (local < MAYO_2_digest_bytes + MAYO_2_salt_bytes + MAYO_2_sk_seed_bytes) {
        int sk_idx = local - MAYO_2_digest_bytes - MAYO_2_salt_bytes;

        
        value = d_seed_sk[batch_id * MAYO_2_sk_seed_bytes + sk_idx];
    }
    else {
        
        value = 0;
    }

    d_tmp[batch_id * tmp_bytes + local] = value;
}

__global__
void insert_salt_in_tmp_kernel(unsigned char *d_tmp,
                               const unsigned char *d_salt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int total = BATCH * MAYO_2_salt_bytes;
    if (tid >= total) return;

    int batch_id = tid / MAYO_2_salt_bytes;
    int j        = tid % MAYO_2_salt_bytes;

    int tmp_bytes =
        MAYO_2_digest_bytes +
        MAYO_2_salt_bytes +
        MAYO_2_sk_seed_bytes +
        1;

    d_tmp[batch_id * tmp_bytes + MAYO_2_digest_bytes + j] =
        d_salt[batch_id * MAYO_2_salt_bytes + j];
}


__global__
void set_ctr_in_tmp_kernel(
    unsigned char *d_tmp,
    int ctr
)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= BATCH) {
        return;
    }

    const int tmp_bytes =
        MAYO_2_digest_bytes +
        MAYO_2_salt_bytes +
        MAYO_2_sk_seed_bytes +
        1;

    
    d_tmp[b * tmp_bytes + tmp_bytes - 1] =
        (unsigned char)ctr;
}
