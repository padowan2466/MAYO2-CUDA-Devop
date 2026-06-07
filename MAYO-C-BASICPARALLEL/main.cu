#include <stdio.h>

#include "cuda_kernel.cuh"
#include "inputs.cuh"
#include "parameters.cuh"
#include "mayo.cuh"
#include <stdlib.h>

#define PRINT_KEYS 0

int main()
{
    int res = MAYO_OK;
    unsigned char msg[32] = {0xe};

    size_t msglen = 32;
    size_t smlen = MAYO_2_sig_bytes + msglen;

    unsigned char *sig = (unsigned char *)calloc(smlen, 1);



#if PRINT_KEYS
    printf("pk: ");
    for (int i = 0; i < MAYO_2_cpk_bytes; i++) {
    printf("%02x, ", pk[i]);
    }
    printf("\n");

    printf("sk: ");
    for (int i = 0; i < MAYO_2_csk_bytes; i++) {
    printf("%02x, ", sk[i]);
    }
    printf("\n");
#endif

    printf("Start sign\r\n");

    res = mayo2_sign(sig, &smlen, msg, msglen, sk);
    printf("Sign result: %d, smlen: %ld\r\n", res, (long)smlen);

    printf("Signature:\n");
    for (int i = 0; i < MAYO_2_sig_bytes; i++) {
        printf("%02x, ", sig[i]);
    }
    printf("\nMessage:\n");
    for (int i = 0; i < msglen; i++) {
        printf("%02x, ", sig[MAYO_2_sig_bytes + i]);
    }
    printf("\n");

    free(sig);
    return 0;
}