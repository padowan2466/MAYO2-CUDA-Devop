#include <stdio.h>

#include "cuda_kernel.cuh"
#include "inputs.cuh"
#include "parameters.cuh"
#include "mayo.cuh"


int main()
{
  
  
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


    return 0;
}