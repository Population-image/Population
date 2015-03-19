#ifndef _POPCUDA_H
#define _POPCUDA_H

#include <iostream>

#include "popconfig.h"

#if defined(HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
 * gpuErrorCheck is a macro to wrap every CUDA call and verify that they succeded. See http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 * To use it:
 * 	gpuErrorCheck( cudaMalloc(&a_d, size*sizeof(int)) );
 * Or, for a kernel:
 * kernel<<<1,1>>>(a);
 * gpuErrorCheck( cudaPeekAtLastError() );
 * gpuErrorCheck( cudaDeviceSynchronize() );
 */
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}
#endif

namespace popcuda {

/* return true if cuda is enabled and there is a cuda device */
bool isCudaAvailable();

#if defined(HAVE_CUDA)
const char* cublasGetErrorString(cublasStatus_t status);

unsigned int getMaxNumberThreadsPerBlock();
#endif

}


#endif
