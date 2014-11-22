#ifndef _POPCUDA_H
#define _POPCUDA_H

#include <iostream>

#include "popconfig.h"

#if defined(HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace popcuda {

/* return true if cuda is enabled and there is a cuda device */
bool isCudaAvailable();

/*
 * gpuErrorCheck is a macro to wrap every CUDA call and verify that they succeded. See http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 * To use it:
 * 	gpuErrchk( cudaMalloc(&a_d, size*sizeof(int)) );
 * Or, for a kernel:
 * kernel<<<1,1>>>(a);
 * gpuErrchk( cudaPeekAtLastError() );
 * gpuErrchk( cudaDeviceSynchronize() );
 */
#if defined(HAVE_CUDA)
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

}

#endif
