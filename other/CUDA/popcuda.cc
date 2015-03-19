#include "popcuda.h"

bool popcuda::isCudaAvailable() {
#if defined(HAVE_CUDA)
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);

    if (error == cudaErrorInsufficientDriver || error == cudaErrorNoDevice) {
        return false;
    } else {
        return (count > 0);
    }
#else
	return false;
#endif
}

#if defined(HAVE_CUDA)
const char* popcuda::cublasGetErrorString(cublasStatus_t status)
{
	switch(status)
	{
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "unknown error";
}

unsigned int popcuda::getMaxNumberThreadsPerBlock() {
	int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    return deviceProp.maxThreadsPerBlock;
}
#endif
