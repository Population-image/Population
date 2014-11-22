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

