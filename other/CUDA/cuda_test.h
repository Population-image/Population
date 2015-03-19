#ifndef MATRIXMUL_H
#define MATRIXMUL_H

#include "popconfig.h"

#if defined(HAVE_CUDA)
void cuda_test();
void test_cublas(void);
void test_convolution(void);
#endif

#endif
