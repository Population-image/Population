#include<iostream>
#include"data/notstable/blas.h"

int main (int argc, char* argv[]) {
    popblas::testBlas::test_scal();
    popblas::testBlas::test_axpy();
    popblas::testBlas::test_ger();
    popblas::testBlas::test_gemv();
    popblas::testBlas::test_gemm();
    popblas::testBlas::test_dot();
    return 0;
}
