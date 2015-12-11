#include<iostream>
#include"data/notstable/blas.h"

int main (int argc, char* argv[]) {
    pop::testBlas::test_scal();
    pop::testBlas::test_axpy();
    pop::testBlas::test_ger();
    pop::testBlas::test_gemv();
    pop::testBlas::test_gemm();
    pop::testBlas::test_dot();
    return 0;
}
