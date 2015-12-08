#ifndef BLAS_H
#define BLAS_H

#include"data/mat/MatN.h"

namespace pop {

struct blas {
    // Math matrix vector BLAS
    // y = ay
    template < typename PixelType >
    void scal(float alpha, MatN<PixelType>& matY) {
        for (unsigned int i=0 ; i < matY.size() ; i ++) {
            matY
        }
    }

    // y = ax + y
    void
    axpy(floatTensor& tensor1, float alpha);
    // A =axyT + A
    void
    ger(floatTensor& x, floatTensor& y, float alpha);
    // y = aAx + by
    void
    gemv(floatTensor& M, char transM, floatTensor& v, float alpha, float beta);
    // C = aAB + bC
    void
    gemm(floatTensor& A, char transA, floatTensor& B, char transB, float alpha,
            float beta);
    // v = x * y
    float
    dot(floatTensor& src);

    float
    sumSquared();

    float
    averageSquare();
};

}

#endif // BLAS_H

