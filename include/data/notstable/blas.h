#ifndef BLAS_H
#define BLAS_H

#include"data/mat/MatN.h"
#ifdef HAVE_ACML
#include "acml.h"
#include "amdlibm.h"
#endif

#ifdef HAVE_ACML
#define copy_ scopy_
#define exp_  amd_vrsa_expf
#define log_  amd_vrsa_logf
#define scal_ sscal_
#define ger_  sger_
#define dot_  sdot_
#define gemv_ sgemv_
#define gemm_ sgemm_
#define axpy_ saxpy_
#define pow_  amd_vrsa_powxf
#define tanh_ lost_
#endif

namespace pop {

struct otherMatN {
    template <int DIM, typename PixelType>
    inline static int& getStrideVector(MatN<DIM, PixelType>& mat) {
        if (mat.columns() == 1) {
            return mat.stride()(0);
        }
        else {
            return mat.stride()(1);
        }
    }
};

struct blas {
    // Math matrix vector BLAS
    // y = ay
    template < int DIM, typename PixelType >
    static void scal(float alpha, MatN<DIM, PixelType>& matY) {
        matY *= alpha;
    }

#ifdef HAVE_ACML
    template < int DIM >
    static void scal(float alpha, MatN<DIM, F32>& matY) {
        std::cout << "use BLAS scal" << std::endl;
        int size = matY.getDomain().multCoordinate();
        scal_(&size, &alpha, &matY[0], &pop::otherMatN::getStrideVector(matY));
    }
#endif

    // y = ax + y
    template < int DIM, typename PixelType >
    static void axpy(float alpha, const MatN<DIM, PixelType>& matX, MatN<DIM, PixelType>& matY) {
        POP_DbgAssertMessage(matY.rows() == matX.rows() && matY.columns() == matX.columns(), "[ERROR] blas::axpy, matX and matY donot have the same size");
        matY += (matX * alpha);
    }
#ifdef HAVE_ACML
    template < int DIM >
    static void axpy(float alpha, MatN<DIM, F32> &matX, MatN<DIM, F32> &matY) {
        std::cout << "use BLAS axpy" << std::endl;
        POP_DbgAssertMessage(matY.rows() == matX.rows() && matY.columns() == matX.columns(), "[ERROR] blas::axpy, matX and matY donot have the same size");
        int length = matY.getDomain().multCoordinate();
        axpy_(&length, &alpha, &matX[0], &pop::otherMatN::getStrideVector(matY), &matY[0], &pop::otherMatN::getStrideVector(matY));
    }
#endif

    // A =axyT + A
    template < typename PixelType >
    static void ger(float alpha, const MatN<2, PixelType>& vecX, const MatN<2, PixelType>& vecY, MatN<2, PixelType>& matA) {
        POP_DbgAssertMessage((vecX.sizeI() == 1 || vecX.sizeJ() == 1) && (vecY.sizeI() == 1 || vecY.sizeJ() == 1) && (matA.sizeI() == vecX.size()) && (matA.sizeJ() == vecY.size()), "[ERROR] blas::ger, vector and matrix sizes are not compatible");
        typename MatN<2, PixelType>::IteratorEDomain it = matA.getIteratorEDomain();
        while(it.next()) {
            matA(it.x()) += alpha*vecX(it.x()(0))*vecY(it.x()(1));
        }
    }

#ifdef HAVE_ACML
    static void ger(float alpha, MatN<2, F32> &vecX, MatN<2, F32> &vecY, MatN<2, F32> &matA);
#endif

    // y = aAx + by
    template<typename PixelType >
    static void gemv(float alpha, const MatN<2, PixelType>& matA, const MatN<2, PixelType>& vecX, float beta, MatN<2, PixelType>& vecY){
        POP_DbgAssertMessage((vecX.sizeI() == 1 || vecX.sizeJ() == 1) && (vecY.sizeI() == 1 || vecY.sizeJ() == 1) && (matA.sizeI() == vecY.size()) && (matA.sizeJ() == vecX.size()), "[ERROR] blas::gemv, vector and matrix sizes are not compatible");
        for (unsigned int i=0 ; i<vecY.size() ; i++) {
            vecY(i) *= beta;
            for (unsigned int j=0 ; j<vecX.size() ; j++) {
                vecY(i) += alpha*matA(i, j)*vecX(j);
            }
        }
    }

#ifdef HAVE_ACML
    static void gemv(float alpha, MatN<2, F32> &matA, MatN<2, F32> &vecX, float beta, MatN<2, F32> &vecY);
#endif

    // C = aAB + bC
    template < typename PixelType >
    static void gemm(float alpha, const MatN<2, PixelType>& matA, const MatN<2, PixelType>& matB, float beta, MatN<2, PixelType>& matC) {
        POP_DbgAssertMessage((matA.sizeI() == matC.sizeI()) && (matA.sizeJ() == matB.sizeI()) && (matB.sizeJ() == matC.sizeJ()), "[ERROR] blas::gemm, matrix sizes are not compatible");
        matC *= beta;
        matC += (matA * matB) * alpha;
    }

#ifdef HAVE_ACML
    static void gemm(float alpha, MatN<2, F32> &matA, MatN<2, F32> &matB, float beta, MatN<2, F32> &matC);
#endif

    // v = x * y
    template < int DIM, typename PixelType >
    static PixelType dot(const MatN<DIM, PixelType>& matX, const MatN<DIM, PixelType>& matY) {
        PixelType sum = 0;
        typename MatN<DIM, PixelType>::IteratorEDomain itX = matX.getIteratorEDomain();
        typename MatN<DIM, PixelType>::IteratorEDomain itY = matY.getIteratorEDomain();
        while (itX.next() && itY.next()) {
            sum += matX(itX.x()) * matY(itY.x());
        }
        return sum;
    }

#ifdef HAVE_ACML
    template<int DIM >
    static F32 dot(MatN<DIM, F32> &matX, MatN<DIM, F32> &matY) {
        std::cout << "use BLAS dot" << std::endl;
        int size = matY.getDomain().multCoordinate();
        return dot_(&size, &matY[0], &pop::otherMatN::getStrideVector(matY), &matX[0], &pop::otherMatN::getStrideVector(matX));
    }
#endif
};

struct testBlas {
    static void test_scal();
    static void test_axpy();
    static void test_ger();
    static void test_gemv();
    static void test_gemm();
    static void test_dot();
};

}

#endif // BLAS_H

