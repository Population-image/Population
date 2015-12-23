#ifndef BLAS_H
#define BLAS_H

#include"data/mat/MatN.h"

#ifdef HAVE_ACML
#include "acml.h"
#include "amdlibm.h"
#endif

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"
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

#ifdef HAVE_CUBLAS
#define dot_ cublasSdot
#endif

namespace popblas {

struct otherMatN {
    template <int DIM, typename PixelType>
    inline static int& getStrideVector(pop::MatN<DIM, PixelType>& mat) {
        if (mat.columns() == 1) {
            return mat.stride()(0);
        }
        else {
            return mat.stride()(1); ;
        }
    }
};

struct blas {
#if HAVE_CUBLAS
//    static cudaError_t cudaStat;
//    static cublasStatus_t stat;
//    static cublasHandle_t handle;
//    static bool _is_cublas_create;
#endif

    // Math matrix vector BLAS
    // y = ay
    template < int DIM, typename PixelType >
    static void scal(float alpha, pop::MatN<DIM, PixelType>& matY) {
        matY *= alpha;
    }

#ifdef HAVE_ACML
    template < int DIM >
    static void scal(float alpha, pop::MatN<DIM, pop::F32>& matY) {
        std::cout << "use BLAS scal" << std::endl;
        int size = matY.getDomain().multCoordinate();
        scal_(&size, &alpha, &matY[0], &otherMatN::getStrideVector(matY));
    } 
#endif

    // y = ax + y
    template < int DIM, typename PixelType >
    static void axpy(float alpha, const pop::MatN<DIM, PixelType>& matX, pop::MatN<DIM, PixelType>& matY) {
        POP_DbgAssertMessage(matY.rows() == matX.rows() && matY.columns() == matX.columns(), "[ERROR] blas::axpy, matX and matY donot have the same size");
        matY += (matX * alpha);
    }
#ifdef HAVE_ACML
    template < int DIM >
    static void axpy(float alpha, pop::MatN<DIM, pop::F32> &matX, pop::MatN<DIM, pop::F32> &matY) {
        std::cout << "use BLAS axpy" << std::endl;
        POP_DbgAssertMessage(matY.rows() == matX.rows() && matY.columns() == matX.columns(), "[ERROR] blas::axpy, matX and matY donot have the same size");
        int length = matY.getDomain().multCoordinate();
        axpy_(&length, &alpha, &matX[0], &otherMatN::getStrideVector(matY), &matY[0], &otherMatN::getStrideVector(matY));
    }
#endif

    // A =axyT + A
    template < typename PixelType >
    static void ger(float alpha, const pop::MatN<2, PixelType>& vecX, const pop::MatN<2, PixelType>& vecY, pop::MatN<2, PixelType>& matA) {
        POP_DbgAssertMessage((vecX.sizeI() == 1 || vecX.sizeJ() == 1) && (vecY.sizeI() == 1 || vecY.sizeJ() == 1) && (matA.sizeI() == vecX.size()) && (matA.sizeJ() == vecY.size()), "[ERROR] blas::ger, vector and matrix sizes are not compatible");
        typename pop::MatN<2, PixelType>::IteratorEDomain it = matA.getIteratorEDomain();
        while(it.next()) {
            matA(it.x()) += alpha*vecX(it.x()(0))*vecY(it.x()(1));
        }
    }

#ifdef HAVE_ACML
    static void ger(float alpha, pop::MatN<2, pop::F32> &vecX, pop::MatN<2, pop::F32> &vecY, pop::MatN<2, pop::F32> &matA);
#endif

    // y = aAx + by
    template<typename PixelType >
    static void gemv(float alpha, const pop::MatN<2, PixelType>& matA, const pop::MatN<2, PixelType>& vecX, float beta, pop::MatN<2, PixelType>& vecY){
        POP_DbgAssertMessage((vecX.sizeI() == 1 || vecX.sizeJ() == 1) && (vecY.sizeI() == 1 || vecY.sizeJ() == 1) && (matA.sizeI() == vecY.size()) && (matA.sizeJ() == vecX.size()), "[ERROR] blas::gemv, vector and matrix sizes are not compatible");
        for (unsigned int i=0 ; i<vecY.size() ; i++) {
            vecY(i) *= beta;
            for (unsigned int j=0 ; j<vecX.size() ; j++) {
                vecY(i) += alpha*matA(i, j)*vecX(j);
            }
        }
    }

#ifdef HAVE_ACML
    static void gemv(float alpha, pop::MatN<2, pop::F32> &matA, pop::MatN<2, pop::F32> &vecX, float beta, pop::MatN<2, pop::F32> &vecY);
#endif

    template<typename PixelType>
    static void gemv(float alpha, const pop::MatN<2, PixelType> &matA, char transA, const pop::MatN<2, PixelType> &vecX, float beta, pop::MatN<2, PixelType> &vecY) {
        if (transA == 'N') {
            gemv(alpha, matA, vecX, beta, vecY);
        } else {
            pop::MatN<2, PixelType> opMatA = matA.transpose();
            gemv(alpha, opMatA, vecX, beta, vecY);
        }
    }

#ifdef HAVE_ACML
    static void gemv(float alpha, pop::MatN<2, pop::F32> &matA, char transA, pop::MatN<2, pop::F32> &vecX, float beta, pop::MatN<2, pop::F32> &vecY);
#endif

    // C = aAB + bC
    template < typename PixelType >
    static void gemm(float alpha, const pop::MatN<2, PixelType>& matA, const pop::MatN<2, PixelType>& matB, float beta, pop::MatN<2, PixelType>& matC) {
        POP_DbgAssertMessage((matA.sizeI() == matC.sizeI()) && (matA.sizeJ() == matB.sizeI()) && (matB.sizeJ() == matC.sizeJ()), "[ERROR] blas::gemm, matrix sizes are not compatible");
        matC *= beta;
        matC += (matA * matB) * alpha;
    }

#ifdef HAVE_ACML
    static void gemm(float alpha, pop::MatN<2, pop::F32> &matA, pop::MatN<2, pop::F32> &matB, float beta, pop::MatN<2, pop::F32> &matC);
#endif

    template < typename PixelType >
    static void gemm(float alpha, const pop::MatN<2, PixelType>& matA, char transA, const pop::MatN<2, PixelType>& matB, char transB, float beta, pop::MatN<2, PixelType>& matC) {
        pop::MatN<2, PixelType> opMatA(matA);
        pop::MatN<2, PixelType> opMatB(matB);
        if (transA == 'T') {
            opMatA = matA.transpose();
        }
        if (transB == 'T') {
            opMatB = matB.transpose();
        }
        gemm(alpha, opMatA, opMatB, beta, matC);
    }

#ifdef HAVE_ACML
    static void gemm(float alpha, pop::MatN<2, pop::F32> &matA, char transA, pop::MatN<2, pop::F32> &matB, char transB, float beta, pop::MatN<2, pop::F32> &matC);
#endif

    // v = x * y
    template < int DIM, typename PixelType >
    static PixelType dot(const pop::MatN<DIM, PixelType>& matX, const pop::MatN<DIM, PixelType>& matY) {
        PixelType sum = 0;
        typename pop::MatN<DIM, PixelType>::IteratorEDomain itX = matX.getIteratorEDomain();
        typename pop::MatN<DIM, PixelType>::IteratorEDomain itY = matY.getIteratorEDomain();
        while (itX.next() && itY.next()) {
            sum += matX(itX.x()) * matY(itY.x());
        }
        return sum;
    }

#ifdef HAVE_ACML
    template<int DIM >
    static pop::F32 dot(pop::MatN<DIM, pop::F32> &matX, pop::MatN<DIM, pop::F32> &matY) {
        std::cout << "use BLAS dot" << std::endl;
        int size = matY.getDomain().multCoordinate();
        return dot_(&size, &matY[0], &otherMatN::getStrideVector(matY), &matX[0], &otherMatN::getStrideVector(matX));
    }
#endif

#ifdef HAVE_CUBLAS
    template<int DIM >
    static pop::F32 dot(pop::MatN<DIM, pop::F32> &matX, pop::MatN<DIM, pop::F32> &matY) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        int size = matX.getDomain().multCoordinate();
        pop::F32 d_matX[size], d_matY[size];
        //cudaAlloc(size, sizeof(matX[0]), (void**)&d_matX);
        cudaMalloc((void**)&d_matX, size*sizeof(matX[0]));
        //cudaAlloc(size, sizeof(matY[0]), (void**)&d_matY);
        cudaMalloc((void**)&d_matY, size*sizeof(matY[0]));
        //const float* dd_X = d_matX, dd_Y = d_matY;
        cublasSetVector(size, sizeof(matX[0]), (void*)matX.data(), otherMatN::getStrideVector(matX), (void*)d_matX, 1);
        cublasSetVector(size, sizeof(matY[0]), (void*)matY.data(), otherMatN::getStrideVector(matY), (void*)d_matY, 1);
        float result;
        cublasSdot(handle, size, d_matX, 1, d_matY, 1, &result);
        cudaFree((void*)d_matX);
        cudaFree((void*)d_matY);
        return result;
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

