
#include"data/notstable/blas.h"
#ifdef HAVE_ACML
void popblas::blas::ger(float alpha, pop::MatN<2, pop::F32> &vecX, pop::MatN<2, pop::F32> &vecY, pop::MatN<2, pop::F32> &matA) {
    std::cout << "use BLAS ger" << std::endl;
    POP_DbgAssertMessage((vecX.sizeI() == 1 || vecX.sizeJ() == 1) && (vecY.sizeI() == 1 || vecY.sizeJ() == 1) && (matA.sizeI() == vecX.size()) && (matA.sizeJ() == vecY.size()), "[ERROR] blas::ger, vector and matrix sizes are not compatible");
    int sizeY = vecY.getDomain().multCoordinate(), sizeX = vecX.getDomain().multCoordinate();
    ger_(&sizeY, &sizeX, &alpha, &vecY[0], &otherMatN::getStrideVector(vecY), &vecX[0],
            &otherMatN::getStrideVector(vecX), &matA[0], &matA.getDomain()(1));
}

void popblas::blas::gemv(float alpha, pop::MatN<2, pop::F32> &matA, pop::MatN<2, pop::F32> &vecX, float beta, pop::MatN<2, pop::F32> &vecY) {
    POP_DbgAssertMessage((vecX.sizeI() == 1 || vecX.sizeJ() == 1) && (vecY.sizeI() == 1 || vecY.sizeJ() == 1) && (matA.sizeI() == vecY.size()) && (matA.sizeJ() == vecX.size()), "[ERROR] blas::gemv, vector and matrix sizes are not compatible");
    char transM = 'T';
    gemv_(&transM, &matA.getDomain()(1), &matA.getDomain()(0), &alpha, &matA[0], &matA.getDomain()(1), &vecX[0],
            &otherMatN::getStrideVector(vecX), &beta, &vecY[0], &otherMatN::getStrideVector(vecY), 1);
}

void popblas::blas::gemv(float alpha, pop::MatN<2, pop::F32> &matA, char transA, pop::MatN<2, pop::F32> &vecX, float beta, pop::MatN<2, pop::F32> &vecY) {
    std::cout << "use BLAS gemv" << std::endl;
    POP_DbgAssertMessage((vecX.sizeI() == 1 || vecX.sizeJ() == 1) && (vecY.sizeI() == 1 || vecY.sizeJ() == 1), "[ERROR] blas::gemv, vector and matrix sizes are not compatible");
    char opTransA = 'T';
    if (transA == 'T') {
        opTransA = 'N';
    }
    gemv_(&opTransA, &matA.getDomain()(1), &matA.getDomain()(0), &alpha, &matA[0], &matA.getDomain()(1), &vecX[0],
            &otherMatN::getStrideVector(vecX), &beta, &vecY[0], &otherMatN::getStrideVector(vecY), 1);
}

void popblas::blas::gemm(float alpha, pop::MatN<2, pop::F32> &matA, pop::MatN<2, pop::F32> &matB, float beta, pop::MatN<2, pop::F32> &matC) {
    int m, n, k;
    int lda, ldb, ldc;
    char transA('N'), transB('N');
    m = matB.getDomain()(1);
    n = matA.getDomain()(0);
    k = matB.getDomain()(0);
    lda = m;
    ldb = k;
    ldc = m;
    gemm_(&transB, &transA, &m, &n, &k, &alpha, &matB[0], &lda, &matA[0], &ldb,
            &beta, &matC[0], &ldc, 1, 1);
}

void popblas::blas::gemm(float alpha, pop::MatN<2, pop::F32> &matA, char transA, pop::MatN<2, pop::F32> &matB, char transB, float beta, pop::MatN<2, pop::F32> &matC) {
    std::cout << "use BLAS gemm" << std::endl;
    int m, n, k;
    int lda, ldb, ldc;
    m = matC.getDomain()(1);
    n = matC.getDomain()(0);
    ldc = m;

    if (transB == 'N') {
        k = matB.getDomain()(0);
        lda = m;
    } else {
        k = matB.getDomain()(1);
        lda = k;
    }
    if (transA == 'N') {
        ldb = k;
    } else {
        ldb = n;
    }
    gemm_(&transB, &transA, &m, &n, &k, &alpha, &matB[0], &lda, &matA[0], &ldb,
            &beta, &matC[0], &ldc, 1, 1);
}

#endif

void popblas::testBlas::test_scal() {
    pop::F32 data[] = {0, 2, 3,
                       1, 5, 8};
    pop::MatN<2, pop::F32> mat(pop::VecN<2, int>(2, 3), data);
    std::cout << mat << std::endl;
    blas::scal(2.2, mat);
    std::cout << mat << std::endl;

    pop::MatN<2, pop::F32> col = mat.selectRow(1);
    std::cout << col << std::endl;
    blas::scal(2, col);
    std::cout << mat << std::endl;
}

void popblas::testBlas::test_axpy() {
    pop::F32 dataX[] = {0, 2, 3,
                        1, 5, 8};
    pop::MatN<2, pop::F32> matX(pop::VecN<2, int>(2, 3), dataX);
    pop::F32 dataY[] = {5, 3, 1,
                        9, 0, 3};
    pop::MatN<2, pop::F32> matY(pop::VecN<2, int>(2, 3), dataY);
    blas::axpy(3, matX, matY);
    std::cout << matY << std::endl;
    pop::MatN<2, pop::F32> col1Y = matY.selectRow(0);
    pop::MatN<2, pop::F32> col2Y = matY.selectRow(1);
    blas::axpy(1, col1Y, col2Y);
    std::cout << matY << std::endl;
}

void popblas::testBlas::test_ger() {
    pop::F32 dataX[] = {2,3,4};
    pop::F32 datamatX[] = {2, 3, 4,
                      1, 5, 3,
                      2, 1, 0};
    pop::F32 dataY[] = {1,3,5,7,2};
    pop::F32 alpha = 2;
    pop::MatN<2, pop::F32> vecX(pop::VecN<2, int>(1, 3), dataX);
    pop::MatN<2, pop::F32> matX(pop::VecN<2, int>(3, 3), datamatX);
    pop::MatN<2, pop::F32> vecY(pop::VecN<2, int>(1, 5), dataY);
    pop::MatN<2, pop::F32> matA(3, 5);
    pop::MatN<2, pop::F32> matB(3, 5);
    matA = 0; matB = 0;
    blas::ger(alpha, vecX, vecY, matA);
    pop::MatN<2, pop::F32> vecmatX = matX.selectRow(0);
    blas::ger(alpha, vecmatX, vecY, matB);
    std::cout << matA << std::endl;
    std::cout << matB << std::endl;
}

void popblas::testBlas::test_gemv() {
    float alpha = 2;
    pop::F32 dataA[] = {0, 3, 2,
                   1, -2, 3,
                   5, 6, 2,
                   0, 1, 2,
                   3, 6, 1};
    pop::F32 dataX[] = {1, 2, 3};
    float beta = 1;
    pop::F32 dataY[] = {1,1,1,1,1};
    pop::MatN<2, pop::F32> vecX(pop::VecN<2, int>(3, 1), dataX);
    pop::MatN<2, pop::F32> vecY(pop::VecN<2, int>(5, 1), dataY);
    pop::MatN<2, pop::F32> matA(pop::VecN<2, int>(5, 3), dataA);
    pop::MatN<2, pop::F32> opMatA = matA.transpose();
    blas::gemv(alpha, opMatA, 'T', vecX, beta, vecY);
    std::cout << vecY << std::endl;
}

void popblas::testBlas::test_gemm() {
    pop::F32 alpha = 1;
    pop::F32 beta = 1;
    pop::F32 dataA[] = {0, 3, 2,
                   1, -2, 3,
                   5, 6, 2,
                   0, 1, 2,
                   3, 6, 1};
    pop::F32 dataB[] = {1, 3, 2, 4,
                   2, 2, 0, 5,
                   0, 1, 7, 2};
    pop::MatN<2, pop::F32> matA(pop::VecN<2, int>(5, 3), dataA);
    pop::MatN<2, pop::F32> opMatA = matA.transpose();
    pop::MatN<2, pop::F32> matB(pop::VecN<2, int>(3, 4), dataB);
    pop::MatN<2, pop::F32> opMatB = matB.transpose();
    pop::MatN<2, pop::F32> matC(5, 4);
    matC = 0;
    blas::gemm(alpha, opMatA, 'T', opMatB, 'T', beta, matC);
    std::cout << matC << std::endl;
}

void popblas::testBlas::test_dot() {
    pop::F32 dataA[] = {1, 2, 1,
                   2, 4, 5,
                   2, 2, 1};
    pop::F32 dataB[] = {2, 0, 0,
                   1, 2, 2,
                   3, 1, 2};
    pop::MatN<2, pop::F32> matA(pop::VecN<2, int>(3, 3), dataA);
    pop::MatN<2, pop::F32> matB(pop::VecN<2, int>(3, 3), dataB);
    std::cout << blas::dot(matA, matB) << std::endl;
//    pop::MatN<2, pop::F32> vecA = matA.selectRow(0);
//    pop::MatN<2, pop::F32> vecB = matB.selectColumn(0);
//    std::cout << blas::dot(vecA, vecB) << std::endl;
}
