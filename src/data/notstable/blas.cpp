
#include"data/notstable/blas.h"
#ifdef HAVE_ACML
void pop::blas::ger(float alpha, MatN<2, F32> &vecX, MatN<2, F32> &vecY, MatN<2, F32> &matA) {
    std::cout << "use BLAS ger" << std::endl;
    POP_DbgAssertMessage((vecX.sizeI() == 1 || vecX.sizeJ() == 1) && (vecY.sizeI() == 1 || vecY.sizeJ() == 1) && (matA.sizeI() == vecX.size()) && (matA.sizeJ() == vecY.size()), "[ERROR] blas::ger, vector and matrix sizes are not compatible");
    int sizeY = vecY.getDomain().multCoordinate(), sizeX = vecX.getDomain().multCoordinate();
    ger_(&sizeY, &sizeX, &alpha, &vecY[0], &pop::otherMatN::getStrideVector(vecY), &vecX[0],
            &pop::otherMatN::getStrideVector(vecX), &matA[0], &matA.getDomain()(1));
}

void pop::blas::gemv(float alpha, MatN<2, F32> &matA, MatN<2, F32> &vecX, float beta, MatN<2, F32> &vecY) {
    std::cout << "use BLAS gemv" << std::endl;
    POP_DbgAssertMessage((vecX.sizeI() == 1 || vecX.sizeJ() == 1) && (vecY.sizeI() == 1 || vecY.sizeJ() == 1) && (matA.sizeI() == vecY.size()) && (matA.sizeJ() == vecX.size()), "[ERROR] blas::gemv, vector and matrix sizes are not compatible");
    char transM = 'T';
    gemv_(&transM, &matA.getDomain()(1), &matA.getDomain()(0), &alpha, &matA[0], &matA.getDomain()(1), &vecX[0],
            &pop::otherMatN::getStrideVector(vecX), &beta, &vecY[0], &pop::otherMatN::getStrideVector(vecY), 1);
}

void pop::blas::gemm(float alpha, MatN<2, F32> &matA, MatN<2, F32> &matB, float beta, MatN<2, F32> &matC) {
    std::cout << "use BLAS gemm" << std::endl;
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
#endif

void pop::testBlas::test_scal() {
    pop::F32 data[] = {0, 2, 3,
                       1, 5, 8};
    MatN<2, F32> mat(VecN<2, int>(2, 3), data);
    std::cout << mat << std::endl;
    pop::blas::scal(2.2, mat);
    std::cout << mat << std::endl;

    MatN<2, F32> col = mat.selectRow(1);
    std::cout << col << std::endl;
    pop::blas::scal(2, col);
    std::cout << mat << std::endl;
}

void pop::testBlas::test_axpy() {
    pop::F32 dataX[] = {0, 2, 3,
                        1, 5, 8};
    MatN<2, F32> matX(VecN<2, int>(2, 3), dataX);
    pop::F32 dataY[] = {5, 3, 1,
                        9, 0, 3};
    MatN<2, F32> matY(VecN<2, int>(2, 3), dataY);
    pop::blas::axpy(3, matX, matY);
    std::cout << matY << std::endl;
    MatN<2, F32> col1Y = matY.selectRow(0);
    MatN<2, F32> col2Y = matY.selectRow(1);
    pop::blas::axpy(1, col1Y, col2Y);
    std::cout << matY << std::endl;
}

void pop::testBlas::test_ger() {
    F32 dataX[] = {2,3,4};
    F32 datamatX[] = {2, 3, 4,
                      1, 5, 3,
                      2, 1, 0};
    F32 dataY[] = {1,3,5,7,2};
    F32 alpha = 2;
    MatN<2, F32> vecX(VecN<2, int>(1, 3), dataX);
    MatN<2, F32> matX(VecN<2, int>(3, 3), datamatX);
    MatN<2, F32> vecY(VecN<2, int>(1, 5), dataY);
    MatN<2, F32> matA(3, 5);
    MatN<2, F32> matB(3, 5);
    matA = 0; matB = 0;
    pop::blas::ger(alpha, vecX, vecY, matA);
    MatN<2, F32> vecmatX = matX.selectRow(0);
    pop::blas::ger(alpha, vecmatX, vecY, matB);
    std::cout << matA << std::endl;
    std::cout << matB << std::endl;
}

void pop::testBlas::test_gemv() {
    F32 alpha = 2;
    F32 dataA[] = {0, 3, 2,
                   1, -2, 3,
                   5, 6, 2,
                   0, 1, 2,
                   3, 6, 1};
    F32 dataX[] = {1, 2, 3};
    F32 beta = 1;
    F32 dataY[] = {1,1,1,1,1};
    MatN<2, F32> vecX(VecN<2, int>(3, 1), dataX);
    MatN<2, F32> vecY(VecN<2, int>(5, 1), dataY);
    MatN<2, F32> matA(VecN<2, int>(5, 3), dataA);
    pop::blas::gemv(alpha, matA, vecX, beta, vecY);
    std::cout << vecY << std::endl;
}

void pop::testBlas::test_gemm() {
    F32 alpha = 1;
    F32 beta = 1;
    F32 dataA[] = {0, 3, 2,
                   1, -2, 3,
                   5, 6, 2,
                   0, 1, 2,
                   3, 6, 1};
    F32 dataB[] = {1, 3, 2, 4,
                   2, 2, 0, 5,
                   0, 1, 7, 2};
    MatN<2, F32> matA(VecN<2, int>(5, 3), dataA);
    MatN<2, F32> matB(VecN<2, int>(3, 4), dataB);
    MatN<2, F32> matC(5, 4);
    matC = 0;
    pop::blas::gemm(alpha, matA, matB, beta, matC);
    std::cout << matC << std::endl;
}

void pop::testBlas::test_dot() {
    F32 dataA[] = {1, 2, 1,
                   2, 4, 5,
                   2, 2, 1};
    F32 dataB[] = {2, 0, 0,
                   1, 2, 2,
                   3, 1, 2};
    MatN<2, F32> matA(VecN<2, int>(3, 3), dataA);
    MatN<2, F32> matB(VecN<2, int>(3, 3), dataB);
    std::cout << pop::blas::dot(matA, matB) << std::endl;
    MatN<2, F32> vecA = matA.selectRow(0);
    MatN<2, F32> vecB = matB.selectColumn(0);
    std::cout << pop::blas::dot(vecA, vecB) << std::endl;
}
