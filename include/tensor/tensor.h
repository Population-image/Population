#ifndef TENSOR_H
#define TENSOR_H

#define INTEL    0
#define AMD      1
#define CBLAS    2
#define OPENBLAS 3


#if (PROC == INTEL)

//#include "mkl_blas.h"
//#include "mkl_vml.h"

#elif (PROC == AMD)

#include "acml.h"
//#include "acml_mv.h"
#include "amdlibm.h"

#elif (PROC == CBLAS)

#include "cblas_f77.h"

#elif (PROC == OPENBLAS)

#include "f77blas.h"

#endif

//#include "ioFile.H"
//#include "outils.H"
//#include "Tensor.H"
//#include "floatTensor.H"
//#include "intTensor.H"

class Tensor
{
public:
    //Data
    // haveMemory
    //  1: Free all: data, size, stride,
    //  0: Don't need to free data,
    // -1: Don't free anything
    int haveMemory;
    // Total memory size for data
    int length;
    Tensor();
    ~Tensor();

};

class floatTensor: public Tensor {
protected:
    // Data
    int* size;
    int* stride;
    float* data;

public:
    // Method
    floatTensor();
    ~floatTensor();
    floatTensor(int size0, int size1);

    // do not use if you still want to use src_tensor later. If you want to have a new copy of src_tensor, please use this.copy(src_tensor). If you want that these 2 tensors share data, we use this = src_tensor
    // equivalent to the move constructor (c++11) which is not supported by the current compiler
    void
    copy_and_delete(floatTensor& src_tensor);
    // Write to stdout
    void
    write();
    // Write information to stdout
    void
    info();
    // Set, get values
    float&
    operator()(int x);
    float
    operator()(int x) const;

    float&
    operator()(int x, int y);
    float
    operator()(int x, int y) const;

    // Resize size
    int
    resize(int size0, int size1);
    int
    resize(floatTensor& src);

    // Manipulate data
    // Select, sub, copy, tie
    void
    sub(floatTensor& src, int x1, int x2, int y1, int y2);
    void
    select(floatTensor& src, int sd, int sliceIndex);
    void
    copy(floatTensor& src);
//    void
//    copy(floatTensor&, floatTensor&);
    void
    tieData(floatTensor& src);

    // Overload =
    // Important, so don't modify
    floatTensor&
    operator=(float value);
    floatTensor&
    operator=(floatTensor &src);
    void
    array2Tensor(float* data, int size0, int size1);

    // Math simple function like sigm, tanh
    // invsigm, invtanh are backward function of sigm, tanh in neural network
    // Two tensors have the same dimension, size, stride and at least one stride = 1
    void
    mexp(floatTensor& src);
    void
    mlog(floatTensor& src);

    void
    sigm(floatTensor& src);
    void
    tanh(floatTensor& src);
    void
    invsigm(floatTensor& src);
    void
    invtanh(floatTensor& src);
    void
    softmax(floatTensor& src, floatTensor &vCol, floatTensor &v1row);

    // y = x * y (element wise)
    void
    product(floatTensor& src);
    // Math matrix vector BLAS
    // y = ay
    void
    scal(float alpha);
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

//    float
//    averageSquareBig();

//    int
//    testNan();

//    int
//    testInf();

//    int
//    testNanShow();

    int*
    getSize();

    int
    getSize(int i);

    void
    setSize(int i, int value);

//    float
//    initializeNormalOneElement(outils* otl);

//    void
//    initializeNormal(outils* otl);

    // Read write function,
    // when haveMemory = 0, we don't write it, so when read it,
    // we must use the same code to create the pointer data.
    // e.g. lbl model, weightLinear.t(weigthLookuptable):
    // we save only weigthLookuptable in reality.

    // Read in default order (column major order)
    // In file: header: 2 4, then data: 1 2 3 4 5 6 7 8
    // tensor is a matrix 2 x 4, we will have
    // tensor.data = [1, 2, 3, 4, 5, 6, 7, 8] (as in file)
    // it represents a matrix:
    // 1 3 5 7
    // 2 4 6 8
//    void
//    read(ioFile* iof);

    //Read in row major order
    // In file: header: 2 4, then data: 1 2 3 4 5 6 7 8
    // tensor is a matrix 2 x 4, we will have
    // tensor.data = [1, 5, 2, 6, 3, 7, 4, 8]
    // it represents a matrix:
    // 1 2 3 4
    // 5 6 7 8
//    void
//    readT(ioFile* iof);

    // Read in row major order but without header
    // In file 1 2 3 4 5 6 7 8
    // tensor is pre-defined as a matrix 2 x 4,
    // tensor.data = [1, 5, 2, 6, 3, 7, 4, 8]
    // after reading, it represents a matrix:
    // 1 2 3 4
    // 5 6 7 8
//    void
//    readStrip(ioFile* iof);

//    void
//    write(ioFile* iof);
//    void
//    writeWoSize(ioFile* iof);

    // Sample from uniform distribution
//    void
//    uniform(float a, float b, outils* otl);

    // calculate angle distance
    float
    angleDist(floatTensor& anotherVector);

//    void
//    correct(outils* otl);

    // transpose
    void
    t();
    void t(floatTensor& out);
    //void tSwap();

};

inline float&
floatTensor::operator()(int x, int y) {
    return data[x * stride[0] + y * stride[1]];
}

inline float floatTensor::operator()(int x, int y) const {
    return data[x * stride[0] + y * stride[1]];
}

inline float&
floatTensor::operator()(int x) {
    return data[x];
}

inline float floatTensor::operator()(int x) const {
    return data[x];
}

#endif // TENSOR_H
