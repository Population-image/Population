#include "PopulationConfig.h"

#if defined(HAVE_ACML)

#include <iostream>
#include "data/notstable/tensor/tensor.h"

Tensor::Tensor() {
}

Tensor::~Tensor() {
}

#if (PROC == INTEL)

#define copy_ scopy
#define exp_  vsExp
#define log_  vsLn
#define scal_ sscal
#define ger_  sger
#define dot_  sdot
#define gemv_ sgemv
#define gemm_ sgemm
#define axpy_ saxpy
#define pow_  vsPowx
#define tanh_ vsTanh

#elif (PROC == AMD)

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

#elif (PROC == CBLAS)

#define copy_ scopy_
#define exp_  lost_
#define log_  lost_
#define scal_ sscal_
#define ger_  sger_
#define dot_  sdotsub_
#define gemv_ sgemv_
#define gemm_ sgemm_
#define axpy_ saxpy_
#define pow_  lost_
#define tanh_ lost_

#elif (PROC == OPENBLAS)

#define copy_ scopy_
#define exp_  lost_
#define log_  lost_
#define scal_ sscal_
#define ger_  sger_
#define dot_  sdot_
#define gemv_ sgemv_
#define gemm_ sgemm_
#define axpy_ saxpy_
#define pow_  lost_
#define tanh_ lost_

#endif

floatTensor::floatTensor() {
// For Mkl, low accuracy for VML is faster
#if (PROC == INTEL)
    vmlSetMode(VML_LA | VML_FLOAT_CONSISTENT);
#endif
    size = 0;
    stride = 0;
    data = 0;
    haveMemory = 0;
    length = 0;
}

floatTensor&
floatTensor::operator =(floatTensor& src) {
    size = src.size;
    stride = src.stride;
    data = src.data;
    haveMemory = -1;
    length = src.length;
    return *this;
}

void floatTensor::array2Tensor(float* data, int size0, int size1) {
    size = new int[2];
    size[0] = size0;
    size[1] = size1;
    stride = new int[2];
    // default, by columns
    stride[0] = 1;
    stride[1] = size[0];
    this->data = data;
    haveMemory = 0;
    length = size0 * size1;
}

floatTensor::~floatTensor() {
    if (haveMemory != -1) {
        if (size != NULL) {
            delete[] size;
        }
        if (stride != NULL) {
            delete[] stride;
        }
        if (data != NULL && haveMemory) {
            delete[] data;
        }
    }
}

floatTensor::floatTensor(int size0, int size1) {
    size = new int[2];
    size[0] = size0;
    size[1] = size1;
    stride = new int[2];
    stride[0] = 1;
    stride[1] = size0;
    try {
        data = new float[size0 * size1];
    } catch (std::bad_alloc& ba) {
        std::cerr << "ERROR floatTensor resize " << ba.what() << std::endl;
    }

    haveMemory = 1;
    length = size[0] * size[1];
}

// do not use if you still want to use src_tensor later. If you want to have a new copy of src_tensor, please use this.copy(src_tensor). If you want that these 2 tensors share data, we use this = src_tensor
void floatTensor::copy_and_delete(floatTensor& src_tensor) {
    size = src_tensor.size;
    stride = src_tensor.stride;
    data = src_tensor.data;
    haveMemory = 1;
    src_tensor.haveMemory = -1;
    length = src_tensor.length;
}

void floatTensor::write() {
    std::cout << "# floatTensor" << std::endl;
    std::cout << "dimension " << size[0] << " " << size[1] << std::endl;
    for (int i = 0 ; i < length ; i ++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void floatTensor::info() {
    std::cout << "# floatTensor" << std::endl;
    std::cout << "dimension " << size[0] << " " << size[1] << std::endl;
    std::cout << "stride " << stride[0] << " " << stride[1] << std::endl;
}
int floatTensor::resize(int size0, int size1) {
    if (haveMemory == 1 && size != NULL && size[0] == size0 && size[1] == size1) {
        return 0;
    }
    if (size == NULL || stride == NULL) {
        size = new int[2];
        stride = new int[2];
    } else {
        delete[] data;
    }
    size[0] = size0;
    size[1] = size1;
    stride[0] = 1;
    stride[1] = size0;
    if (data == NULL) {
        try {
            data = new float[size0 * size1];
        } catch (std::bad_alloc& ba) {
            std::cerr << "ERROR floatTensor resize " << ba.what() << std::endl;
        }
    }

    haveMemory = 1;
    length = size[0] * size[1];
    return 1;
}

int floatTensor::resize(floatTensor& src) {
    if (haveMemory == 1 && size[0] == src.size[0] && size[1] == src.size[1] && stride[0] == src.stride[0] && stride[1] == src.stride[1]) {
        return 0;
    }
    if (size == NULL) {
        size = new int[2];
        stride = new int[2];
    } else {
        delete[] data;
    }
    size[0] = src.size[0];
    size[1] = src.size[1];
    stride[0] = src.stride[0];
    stride[1] = src.stride[1];
    try {
        data = new float[size[0] * size[1]];
    } catch (std::bad_alloc& ba) {
        std::cerr << "ERROR floatTensor resize " << ba.what() << std::endl;
    }

    haveMemory = 1;
    length = size[0] * size[1];
    return 1;
}
void floatTensor::sub(floatTensor& src, int x1, int x2, int y1, int y2) {
    haveMemory = 0;
    if (size == NULL) {
        size = new int[2];
        stride = new int[2];
    }
    size[0] = x2 - x1 + 1;
    size[1] = y2 - y1 + 1;
    stride[0] = src.stride[0];
    stride[1] = src.stride[1];
    data = &(src(x1, y1));
    length = size[0] * size[1];
}
void floatTensor::select(floatTensor& src, int sd, int sliceIndex) {
    haveMemory = 0;
    if (size == NULL) {
        size = new int[2];
        stride = new int[2];
    }
    size[0] = src.size[1 - sd];
    size[1] = 1;
    stride[0] = src.stride[1 - sd];
    stride[1] = src.stride[sd];
    if (sd) {
        data = &(src(0, sliceIndex));
    } else {
        data = &(src(sliceIndex, 0));
    }
    length = size[0];
}
//Overload =
floatTensor&
floatTensor::operator=(float value) {
    for (int i = 0 ; i < this->length ; i ++) {
        data[i] = value;
    }
    return *this;
}

void floatTensor::copy(floatTensor& src) {
    if (size == NULL) {
        resize(src);
    }
    if (length != src.length || size[0] != src.size[0]
            || size[1] != src.size[1] || stride[0] != src.stride[0] || stride[1] != src.stride[1]) {
        std::cerr << "ERROR: Copy float tensor with different size\n";
        std::cerr << length << " " << src.length << " " << size[0] << " "
                << src.size[0] << " " << size[1] << " " << src.size[1] << std::endl;
        exit(1);
    }
    for (int i = 0 ; i < length ; i ++) {
        data[i] = src.data[i];
    }
}

//void floatTensor::copy(floatTensor& src, floatTensor& weight) {
//    if (size == NULL) {
//        resize(src);
//    }
//    if (length != src.length || size[0] != src.size[0]
//            || size[1] != src.size[1]) {
//        std::cerr << "ERROR: Copy float tensor with different size\n";
//        std::cerr << length << " " << src.length << " " << size[0] << " "
//                << src.size[0] << " " << size[1] << " " << src.size[1] << std::endl;
//        exit(1);
//    }
//    for (int i = 0; i < size[0]; i++) {
//        for (int j = 0; j < size[1]; j++) {
//            data[i * stride[0] + j * stride[1]] = src.data[i * src.stride[0]
//                    + j * src.stride[1]];
//        }
//    }
//}

void floatTensor::tieData(floatTensor& src) {
    haveMemory = 0;
    delete[] data;
    if (length != src.length || size[0] != src.size[0]
            || size[1] != src.size[1]) {
        std::cerr << "ERROR: Tie tensor with different size\n";
        exit(1);
    }
    data = src.data;
}

void floatTensor::mexp(floatTensor& src) {
#if (PROC == INTEL) || (PROC == AMD)
    exp_(length, src.data, data); // y = exp(x)
#else
            for (int i = 0; i < length; i++)
            {
                data[i] = exp(src.data[i]);
            }
#endif
}

void floatTensor::mlog(floatTensor& src) {
#if (PROC == INTEL) || (PROC == AMD)
    log_(length, src.data, data); // y = log(x)
#else
            for (int i = 0; i < length; i++)
            {
                data[i] = ::log(src.data[i]);
            }
#endif
}

void floatTensor::sigm(floatTensor& src) {
    copy(src);
    scal(-1.0); // y = -x
    mexp(*this); // y = exp(-x)
    for (int i = 0; i < length; i++) {
        data[i] = 1.0 / (1.0 + data[i]);
    }
}

void floatTensor::tanh(floatTensor& src) {
#if (PROC == INTEL)
    tanh_(length, src.data, data);
#else
    int n = 1;
    //copy_(&length, src.data, &n, data, &n); //copy x to y
    copy(src);
    scal(2.0);// y = 2x
    mexp(*this);// y = exp(2x)
    for (int i = 0; i < length; i++)
    {
        data[i] = (data[i] - 1.0) / (data[i] + 1.0);
    }
#endif
}

void floatTensor::invsigm(floatTensor& src) {
#if (PROC == INTEL) || (PROC == AMD)
    float a = 2;
    pow_(length, src.data, a, data); // y = x * x
#else
            for (int i = 0; i < length; i++)
            {
                data[i] = src.data[i] * src.data[i];
            }
#endif
    scal(-1.0); // y = - x * x
    axpy(src, 1.0);
}

void floatTensor::invtanh(floatTensor& src) {
#if (PROC == INTEL) || (PROC == AMD)
    pow_(length, src.data, 2, data); // y = x * x
#else
            for (int i = 0; i < length; i++)
            {
                data[i] = src.data[i] * src.data[i];
            }
#endif
    for (int i = 0; i < length; i++) {
        data[i] = 1.0 - data[i];
    }
}

void floatTensor::softmax(floatTensor& src, floatTensor& vCol,
        floatTensor& v1row) {
    for (int i = 0; i < size[1]; i++) {
        float min = 10000000;
        for (int j = 0; j < size[0]; j++) {
            if (src(j, i) < min) {
                min = src(j, i);
            }
        }
        for (int j = 0; j < size[0]; j++) {
            src(j, i) -= min;
        }
    }
    mexp(src);
    vCol.gemv(*this, 'T', v1row, 1, 0);
    floatTensor selectThis;
    for (int i = 0; i < size[1]; i++) {
        selectThis.select(*this, 1, i);
        selectThis.scal(1.0 / vCol(i));
    }
}

void floatTensor::product(floatTensor& src) {
    for (int i = 0 ; i < length ; i ++) {
        data[i] *= src.data[i];
    }
}
void floatTensor::scal(float alpha) //y = ay
        {
    int n = 1;
    scal_(&length, &alpha, data, &n);
}

void floatTensor::axpy(floatTensor& tensor1, float alpha) //y = ax + y
        {
    int lstride;
    lstride = stride[0];
    axpy_(&length, &alpha, tensor1.data, &lstride, data, &lstride);

}

void floatTensor::ger(floatTensor& x, floatTensor& y, float alpha) // A =axyT + A
        {
    if (stride[0] == 1) {
        ger_(&x.size[0], &y.size[0], &alpha, x.data, &x.stride[0], y.data,
                &y.stride[0], data, &stride[1]);
    } else {
        ger_(&y.size[0], &x.size[0], &alpha, y.data, &y.stride[0], x.data,
                &x.stride[0], data, &stride[0]);
    }
}

void floatTensor::gemv(floatTensor& M, char transM, floatTensor& v, float alpha,
        float beta) //y = aAx + by
        {
    int ldm;
    ldm = M.size[0];

#if (PROC == INTEL) || (PROC == CBLAS) || (PROC == OPENBLAS)
    gemv_(&transM, &M.size[0], &M.size[1], &alpha, M.data, &ldm, v.data,
            &v.stride[0], &beta, data, &stride[0]);
#elif (PROC == AMD)
    gemv_(&transM, &M.size[0], &M.size[1], &alpha, M.data, &ldm, v.data,
            &v.stride[0], &beta, data, &stride[0], 1);
#endif
}

void floatTensor::gemm(floatTensor& A, char transA, floatTensor& B, char transB,
        float alpha, float beta) // C := aAB + bC
        {
    int m, n, k;
    int lda, ldb, ldc;
    m = size[0];
    n = size[1];
    ldc = m;
    if (transA == 'N') {
        k = A.size[1];
        lda = m;
    } else {
        k = A.size[0];
        lda = k;
    }
    if (transB == 'N') {
        ldb = k;
    } else {
        ldb = n;
    }

#if (PROC == INTEL) || (PROC == CBLAS) || (PROC == OPENBLAS)
    gemm_(&transA, &transB, &m, &n, &k, &alpha, A.data, &lda, B.data, &ldb,
            &beta, data, &ldc);
#elif (PROC == AMD)
    gemm_(&transA, &transB, &m, &n, &k, &alpha, A.data, &lda, B.data, &ldb,
            &beta, data, &ldc, 1, 1);
#endif
}

float floatTensor::dot(floatTensor& src) {
    if (stride[0] == src.stride[0] && stride[1] == src.stride[1]) {
        int n = stride[0];
#if (PROC == INTEL) || (PROC == AMD) || (PROC == OPENBLAS)
        return dot_(&length, src.data, &n, data, &n);
#elif (PROC == CBLAS)
        float dot;
        dot_(&length, src.data, &n, data, &n, &dot);
        return dot;
#endif
    } else {
        std::cout << "floatTensor::dot here1" << std::endl;
        float sum = 0;
        for (int i = 0 ; i < length ; i ++) {
            sum += data[i] * src.data[i];
        }
        return sum;
    }

}
//void floatTensor::read(ioFile* iof) {
//    if (READ_SHARE_W == 1) {
//        iof->readInt(haveMemory);
//    }
//    if (haveMemory == -1) {
//        cout << "WARNING: Try reading tensor with haveMemory = -1" << endl;
//        return;
//    }
//    int* sizeF = new int[2];
//    iof->readIntArray(sizeF, 2);
//    if (haveMemory == 0) {
//        cout << "WARNING: Try reading tensor with haveMemory = 0" << endl;
//        return;
//    }
//    resize(sizeF[0], sizeF[1]);
//    length = sizeF[0] * sizeF[1];
//    data = iof->readFloatArray(data, length);
//}
//void floatTensor::write(ioFile* iof) {
//    iof->writeInt(haveMemory);
//    if (haveMemory == -1) {
//        cout << "WARNING: Try writing tensor with haveMemory = -1" << endl;
//        return;
//    }
//    iof->writeIntArray(size, 2);
//    if (haveMemory == 0) {
//        cout << "WARNING: Try writing tensor with haveMemory = 0" << endl;
//        return;
//    }
//    iof->writeFloatArray(data, length);
//}

//void floatTensor::uniform(float a, float b, outils* otl) {
//    for (int i = 0; i < size[0]; i++) {
//        for (int j = 0; j < size[1]; j++) {
//            data[i * stride[0] + j * stride[1]] = a + otl->genrand() * (b - a);
//        }
//    }

//}

//void floatTensor::writeWoSize(ioFile* iof) {
//    if (haveMemory == 0) {
//        cerr << "ERROR: Try writing tensor with haveMemory = 0" << endl;
//    }
//    iof->writeFloatArray(data, length);

//}

//void floatTensor::readT(ioFile* iof) {
//    int* newSize = new int[2];
//    newSize = iof->readIntArray(newSize, 2);
//    resize(newSize[0], newSize[1]);
//    length = size[0] * size[1];
//    delete[] newSize;
//    for (int i = 0; i < size[0]; i++) {
//        for (int j = 0; j < size[1]; j++) {
//            iof->readFloat(data[i * stride[0] + j * stride[1]]);
//        }
//    }
//}

//void floatTensor::readStrip(ioFile* iof) {
//    for (int i = 0; i < size[0]; i++) {
//        for (int j = 0; j < size[1]; j++) {
//            iof->readFloat(data[i * stride[0] + j * stride[1]]);
//        }
//    }
//}

//void floatTensor::initializeNormal(outils* otl) {
//    for (int i = 0; i < this->size[0]; i++) {
//        for (int j = 0; j < this->size[1]; j++) {
//            data[i * stride[0] + j * stride[1]] = initializeNormalOneElement(
//                    otl);
//        }
//    }
//}

//float floatTensor::initializeNormalOneElement(outils* otl) {
//    // uniformly distributed float numbers
//    float U1 = otl->genrand();
//    if (U1 < 0.000001)
//        U1 = 0.000001;
//    float U2 = otl->genrand();
//    return sqrt(-2 * log(U1)) * cos(2 * M_PI * U2);
//}

float floatTensor::sumSquared() {
    return this->dot(*this);
}

float floatTensor::averageSquare() {
    return sumSquared() / (this->size[0] * this->size[1]);
}

int*
floatTensor::getSize() {
    return size;
}

int floatTensor::getSize(int i) {
    return size[i];
}

void floatTensor::setSize(int i, int value) {
    size[i] = value;
}

//float floatTensor::averageSquareBig() {
//    float sum = 0;
//    if (size[0] > size[1]) {
//        for (int i = 0; i < size[0]; i++) {
//            floatTensor select;
//            select.select(*this, 0, i);
//            sum += select.averageSquare();
//        }
//        return sum / size[0];
//    } else {
//        for (int i = 0; i < size[1]; i++) {
//            floatTensor select;
//            select.select(*this, 1, i);
//            sum += select.averageSquare();
//        }
//        return sum / size[1];
//    }
//}

//int floatTensor::testNan() {
//    if (size[0] < size[1]) {
//        for (int i = 0; i < size[1]; i++) {
//            floatTensor select;
//            select.select(*this, 1, i);
//            if (isnan(select.averageSquareBig()) != 0) {
//                return 1;
//            }
//        }
//    } else {
//        for (int i = 0; i < size[0]; i++) {
//            floatTensor select;
//            select.select(*this, 0, i);
//            if (isnan(select.averageSquareBig()) != 0) {
//                return 1;
//            }
//        }
//    }
//    return 0;
//}

//int floatTensor::testInf() {
//    for (int i = 0; i < size[0]; i++) {
//        for (int j = 0; j < size[1]; j++) {
//            int res = isinf(data[i * stride[0] + j * stride[1]]);
//            if (res != 0) {
//                std::cout << "floatTensor::testInf " << res << " " << i << " " << j
//                        << std::endl;
//                return res;
//            }
//        }
//    }
//    return 0;
//}

//int floatTensor::testNanShow() {
//    if (size[0] < size[1]) {
//        for (int i = 0; i < size[1]; i++) {
//            floatTensor select;
//            select.select(*this, 1, i);
//            if (isnan(select.averageSquare()) != 0) {
//                return 1;
//            } else {
//                std::cout << "floatTensor::testNanShow 1" << std::endl;
//                std::cout << "floatTensor::testNanShow i: " << i << std::endl;
//                std::cout << "floatTensor::testNanShow aS: "
//                        << select.averageSquare() << std::endl;
//            }
//        }
//    } else {
//        for (int i = 0; i < size[0]; i++) {
//            floatTensor select;
//            select.select(*this, 0, i);
//            if (isnan(select.averageSquare()) != 0) {
//                return 1;
//            } else {
//                std::cout << "floatTensor::testNanShow 0" << std::endl;
//                std::cout << "floatTensor::testNanShow i: " << i << std::endl;
//                std::cout << "floatTensor::testNanShow aS: "
//                        << select.averageSquare() << std::endl;
//            }
//        }
//    }
//    return 0;
//}

//float floatTensor::angleDist(floatTensor& anotherVector) {
//    if (this->size[0] != anotherVector.size[0]
//            || this->size[1] != anotherVector.size[1]) {
//        std::cout << "The two vectors do not have same size" << std::endl;
//        this->info();
//        anotherVector.info();
//        return -1000;
//    } else {
//        float sum = 0;
//        if (size[0] > size[1]) {
//            for (int i = 0; i < size[0]; i++) {
//                floatTensor select1;
//                select1.select(*this, 0, i);
//                floatTensor select2;
//                select2.select(anotherVector, 0, i);
//                sum += select1.dot(select2);
//            }
//        } else {
//            for (int i = 0; i < size[1]; i++) {
//                floatTensor select1;
//                select1.select(*this, 1, i);
//                floatTensor select2;
//                select2.select(anotherVector, 1, i);
//                sum += select1.dot(select2);
//            }
//        }
//        return sum
//                / sqrt(
//                        this->averageSquare() * this->size[0] * this->size[1]
//                                * anotherVector.averageSquare()
//                                * anotherVector.size[0] * anotherVector.size[1]);
//    }
//}

//void floatTensor::correct(outils* otl) {
//    for (int i = 0; i < size[0]; i++) {
//        for (int j = 0; j < size[1]; j++) {
//            if (isnan(data[i * stride[0] + j * stride[1]]) != 0
//                    || isinf(data[i * stride[0] + j * stride[1]]) != 0) {
//                data[i * stride[0] + j * stride[1]] =
//                        this->initializeNormalOneElement(otl);
//                cout << "floatTensor::correct " << i << " " << j << endl;
//            }
//        }
//    }
//}

void floatTensor::t() {
    int m;
    m = size[0];
    size[0] = size[1];
    size[1] = m;
    m = stride[0];
    stride[0] = stride[1];
    stride[1] = m;
}

void floatTensor::t(floatTensor &out) {
    out.resize(getSize(1), getSize(0));
    for (int j = 0 ; j < out.getSize(1) ; j ++) {
        for (int i = 0 ; i < out.getSize(0) ; i ++) {
            out(i, j) = data[i*stride[1] + j*stride[0]];
        }
    }
}

//void floatTensor::tSwap() {
//    for (int j = 0 ; j < size[1] ; j ++) {
//        for (int i = 0 ; i < size[0] ; i ++) {
//            swap(&data[i*stride[0] + j*stride[1]], data[i])
//        }
//    }
//}

#endif // HAVE_ACML
