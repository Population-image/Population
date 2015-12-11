#include "PopulationConfig.h"

#if defined(HAVE_ACML)

#include "data/notstable/tensor/processingtensor.h"
#include <cmath>
#include<chrono>

void ProcessingTensor::smoothDeriche(floatTensor & f, floatTensor& fCurrent, pop::F32 alpha){
    double e_a = std::exp(- alpha);
    double e_2a = std::exp(- 2.f * alpha);
    double k = (1.f - e_a) * (1.f - e_a) / (1.f + (2 * alpha * e_a) - e_2a);

    double a0_c= k;
    double a1_c=  k * e_a * (alpha - 1.f);
    double a2_c=  0;
    double a0_ac= 0;
    double a1_ac=  k * e_a * (alpha + 1.f);
    double a2_ac=  - k * e_2a;

    double b1= 2 * e_a;
    double b2 = - e_2a;

    double a0_c_border0 = ((a0_c + a1_c) / (1.f - b1 - b2));
    double a0_c_border1 = a0_c ;
    double a1_c_border1 = a1_c ;

    double a0_ac_border0 = ((a1_ac + a2_ac) / (1.f - b1 - b2));
    double a0_ac_border1 = 0 ;
    double a1_ac_border1 = a1_ac + a2_ac ;

    double b1_border1 = b1 + b2 ;

    tensor::FunctorRecursiveOrder2 funccausal0
            (a0_c,a1_c,a2_c,
             b1,b2,
             a0_c_border0,
             a0_c_border1,a1_c_border1,b1_border1, 0, 1);

    tensor::FunctorRecursiveOrder2 funcanticausal0
            (a0_ac,a1_ac,a2_ac,
             b1,b2,
             a0_ac_border0,
             a0_ac_border1,a1_ac_border1,b1_border1, 0, -1);

    tensor::FunctorRecursiveOrder2 funccausal1
            (a0_c,a1_c,a2_c,
             b1,b2,
             a0_c_border0,
             a0_c_border1,a1_c_border1,b1_border1, 1, 1);

    tensor::FunctorRecursiveOrder2 funcanticausal1
            (a0_ac,a1_ac,a2_ac,
             b1,b2,
             a0_ac_border0,
             a0_ac_border1,a1_ac_border1,b1_border1, 1, -1);

    floatTensor g1(f.getSize(0), f.getSize(1)), g2(f.getSize(0), f.getSize(1)), g3(f.getSize(1), f.getSize(0)), g4(f.getSize(1), f.getSize(0));
    fCurrent.resize(f.getSize(0), f.getSize(1));

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    funccausal0.transform(f, fCurrent);
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    funcanticausal0.transform(f, g2);
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    fCurrent.axpy(g2, 1);
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    g1 = 0;
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    g2 = 0;
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    funccausal1.transform(fCurrent, g1);
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    funcanticausal1.transform(fCurrent, g2);
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    g1.axpy(g2, 1);
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    fCurrent.copy(g1);
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;
}

void ProcessingTensor::pop2tensor(const pop::Mat2UI8& img, floatTensor& out) {
    out.resize(img.rows(), img.columns());
    for (unsigned int j = 0 ; j < out.getSize(1) ; j ++) {
        for (unsigned int i = 0 ; i < out.getSize(0) ; i ++) {
            out(i, j) = (float)(img(i, j));
        }
    }
}

pop::Mat2UI8 ProcessingTensor::tensor2pop(floatTensor& img) {
    pop::Mat2UI8 out(img.getSize(0), img.getSize(1));
    for (unsigned int i = 0 ; i < img.getSize(0) ; i ++) {
        for (unsigned int j = 0 ; j < img.getSize(1) ; j ++) {
            out(i, j) = (unsigned char)(img(i, j));
        }
    }
    return out;
}

pop::Mat2UI8 ProcessingTensor::thresoldAdaptiveSmoothDeriche(floatTensor& f, pop::F32 sigma,pop::F32 offset_value) {
    floatTensor smooth;
    pop::Mat2UI8 threshold(f.getSize(0), f.getSize(1));
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    ProcessingTensor::smoothDeriche(f, smooth, sigma);
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " smoothDerich finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int j = 0 ; j < smooth.getSize(1) ; j ++) {
        for (int i = 0 ; i < smooth.getSize(0) ; i ++) {
            if(f(i, j) > smooth(i, j)-offset_value){
                threshold(i, j)=255;
            }else{
                threshold(i, j)=0;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << " smoothDerich finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;
    return threshold;
}

pop::Mat2UI8 ProcessingTensor::thresoldAdaptiveSmoothDeriche(const pop::Mat2UI8 &f, pop::F32 sigma, pop::F32 offset_value) {
    std::chrono::time_point<std::chrono::system_clock> start1, end1;
    floatTensor f_tensor;
    start1 = std::chrono::high_resolution_clock::now();
    ProcessingTensor::pop2tensor(f, f_tensor);
    end1 = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << "finish after " << std::chrono::duration<double, std::milli>(end1-start1).count() << std::endl;

    pop::Mat2UI8 threshold = ProcessingTensor::thresoldAdaptiveSmoothDeriche(f_tensor, sigma, offset_value);
    return threshold;
}

tensor::FunctorRecursiveOrder2::FunctorRecursiveOrder2(double a0, double a1, double a2,
                                   double b1, double b2,
                                   double a0b0,
                                   double a0b1, double a1b1, double b1b1, int slide, int dir) :
    _a0(a0), _a1(a1), _a2(a2), _b1(b1), _b2(b2), _a0b0(a0b0), _a0b1(a0b1), _a1b1(a1b1), _b1b1(b1b1), _slide(slide), _dir(dir) {

}

void tensor::FunctorRecursiveOrder2::transform(floatTensor &f, floatTensor &g) {
    if (g.getSize(0) != f.getSize(0) || g.getSize(1) != f.getSize(1)) {
        g.resize(f);
    }
    int firstIte, lastIte, lengthProcess;
    lengthProcess = f.getSize(1);
    if (_dir == 1) {
        firstIte = 0;
        lastIte = lengthProcess;
    } else {
        firstIte = lengthProcess - 1;
        lastIte = -1;
    }
    floatTensor sub_f, sub_g;
    if (_slide == 0) {
        floatTensor sub_f1, sub_f2, sub_g1, sub_g2;
        for (int i = firstIte ; i != lastIte ; i += _dir) {
            sub_f.select(f, 1 - _slide, i);
            sub_g.select(g, 1 - _slide, i);
            if (std::abs((int)(i) - firstIte) >= 1) {
                sub_f1.select(f, 1 - _slide, i - _dir);
                sub_g1.select(g, 1 - _slide, i - _dir);
            }
            if (std::abs((int)(i) - firstIte) >= 2) {
                sub_f2.select(f, 1 - _slide, i - 2*_dir);
                sub_g2.select(g, 1 - _slide, i - 2*_dir);
            }
            if (i == firstIte) {
                sub_g.axpy(sub_f, _a0b0);
            } else if (std::abs((int)(i) - firstIte) == 1) {
                sub_g.axpy(sub_f, _a0b1);
                sub_g.axpy(sub_f1, _a1b1);
                sub_g.axpy(sub_g1, _b1b1);
            } else {
                sub_g.axpy(sub_f, _a0);
                sub_g.axpy(sub_f1, _a1);
                sub_g.axpy(sub_g1, _b1);
                sub_g.axpy(sub_f2, _a2);
                sub_g.axpy(sub_g2, _b2);
            }
        }
    } else {
        int lengthSubProcess = f.getSize(0), subFirstIte, subLastIte;
        if (_dir == 1) {
            subFirstIte = 0;
            subLastIte = lengthSubProcess;
        } else {
            subFirstIte = lengthSubProcess - 1;
            subLastIte = -1;
        }
        for (int k = firstIte ; k != lastIte ; k += _dir) {
            sub_f.select(f, 1, k);
            sub_g.select(g, 1, k);
            for (int i = subFirstIte ; i != subLastIte ; i += _dir) {
                if (i == firstIte) {
                    sub_g(i) = sub_f(i) * _a0b0;
                } else if (std::abs((int)(i) - firstIte) == 1) {
                    sub_g(i) = sub_g(i - _dir) * _b1b1 + sub_f(i) * _a0b1 + sub_f(i - _dir) * _a1b1;
                } else {
                    sub_g(i) = sub_g(i - _dir) * _b1 + sub_g(i - 2 * _dir) * _b2 + sub_f(i) * _a0 + sub_f(i - _dir) * _a1 + sub_f(i - 2 * _dir) * _a2;
                }
            }
        }
    }
}

void tensor::FunctorRecursiveOrder2::fastTransform(floatTensor &f, floatTensor &g) {
    if (g.getSize(0) != f.getSize(0) || g.getSize(1) != f.getSize(1)) {
        g.resize(f);
    }
    floatTensor sub_g, sub_f, sub_f1, sub_f2;
    if (_dir == 1) {
        sub_g.sub(g, 0, f.getSize(0) - 1, 2, f.getSize(1) - 1);
        sub_f.sub(f, 0, f.getSize(0) - 1, 2, f.getSize(1) - 1);
        sub_f1.sub(f, 0, f.getSize(0) - 1, 1, f.getSize(1) - 2);
        sub_f2.sub(f, 0, f.getSize(0) - 1, 0, f.getSize(1) - 3);
    } else {
        sub_g.sub(g, 0, f.getSize(0) - 1, 0, f.getSize(1) - 3);
        sub_f.sub(f, 0, f.getSize(0) - 1, 0, f.getSize(1) - 3);
        sub_f1.sub(f, 0, f.getSize(0) - 1, 1, f.getSize(1) - 2);
        sub_f2.sub(f, 0, f.getSize(0) - 1, 2, f.getSize(1) - 1);
    }
    sub_g.axpy(sub_f, this->_a0);
    sub_g.axpy(sub_f1, this->_a1 + this->_b1);
    sub_g.axpy(sub_f2, this->_a2 + this->_b2);
}

#endif // HAVE_ACML
