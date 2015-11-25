#include "algorithm/processingtensor.h"
#include <cmath>

floatTensor ProcessingTensor::smoothDeriche(floatTensor & f, double alpha){
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

    floatTensor fCurrent(f), g1(f), g2(f);
    fCurrent = 0;
    g2 = 0;
    funccausal0.transform(f, fCurrent);
    funcanticausal0.transform(f, g2);
    fCurrent.axpy(g2, 1);
    g1 = 0; g2 = 0;
    funccausal1.transform(fCurrent, g1);
    funcanticausal1.transform(fCurrent, g2);
    g1.axpy(g2, 1);
    fCurrent.copy(g1);
    return fCurrent;
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
        g = 0;
    }
    int lengthProcess = f.getSize(1 - _slide);
    int firstIte, lastIte;
    if (_dir == 1) firstIte = 0, lastIte = lengthProcess;
    else firstIte = lengthProcess - 1, lastIte = -1;
    floatTensor sub_f, sub_f1, sub_f2, sub_g, sub_g1, sub_g2;
    for (unsigned int i = firstIte ; i != lastIte ; i += _dir) {
        sub_f.select(f, 1 - _slide, i);
        sub_g.select(g, 1 - _slide, i);
        if (std::abs(i - firstIte) >= 1) {
            sub_f1.select(f, 1 - _slide, i - _dir);
            sub_g1.select(g, 1 - _slide, i - _dir);
        }
        if (std::abs(i - firstIte) >= 2) {
            sub_f2.select(f, 1 - _slide, i - 2*_dir);
            sub_g2.select(g, 1 - _slide, i - 2*_dir);
        }
        if (i == firstIte) {
            sub_g.axpy(sub_f, _a0b0);
        } else if (std::abs(i - firstIte) == 1) {
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
}
