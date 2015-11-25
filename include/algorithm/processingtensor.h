#ifndef PROCESSINGTENSOR_H
#define PROCESSINGTENSOR_H

#include "tensor/tensor.h"

struct ProcessingTensor {
    static floatTensor smoothDeriche(floatTensor & f, double alpha=1);
};

namespace tensor {
    class FunctorRecursiveOrder2 {
    private:
        double _a0, _a1, _a2, _a0b0, _a0b1, _a1b1;
        double _b1, _b2, _b1b1;
        int _slide;
        int _dir;
    public:
        FunctorRecursiveOrder2(double a0, double a1, double a2,
                               double b1, double b2,
                               double a0b0,
                               double a0b1, double a1b1, double b1b1, int slide, int dir);
        void transform(floatTensor& f, floatTensor& g);
    };
}

#endif // PROCESSINGTENSOR_H

