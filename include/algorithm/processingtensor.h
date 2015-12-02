#ifndef PROCESSINGTENSOR_H
#define PROCESSINGTENSOR_H

#include "PopulationConfig.h"

#if defined(HAVE_ACML)

#include "tensor/tensor.h"
#include "Population.h"

struct ProcessingTensor {
    static void smoothDeriche(floatTensor & f, floatTensor& out, pop::F32 alpha=1);
    static void pop2tensor(const pop::Mat2UI8& img, floatTensor& out);
    static pop::Mat2UI8 tensor2pop(floatTensor& img);
    static pop::Mat2UI8 thresoldAdaptiveSmoothDeriche(floatTensor& f, pop::F32 sigma=0.5, pop::F32 offset_value=0);
    static pop::Mat2UI8 thresoldAdaptiveSmoothDeriche(const pop::Mat2UI8& f, pop::F32 sigma=0.5, pop::F32 offset_value=0);
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
        void fastTransform(floatTensor& f, floatTensor& g);
    };
}

#endif // HAVE_ACML

#endif // PROCESSINGTENSOR_H

