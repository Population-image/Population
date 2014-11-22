#ifndef OPERATORSPECIFICPerimeterAREAMatN_H
#define OPERATORSPECIFICPerimeterAREAMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorPerimeterMatN : public COperator
{
public:
    OperatorPerimeterMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,Mat2F64*&m){
             m  = new Mat2F64;
            *m  = Analysis::perimeter(* in1cast);
        }
    };

};

#endif // OPERATORSPECIFICPerimeterAREAMatN_H
