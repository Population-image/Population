#ifndef OPERATORMATHERONGRANULOMETRYMatN_H
#define OPERATORMATHERONGRANULOMETRYMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorMatheronGranulometryMatN : public COperator
{
public:
    OperatorMatheronGranulometryMatN();
    void exec();
    COperator * clone();
        void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int norm,Mat2F64*& m,BaseMatN * &out){
            m  = new Mat2F64;
            MatN<DIM,unsigned char> * outcast = new MatN<DIM,unsigned char>(in1cast->getDomain());
            *m = Analysis::granulometryMatheron(*in1cast,norm,*outcast);
            out=outcast;
        }
    };

};

#endif // OPERATORMATHERONGRANULOMETRYMatN_H
