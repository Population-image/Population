#ifndef OPERATORCONVERTCOLOR2RGBMatN_H
#define OPERATORCONVERTCOLOR2RGBMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvertColor2RGBMatN : public COperator
{
public:
    OperatorConvertColor2RGBMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGB<Type> > * in1cast, BaseMatN * &r,BaseMatN * &g,BaseMatN * &b)
        {

              MatN<DIM,Type> * rcast = new MatN<DIM,Type>(in1cast->getDomain());
              MatN<DIM,Type> * gcast = new MatN<DIM,Type>(in1cast->getDomain());
              MatN<DIM,Type> * bcast = new MatN<DIM,Type>(in1cast->getDomain());
              Convertor::toRGB(* in1cast,*rcast,*gcast,*bcast);
              r =rcast;
              g =gcast;
              b =bcast;
        }
    };

};

#endif // OPERATORCONVERTCOLOR2RGBMatN_H
