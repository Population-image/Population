#ifndef OPERATORCONVERTRGB2COLORMatN_H
#define OPERATORCONVERTRGB2COLORMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;

class OperatorConvertRGB2ColorMatN : public COperator
{
public:
    OperatorConvertRGB2ColorMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type > * rcast, BaseMatN * g,BaseMatN * b,BaseMatN * &color)throw(pexception)
        {
            MatN<DIM,RGB<Type> > * colorcast = new  MatN<DIM,RGB<Type> >(rcast->getDomain());
            if( MatN<DIM,Type > * gcast= dynamic_cast<MatN<DIM,Type > *>(g))
            {
                if(MatN<DIM,Type > * bcast=  dynamic_cast<MatN<DIM,Type > *>(b) )
                {
                    Convertor::fromRGB(* rcast,*gcast,*bcast,*colorcast);
                }
                else
                {
                    throw(pexception("Pixel/voxel type of input image must be 1Byte"));
                }

            }
            else
            {
                throw(pexception("Pixel/voxel type of input image must be 1Byte"));
            }

            color = colorcast;
        }
    };

};

#endif // OPERATORCONVERTRGB2COLORMatN_H
