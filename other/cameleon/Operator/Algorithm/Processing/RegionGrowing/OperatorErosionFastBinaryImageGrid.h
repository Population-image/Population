#ifndef OPERATOREROSIONFASTBINARYMatN_H
#define OPERATOREROSIONFASTBINARYMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorErosionFastBinaryMatN : public COperator
{
public:
    OperatorErosionFastBinaryMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,pop::F64 radius,pop::F64 norm,  BaseMatN * &out)
        {

            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            *outcast = Processing::erosionRegionGrowing(*in1cast,radius,norm);
            out=outcast;
        }
    };

};
#endif // OPERATOREROSIONFASTBINARYMatN_H
