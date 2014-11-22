#ifndef OPERATORGETPLANEMatN_H
#define OPERATORGETPLANEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
using namespace pop;
class OperatorGetPlaneMatN : public COperator
{
public:
    OperatorGetPlaneMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int index, int coordinate, BaseMatN * &out)
        {
            if(coordinate==-1)
                coordinate=DIM-1;
            VecN<DIM-1,int> domain = in1cast->getPlaneDomain(coordinate);
            MatN<DIM-1,Type> * outcast = new MatN<DIM-1,Type>(domain);
            *outcast = in1cast->getPlane(coordinate,index);
            out=outcast;
        }
    };
};

#endif // OPERATORGETPLANEMatN_H
