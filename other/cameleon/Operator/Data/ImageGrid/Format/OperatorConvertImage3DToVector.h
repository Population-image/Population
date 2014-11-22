#ifndef OPERATORCONVERTIMAGE3DTOVECTOR_H
#define OPERATORCONVERTIMAGE3DTOVECTOR_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;


class OperatorConvertImage3DToVector  : public COperator
{
public:
    OperatorConvertImage3DToVector();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,vector<BaseMatN *> &out)throw(pexception)
        {
            for(int i =0;i<in1cast->getDomain()(DIM-1);i++)
            {
                MatN<DIM-1,Type> * plane = new MatN<DIM-1,Type>(in1cast->getPlaneDomain(DIM-1));
                in1cast->setPlane(DIM-1,i,*plane);
                out.push_back(plane);
            }

        }
    };

};


#endif // OPERATORCONVERTIMAGE3DTOVECTOR_H
