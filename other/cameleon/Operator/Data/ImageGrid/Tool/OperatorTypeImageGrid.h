#ifndef OPERATORTYPEMatN_H
#define OPERATORTYPEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
using namespace pop;
class OperatorTypeMatN : public COperator
{
public:
    OperatorTypeMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * ,string & type)
        {

            type = Type2Id<MatN<DIM,Type> >::id[1];
        }
    };
};

#endif // OPERATORTYPEMatN_H
