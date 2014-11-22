#ifndef OPERATORISEMPTYMatN_H
#define OPERATORISEMPTYMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
using namespace pop;
class OperatorIsEmptyMatN : public COperator
{
public:
    OperatorIsEmptyMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,bool & isempty)
        {
            MatN<DIM,Type> test(in1cast->getDomain());
            if(*in1cast==test)
                isempty =true;
            else
                isempty =false;
        }
    };

};

#endif // OPERATORISEMPTYMatN_H
