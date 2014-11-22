#ifndef OPERATORLoadRawRAWMatN_H
#define OPERATORLoadRawRAWMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"

#include"data/vec/Vec.h"
using namespace pop;
class OperatorLoadRawMatN: public COperator
{
public:
    OperatorLoadRawMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<2,Type> * in1cast,string file,VecF64  x)throw(pexception){
            in1cast->loadRaw(file.c_str(),x(0),x(1));
        }
        template<typename Type>
        void operator()(MatN<3,Type> * in1cast,string file,VecF64  x)throw(pexception){
            in1cast->loadRaw(file.c_str(),x(0),x(1),x(2));
        }
    };
};


#endif // OPERATORLoadRawRAWMatN_H
