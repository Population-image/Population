#ifndef OPERATORFROMMatNMATRIX_H
#define OPERATORFROMMatNMATRIX_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatN.h"
using namespace pop;
class OperatorConvertFromMatNMatrix: public COperator
{
public:
    OperatorConvertFromMatNMatrix();
    virtual void exec();
    virtual COperator * clone();

    struct foo
    {
        template<typename Type>
        void operator()(MatN<2,Type> * in1cast,Mat2F64* &m)throw(pexception){

            m= new Mat2F64(in1cast->getDomain()(1),in1cast->getDomain()(0));
            for(int i=0;i<m->sizeI();i++)
                for(int j=0;j<m->sizeJ();j++)
                {
                    m->operator ()(i,j) = in1cast->operator ()(j,i);
                }

        }
    };
};

#endif // OPERATORFROMMatNMATRIX_H
