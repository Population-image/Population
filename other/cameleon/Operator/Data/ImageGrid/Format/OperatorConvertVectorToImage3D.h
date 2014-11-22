#ifndef OPERATORCONVERTVECTORTOIMAGE3D_H
#define OPERATORCONVERTVECTORTOIMAGE3D_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Visualization.h"
using namespace pop;


class OperatorConvertVectorToImage3D : public COperator
{
public:
    OperatorConvertVectorToImage3D();
    void exec();
    COperator * clone();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<2,Type> * in1cast,vector<BaseMatN *> v_in, BaseMatN * &out)throw(pexception){
            VecN<3,int> domain;
            for(int i = 0;i<2;i++){
                domain(i)= in1cast->getDomain()(i);
            }
            domain(2)=(int)v_in.size();
            MatN<3,Type> * outcast = new MatN<3,Type>(domain);
            for(int i =0;i<(int)v_in.size();i++){
                if(MatN<2,Type> * in = dynamic_cast<MatN<2,Type> *>(v_in[i]) ){
                    outcast->setPlane(2,i,* in);
                }
            }
            out = outcast;
        }
    };

};

#endif // OPERATORCONVERTVECTORTOIMAGE3D_H
