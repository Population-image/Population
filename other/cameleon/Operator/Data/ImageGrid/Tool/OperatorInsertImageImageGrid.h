#ifndef OPERATORINSERTIMAGEMatN_H
#define OPERATORINSERTIMAGEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/vec/Vec.h"
#include"algorithm/Draw.h"
using namespace pop;
class OperatorInsertImageMatN : public COperator
{
public:
    OperatorInsertImageMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in,BaseMatN * postit, VecF64  x, BaseMatN * &out)throw(pexception)//double rot,
        {

            if(MatN<DIM,Type> *  poistitcast = dynamic_cast<MatN<DIM,Type> * >(postit)){
                MatN<DIM,Type> * outcast =  new MatN<DIM,Type>(*in);
                VecN<DIM,pop::F64> xx;
                xx=x;
                *outcast = Draw::insertMatrix(*in,*poistitcast,xx);
                out=outcast;
            }else{
                throw(pexception("f and postit images must have the same pixel/voxel type"));
            }
        }
    };


};

#endif // OPERATORINSERTIMAGEMatN_H
