#ifndef OPERATORWHITEFACEMatN_H
#define OPERATORWHITEFACEMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/vec/Vec.h"
#include"algorithm/Draw.h"
using namespace pop;
class OperatorWhiteFaceMatN : public COperator
{
public:
    OperatorWhiteFaceMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in,int coordinate, int face,BaseMatN * &out){
            MatN<DIM,Type> * outcast =  new MatN<DIM,Type>(*in);
            * outcast = Draw::faceWhite(* in,coordinate,face);
            out = outcast;
        }
    };
};
#endif // OPERATORWHITEFACEMatN_H
