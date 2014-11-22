#ifndef OPERATORHITORMISSMatN_H
#define OPERATORHITORMISSMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
#include"data/mat/MatNListType.h"
#include"data/GP/CartesianProduct.h"
#include"data/GP/Dynamic2Static.h"
using namespace pop;
class OperatorHitOrMissMatN : public COperator
{
public:
    OperatorHitOrMissMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,BaseMatN * C,BaseMatN * D,  BaseMatN * &out)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGridScalar,0,Loki::Int2Type<DIM> >::Result ListFilter;
            foo2 func;
            try{Dynamic2Static<ListFilter>::Switch(func,C,in1cast,D,out,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }
        }
    };
    struct foo2
    {
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * Ccast,MatN<DIM,Type2> * in1cast,BaseMatN * D, BaseMatN * &out)throw(pexception)
        {
            if(MatN<DIM,Type1> * Dcast=dynamic_cast<MatN<DIM,Type1>*>(D)){

                MatN<DIM,Type2> * outcast = new MatN<DIM,Type2>(in1cast->getDomain());
               *outcast =Processing::hitOrMiss(* in1cast,*Ccast,*Dcast);
                out=outcast;
            }else{
                throw(std::string("C and D must have the same pixel/voxel type"));
            }

        }
    };

};
#endif // OPERATORHITORMISSMatN_H
