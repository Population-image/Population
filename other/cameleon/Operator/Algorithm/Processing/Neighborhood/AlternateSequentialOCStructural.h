#ifndef OPERATORALTERNATESEQUENTIALOCSTRUCTURALMatN_H
#define OPERATORALTERNATESEQUENTIALOCSTRUCTURALMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
#include"data/mat/MatNListType.h"
#include"data/GP/CartesianProduct.h"
#include"data/GP/Dynamic2Static.h"
using namespace pop;
class OperatorAlternateSequentialOCStructuralMatN : public COperator
{
public:
    OperatorAlternateSequentialOCStructuralMatN();
    void exec();
    COperator * clone();

    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int max_radius, BaseMatN * struc_elt,  BaseMatN * &out)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGridScalar,0,Loki::Int2Type<DIM> >::Result ListFilter;
            foo2 func;
            try{Dynamic2Static<ListFilter>::Switch(func,struc_elt,in1cast,max_radius,out,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }
        }
    };
    struct foo2
    {
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * struc_elt,MatN<DIM,Type2> * in1cast,int max_radius, BaseMatN * &out)
        {
            MatN<DIM,Type2> * outcast = new MatN<DIM,Type2>(in1cast->getDomain());
            *outcast = Processing::alternateSequentialOCStructuralElement(*in1cast,max_radius,* struc_elt);
            out=outcast;

        }
    };

};

#endif // OPERATORALTERNATESEQUENTIALOCSTRUCTURALMatN_H
