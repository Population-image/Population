#ifndef OPERATORGEODESICReconstructionMatN_H
#define OPERATORGEODESICReconstructionMatN_H
#include<limits>
using namespace std;

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
#include"data/mat/MatNListType.h"
#include"data/GP/CartesianProduct.h"
#include"data/GP/Dynamic2Static.h"
using namespace pop;
class OperatorGeodesicReconstructionMatN : public COperator
{
public:
    OperatorGeodesicReconstructionMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * fcast,BaseMatN * g, int norm,BaseMatN * &h)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<DIM> >::Result ListFilter;

            foo2 func;
            try{Dynamic2Static<ListFilter>::Switch(func,g,fcast,norm,h,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }
        }
    };
    struct foo2
    {
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * gcast,MatN<DIM,Type2> * fcast, int norm,BaseMatN * &h)
        {
            MatN<DIM,Type2> * hcast =  new MatN<DIM,Type2>;
            *hcast = Processing::geodesicReconstruction(*fcast,*gcast,norm);
            h=hcast;
        }
    };

};

#endif // OPERATORGEODESICReconstructionMatN_H
