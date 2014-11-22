#ifndef OPERATORADAMSBISCHOFMatN_H
#define OPERATORADAMSBISCHOFMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
#include"data/mat/MatNListType.h"
#include"data/GP/CartesianProduct.h"
#include"data/GP/Dynamic2Static.h"
using namespace pop;
class OperatorAdamsBischofMatN : public COperator
{
public:
    OperatorAdamsBischofMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * seedcast,BaseMatN * topo, int norm,int mode, BaseMatN * &region)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<DIM> >::Result ListFilter;
            foo2 func;
            try{Dynamic2Static<ListFilter>::Switch(func,topo,seedcast,norm,mode,region,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }
        }
    };
    struct foo2
    {
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * topocast,MatN<DIM,Type2> * seedcast, int norm,int mode,BaseMatN * &region)
        {
            MatN<DIM,Type2> *regioncast =  new MatN<DIM,Type2>;
            if(mode==0)
                (*regioncast) = Processing::regionGrowingAdamsBischofMeanOverStandardDeviation(*seedcast,*topocast,norm);
            else
                (*regioncast) = Processing::regionGrowingAdamsBischofMean(*seedcast,*topocast,norm);

            region=regioncast;

        }
    };


};

#endif // OPERATORADAMSBISCHOFMatN_H
