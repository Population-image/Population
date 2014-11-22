#ifndef OPERATORWATERSHEDMatN_H
#define OPERATORWATERSHEDMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
#include"data/mat/MatNListType.h"
#include"data/GP/CartesianProduct.h"
#include"data/GP/Dynamic2Static.h"
using namespace pop;
class OperatorWatershedMatN : public COperator
{
public:
    OperatorWatershedMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * seedcast,BaseMatN * topo, int norm,BaseMatN * &water)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<DIM> >::Result ListFilter;

            foo2 func;
            try{Dynamic2Static<ListFilter>::Switch(func,topo,seedcast,norm,water,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * seedcast,BaseMatN * topo,BaseMatN * mask, int norm,BaseMatN * &water)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<DIM> >::Result ListFilter;

            foo2 func;
            try{Dynamic2Static<ListFilter>::Switch(func,topo,seedcast,mask,norm,water,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }


        }
    };
    struct foo2
    {
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * topocast,MatN<DIM,Type2> * seedcast, int norm,BaseMatN * &water)
        {
            MatN<DIM,Type2> *watercast =  new MatN<DIM,Type2>;
            *watercast = Processing::watershed(*seedcast,*topocast,  norm);
            water=watercast;
        }
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * topocast,MatN<DIM,Type2> * seedcast,BaseMatN * mask, int norm,BaseMatN * &water)throw(pexception)
        {
            if(MatN<DIM,unsigned char> * maskcast = dynamic_cast<MatN<DIM,unsigned char> *>(mask) ){
                MatN<DIM,Type2> *watercast =  new MatN<DIM,Type2>;
                *watercast = Processing::watershed(*seedcast,*topocast,*maskcast,  norm);
                water=watercast;
            }
            else{
                throw(pexception("The pixel/voxel type of the mask must be unsigned 1Byte"));
            }
        }
    };
    struct foob
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * seedcast,BaseMatN * topo, int norm,BaseMatN * &water)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<DIM> >::Result ListFilter;
            foo2b func;
            try{Dynamic2Static<ListFilter>::Switch(func,topo,seedcast,norm,water,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * seedcast,BaseMatN * topo,BaseMatN * mask, int norm,BaseMatN * &water)throw(pexception)
        {
            typedef typename FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<DIM> >::Result ListFilter;

            foo2b func;
            try{Dynamic2Static<ListFilter>::Switch(func,topo,seedcast,mask,norm,water,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception(msg));
            }
        }
    };
    struct foo2b
    {
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * topocast,MatN<DIM,Type2> * seedcast, int norm,BaseMatN * &water)
        {
            MatN<DIM,Type2> *watercast =  new MatN<DIM,Type2>;
            *watercast = Processing::watershedBoundary(*seedcast,*topocast, norm);
            water=watercast;
        }
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type1> * topocast,MatN<DIM,Type2> * seedcast,BaseMatN * mask, int norm,BaseMatN * &water)throw(pexception)
        {
            if(MatN<DIM,unsigned char> * maskcast = dynamic_cast<MatN<DIM,unsigned char> *>(mask) ){
                MatN<DIM,Type2> *watercast =  new MatN<DIM,Type2>;
                *watercast = Processing::watershedBoundary(*seedcast,*topocast,*maskcast,  norm);
                water=watercast;
            }
            else{
                throw(pexception("The pixel/voxel type of the mask must be unsigned 1Byte"));
            }
        }
    };

};


#endif // OPERATORWATERSHEDMatN_H
