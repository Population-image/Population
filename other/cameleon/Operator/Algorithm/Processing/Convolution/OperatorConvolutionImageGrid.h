#ifndef OPERATORCONVOLUTIONMatN_H
#define OPERATORCONVOLUTIONMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"data/mat/MatNListType.h"
#include"data/GP/CartesianProduct.h"
#include"data/GP/Dynamic2Static.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorConvolutionMatN : public COperator
{
public:
    OperatorConvolutionMatN();
    void exec();
    COperator * clone();
    struct foo2
    {
        template<int DIM,typename Type1,typename Type2>
        void operator()(MatN<DIM,Type2> * kernel, MatN<DIM,Type1> * f,BaseMatN * &out)throw(pexception)
        {
            MatN<DIM,Type1> * outcast = new MatN<DIM,Type1>  (f->getDomain());
            *outcast= Processing::convolution( * f,* kernel);

            out=outcast ;
        }
    };
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * f,BaseMatN* kernel,  BaseMatN * &out)throw(pexception)
        {


            typedef typename FilterKeepTlistTlist<TListImgGridFloat,0,Loki::Int2Type<DIM> >::Result ListFilter;

           OperatorConvolutionMatN::foo2 func;
            try{Dynamic2Static<ListFilter>::Switch(func,kernel,f,out,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg)
            {
                throw(pexception("Pixel type of kernel image must be Float"));
            }

        }
    };

};
#endif // OPERATORCONVOLUTIONMatN_H
