#ifndef OPERATORCONTRASTSCALEMatN_H
#define OPERATORCONTRASTSCALEMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorContrastScaleMatN : public COperator
{
public:
    OperatorContrastScaleMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double scale, BaseMatN * &out)
        {
            MatN<DIM,Type > * outcast = new MatN<DIM,Type>(in1cast->getDomain());
            *outcast = Processing::greylevelScaleContrast(*in1cast,scale);
            out =outcast;

        }
    };

};
class OperatorContrastScaleColorMatN : public COperator
{
public:
    OperatorContrastScaleColorMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGB<Type> > * in1cast,double r, double g,double b, BaseMatN * &out)
        {
            MatN<DIM,Type> rcast (in1cast->getDomain());
            MatN<DIM,Type> gcast (in1cast->getDomain());
            MatN<DIM,Type>  bcast (in1cast->getDomain());
            Convertor::toRGB(*in1cast,rcast,gcast,bcast);



            rcast = Processing::greylevelScaleContrast(rcast,r);
            gcast = Processing::greylevelScaleContrast(gcast,g);
            bcast = Processing::greylevelScaleContrast(bcast,b);

            MatN<DIM,RGBUI8 > * colorcast = new MatN<DIM,RGBUI8>(in1cast->getDomain());
            Convertor::fromRGB(rcast,gcast,bcast,*colorcast);
            out =colorcast;

        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGBA<Type> > * in1cast,double r, double g,double b, BaseMatN * &out)
        {
            MatN<DIM,Type> rcast (in1cast->getDomain());
            MatN<DIM,Type> gcast (in1cast->getDomain());
            MatN<DIM,Type>  bcast (in1cast->getDomain());
            Convertor::toRGB(*in1cast,rcast,gcast,bcast);



            rcast = Processing::greylevelScaleContrast(rcast,r);
            gcast = Processing::greylevelScaleContrast(gcast,g);
            bcast = Processing::greylevelScaleContrast(bcast,b);

            MatN<DIM,RGBUI8 > * colorcast = new MatN<DIM,RGBUI8>(in1cast->getDomain());
            Convertor::fromRGB(rcast,gcast,bcast,*colorcast);
            out =colorcast;

        }
    };

};
#endif // OPERATORCONTRASTSCALEMatN_H
