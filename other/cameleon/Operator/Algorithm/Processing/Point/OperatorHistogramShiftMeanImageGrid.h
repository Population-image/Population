#ifndef OPERATORHISTOGRAMSHIFTMEANMatN_H
#define OPERATORHISTOGRAMSHIFTMEANMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"

#include"algorithm/Visualization.h"
using namespace pop;
#include"data/distribution/DistributionFromDataStructure.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorHistogramShiftMeanmageGrid : public COperator
{
public:
    OperatorHistogramShiftMeanmageGrid();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGB<Type> > * in1cast,double red,double green ,double blue, BaseMatN * &out)throw(pexception)
        {

            MatN<DIM,RGB<Type > > * outcast = new MatN<DIM,RGB<Type > >;
            RGBUI8 p (red,green,blue);
            *outcast = Processing::greylevelTranslateMeanValue(* in1cast,p);
            out = outcast;
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,RGBA<Type> > * in1cast,double red,double green ,double blue, BaseMatN * &out)throw(pexception)
        {

            MatN<DIM,RGBA<Type > > * outcast = new MatN<DIM,RGBA<Type > >;
            RGBUI8 p (red,green,blue);
            *outcast = Processing::greylevelTranslateMeanValue(MatN<DIM,RGB<Type> >(* in1cast),p);
            out = outcast;
        }
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double mean, BaseMatN * &out)throw(pexception)
        {
            MatN<DIM,Type> * outcast = new MatN<DIM,Type>;
            *outcast = Processing::greylevelTranslateMeanValue(* in1cast,mean);
            out = outcast;


        }
    };

};
#endif // OPERATORHISTOGRAMSHIFTMEANMatN_H
