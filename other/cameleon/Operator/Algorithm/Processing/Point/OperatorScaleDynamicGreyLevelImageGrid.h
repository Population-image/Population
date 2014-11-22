#ifndef OPERATORSCALEDYNAMICGREYLEVELMatN_H
#define OPERATORSCALEDYNAMICGREYLEVELMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorScaleDynamicGreyLevelMatN : public COperator
{
public:
    OperatorScaleDynamicGreyLevelMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,double minv,double maxv, BaseMatN * &out)
        {
            MatN<DIM,Type> * outcast = new MatN<DIM,Type>(in1cast->getDomain());

            if(minv==numeric_limits<double>::max())
                minv=numeric_limits_perso<Type>::min();
            if(maxv==numeric_limits<double>::max())
                maxv=numeric_limits_perso<Type>::max();

            *outcast = Processing::greylevelRange(*in1cast,minv,maxv);
            out=outcast;
        }
    };

};

#endif // OPERATORSCALEDYNAMICGREYLEVELMatN_H
