#ifndef OPERATORGEOMETRICALTORTUOSITYMatN_H
#define OPERATORGEOMETRICALTORTUOSITYMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
using namespace pop;
class OperatorGeometricalTortuosityMatN : public COperator
{
public:
    OperatorGeometricalTortuosityMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,int norm, Mat2F64*& m)
        {
           m  = new Mat2F64;
           *m = Analysis::geometricalTortuosity(*in1cast,norm);
        }
    };

};
#endif // OPERATORGEOMETRICALTORTUOSITYMatN_H
