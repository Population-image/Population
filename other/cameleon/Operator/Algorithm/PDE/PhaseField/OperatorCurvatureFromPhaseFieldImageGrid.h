#ifndef OPERATORCURVATUREFROMPHASEFIELDMatN_H
#define OPERATORCURVATUREFROMPHASEFIELDMatN_H


#include"COperator.h"

class OperatorCurvatureFromPhaseFieldMatN : public COperator
{
public:
    OperatorCurvatureFromPhaseFieldMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
//        template<int DIM,typename Type>
//        void operator()(MatN<DIM,Type> *  scalarfield_singlephasefield_cast,Image * bulk ,Image * &curvature)throw(pexception)
//        {


//        }
    };

};
#endif // OPERATORCURVATUREFROMPHASEFIELDMatN_H
