#ifndef OPERATORGETFIELDFROMMULTIPHASEFIELDMatN_H
#define OPERATORGETFIELDFROMMULTIPHASEFIELDMatN_H


#include"COperator.h"
#include"algorithm/PDE.h"
using namespace pop;
class OperatorGetFieldFromMultiPhaseFieldMatN : public COperator
{
public:
    OperatorGetFieldFromMultiPhaseFieldMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * label_multiphasefield_cast,BaseMatN*  scalarfield_multiphasefield, int label,int width,BaseMatN* bulk,BaseMatN * &scalarfield_singlefield)throw(pexception)
        {



        }
    };

};
#endif // OPERATORGETFIELDFROMMULTIPHASEFIELDMatN_H
