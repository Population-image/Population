#ifndef OPERATORGAUSSIANRANDOMFIELDMatN_H
#define OPERATORGAUSSIANRANDOMFIELDMatN_H
#include"COperator.h"

class OperatorGaussianRandomFieldMatN : public COperator
{
public:
    OperatorGaussianRandomFieldMatN();
    void exec();
    COperator * clone();


};

#endif // OPERATORGAUSSIANRANDOMFIELDMatN_H
