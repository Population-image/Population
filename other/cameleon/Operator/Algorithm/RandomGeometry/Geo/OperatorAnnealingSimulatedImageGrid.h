#ifndef OPERATORANNEALINGSIMULATEDMatN_H
#define OPERATORANNEALINGSIMULATEDMatN_H

#include"COperator.h"

class OperatorAnnealingSimulatedMatN : public COperator
{
public:
    OperatorAnnealingSimulatedMatN();
    void exec();
    COperator * clone();
    void initState();
};
#endif // OPERATORANNEALINGSIMULATEDMatN_H
