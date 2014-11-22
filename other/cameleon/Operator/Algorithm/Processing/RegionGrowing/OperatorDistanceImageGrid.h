#ifndef OPERATORDISTANCEMatN_H
#define OPERATORDISTANCEMatN_H
#include<OperatorVoronoiTesselationImageGrid.h>
class OperatorDistanceMatN : public OperatorVoronoiTesselationMatN
{
public:
    OperatorDistanceMatN();
    COperator * clone();

};

#endif // OPERATORDISTANCEMatN_H
