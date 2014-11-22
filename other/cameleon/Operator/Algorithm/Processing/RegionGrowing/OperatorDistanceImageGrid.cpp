#include "OperatorDistanceImageGrid.h"

OperatorDistanceMatN::OperatorDistanceMatN()
    :OperatorVoronoiTesselationMatN()
{
    this->setKey("OperatorDistance");
    this->setName("distanceFunction");
}
COperator * OperatorDistanceMatN::clone(){
    return new OperatorDistanceMatN;
}
