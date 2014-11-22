#include "DataPoint.h"

DataPoint::DataPoint()
    :CDataByValue<VecF64   >()
{
    this->_key = DataPoint::KEY;
}
string DataPoint::KEY ="DATAPOINT";
DataPoint *DataPoint::clone()
{
    return new DataPoint;
}
