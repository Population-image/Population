#ifndef DATAPOINT_H
#define DATAPOINT_H
#include<CDataByValue.h>

#include"data/vec/Vec.h"

using namespace std;
using namespace pop;
class DataPoint: public CDataByValue<VecF64   >
{
public:
    static string KEY;
    DataPoint();
    virtual DataPoint * clone();
};
#endif // DATAPOINT_H
