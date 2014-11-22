#ifndef DATADISTRIBUTION_H
#define DATADISTRIBUTION_H
#include<CDataByValue.h>

#include"data/distribution/Distribution.h"
#include"data/distribution/DistributionAnalytic.h"
using namespace pop;
class DataDistribution : public CDataByValue<Distribution>
{
public:

    DataDistribution();
    static string KEY;
    DataDistribution* clone();

};

#endif // DATADISTRIBUTION_H
