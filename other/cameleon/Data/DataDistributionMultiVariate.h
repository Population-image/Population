#ifndef DATADISTRIBUTIONMULTIVARIATE_H
#define DATADISTRIBUTIONMULTIVARIATE_H

#include<CDataByValue.h>

#include"data/distribution/DistributionMultiVariate.h"
using namespace pop;
class DataDistributionMultiVariate : public CDataByValue<DistributionMultiVariate>
{
public:

    DataDistributionMultiVariate();
    static string KEY;
    DataDistributionMultiVariate* clone();

};
#endif // DATADISTRIBUTIONMULTIVARIATE_H
