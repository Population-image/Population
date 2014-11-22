#include "DataDistribution.h"
#include"data/distribution/DistributionAnalytic.h"
DataDistribution::DataDistribution()
    :CDataByValue<Distribution>()
{
    this->_key = DataDistribution::KEY;
}
string DataDistribution::KEY ="DATADISTRIBUTION";
DataDistribution * DataDistribution::clone(){
    return new DataDistribution();
}


