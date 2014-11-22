#include "DataDistributionMultiVariate.h"

#include"data/distribution/DistributionMultiVariateFromDataStructure.h"
DataDistributionMultiVariate::DataDistributionMultiVariate()
    :CDataByValue<DistributionMultiVariate>()
{
    this->_key = DataDistributionMultiVariate::KEY;
}
string DataDistributionMultiVariate::KEY ="DATADISTRIBUTIONMULTIVARIATE";
DataDistributionMultiVariate * DataDistributionMultiVariate::clone(){
    return new DataDistributionMultiVariate();
}
