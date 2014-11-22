#include "OperatorNormalDistributionMultiVariate.h"
#include "data/distribution/DistributionMultiVariateFromDataStructure.h"
#include<DataPoint.h>
#include<DataMatrix.h>
OperatorNormalDistributionMultiVariate::OperatorNormalDistributionMultiVariate()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Catalog");
    this->setKey("PopulationOperatorNormalDistributionMultiVariate");
    this->setName("normal");
    this->setInformation("http://en.wikipedia.org/wiki/Multivariate_normal_distribution f(x)=$\\frac\\{1\\}\\{\\sqrt\\{2 \\pi \\sigma^2\\}\\} e^\\{ - \\frac\\{(x-mean)^2\\}\\{2 \\sigma^2\\}\\}$");
    this->structurePlug().addPlugIn(DataPoint::KEY,"mean.num location");
    this->structurePlug().addPlugIn(DataMatrix::KEY,"sigma.num covariance");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"f.dist");
}

void OperatorNormalDistributionMultiVariate::exec(){
    VecF64 mean = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix *>(this->plugIn()[1]->getData())->getData();
    DistributionMultiVariateNormal gen;
    gen.fromMeanVecAndCovarianceMatrix(mean,*m.get());
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(gen);

}

COperator * OperatorNormalDistributionMultiVariate::clone(){
    return new OperatorNormalDistributionMultiVariate();
}
