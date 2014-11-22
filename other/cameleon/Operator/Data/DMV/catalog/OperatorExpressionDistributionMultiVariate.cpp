#include "OperatorExpressionDistributionMultiVariate.h"

#include "OperatorExpressionDistributionMultiVariate.h"
#include "data/distribution/DistributionMultiVariateFromDataStructure.h"
#include<DataString.h>
OperatorExpressionDistributionMultiVariate::OperatorExpressionDistributionMultiVariate()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Catalog");
    this->setKey("PopulationOperatorExpressionDistributionMultiVariate");
    this->setName("expression");
    this->setInformation("The distribution is defined from a regulation expression with the variables defined in the secound string input with a comma to separate their");
    this->structurePlug().addPlugIn(DataString::KEY,"regular.str as x*y");
    this->structurePlug().addPlugIn(DataString::KEY,"variable.str as x,y");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"f.dist");
}

void OperatorExpressionDistributionMultiVariate::exec(){
    string reg = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
    string var = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    DistributionMultiVariateExpression gen;
    gen.fromRegularExpression(make_pair(reg,var));
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(gen);

}

COperator * OperatorExpressionDistributionMultiVariate::clone(){
    return new OperatorExpressionDistributionMultiVariate();
}
