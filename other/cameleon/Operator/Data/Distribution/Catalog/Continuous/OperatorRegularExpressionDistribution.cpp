#include "OperatorRegularExpressionDistribution.h"
#include "data/distribution/DistributionFromDataStructure.h"
#include<DataString.h>
OperatorRegularExpressionDistribution::OperatorRegularExpressionDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Continuous");
    this->setKey("PopulationOperatorRegularExpressionDistribution");
    this->setName("regularExpression");
    this->setInformation("The distribution is defined from a regulation expression with x as variable");
    this->structurePlug().addPlugIn(DataString::KEY,"expression.string");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}

void OperatorRegularExpressionDistribution::exec(){
    string expression = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
    DistributionExpression gen;
    bool test = gen.fromRegularExpression(expression);
    if(test==false)
    {
        this->error("Syntax error in parsing "+expression);
        return;
    }
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);

}

COperator * OperatorRegularExpressionDistribution::clone(){
    return new OperatorRegularExpressionDistribution();
}
