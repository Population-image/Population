#include "OperatorLoadDistribution.h"

#include<DataDistribution.h>
#include<DataString.h>
OperatorLoadDistribution::OperatorLoadDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("InOut");
    this->setKey("OperatorLoadDistribution");
    this->setName("load");
    this->setInformation("Load distribution by file");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"h.dist");
}

void OperatorLoadDistribution::exec(){

    string file = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
    try{
        Distribution d;
        d.load(file.c_str());

            dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(d);
    }
    catch(pexception msg){
        this->error(msg.what());
    }

}

COperator * OperatorLoadDistribution::clone(){
    return new OperatorLoadDistribution();
}
