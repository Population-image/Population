#include "OperatorLoadDistributionMultiVariate.h"

#include<DataDistributionMultiVariate.h>
#include<DataString.h>
OperatorLoadDistributionMultiVariate::OperatorLoadDistributionMultiVariate()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("InOut");
    this->setKey("OperatorLoadDistributionMultiVariate");
    this->setName("load");
    this->setInformation("Load DistributionMultiVariate by file");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"h.dist");
}

void OperatorLoadDistributionMultiVariate::exec(){

    string file = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
    try{
        DistributionMultiVariate d;
        d.load(file.c_str());

            dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(d);
    }
    catch(pexception msg){
        this->error(msg.what());
    }

}

COperator * OperatorLoadDistributionMultiVariate::clone(){
    return new OperatorLoadDistributionMultiVariate();
}
