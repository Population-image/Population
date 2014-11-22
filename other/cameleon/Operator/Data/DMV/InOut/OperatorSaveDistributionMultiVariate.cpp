#include "OperatorSaveDistributionMultiVariate.h"

#include<DataDistributionMultiVariate.h>
#include<DataString.h>
#include<DataBoolean.h>
OperatorSaveDistributionMultiVariate::OperatorSaveDistributionMultiVariate()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorSaveDistributionMultiVariate");
    this->setName("save");
    this->setInformation("Save image by file");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"h.pgm");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataBoolean::KEY,"out.bool");
    this->setInformation("Save DistributionMultiVariate to the given file,  out= false for bad writing, true otherwise");
}

void OperatorSaveDistributionMultiVariate::exec(){
    DistributionMultiVariate h = dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();
    string file = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    h.save(file.c_str());
    dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(true);

}

COperator * OperatorSaveDistributionMultiVariate::clone(){
    return new OperatorSaveDistributionMultiVariate();
}

