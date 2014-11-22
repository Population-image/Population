#include "OperatorSaveDistribution.h"

#include<DataDistribution.h>
#include<DataString.h>
#include<DataBoolean.h>
OperatorSaveDistribution::OperatorSaveDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorSaveDistribution");
    this->setName("save");
    this->setInformation("Save image by file");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"h.pgm");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataBoolean::KEY,"out.bool");
    this->setInformation("Save distribution to the given file,  out= false for bad writing, true otherwise");
}

void OperatorSaveDistribution::exec(){
    Distribution h = dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
    string file = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    h.save(file.c_str());
    dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(true);

}

COperator * OperatorSaveDistribution::clone(){
    return new OperatorSaveDistribution();
}

