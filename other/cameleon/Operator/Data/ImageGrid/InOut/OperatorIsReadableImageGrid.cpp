#include "OperatorIsReadableImageGrid.h"

#include<DataImageGrid.h>
#include<DataString.h>
#include<DataBoolean.h>
OperatorIsReadableMatN::OperatorIsReadableMatN()
    :COperator()
{


    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorIsReadableImageGrid");
    this->setName("isReadable");
    this->setInformation("bool =true is input file is readable otherwise false");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataBoolean::KEY,"bool.bool");
}

void OperatorIsReadableMatN::exec(){

    string file = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
    try{
        BaseMatN * h = BaseMatN::create(file);
        dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(true);
        delete h;
    }
    catch(pexception msg){
        dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(false);
    }

}

COperator * OperatorIsReadableMatN::clone(){
    return new OperatorIsReadableMatN();
}
