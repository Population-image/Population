#include "OperatorSaveMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataString.h>

#include<DataBoolean.h>
OperatorSaveMatrix::OperatorSaveMatrix(){
    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataBoolean::KEY,"out.bool");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
    this->path().push_back("InOut");
    this->setKey("OperatorSaveMatrix");
    this->setName("save");
    this->setInformation("Save matrix to given file,  out= false for bad writing, true otherwise");
}

void OperatorSaveMatrix::exec(){
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    string str  = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    m->save(str.c_str());
        dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(true);
}



COperator * OperatorSaveMatrix::clone(){
    return new OperatorSaveMatrix();
}
