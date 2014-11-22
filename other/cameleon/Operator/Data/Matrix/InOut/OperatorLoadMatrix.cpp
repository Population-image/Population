#include "OperatorLoadMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataString.h>

OperatorLoadMatrix::OperatorLoadMatrix(){

    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"A.m");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
    this->path().push_back("InOut");
    this->setKey("OperatorLoadMatrix");
    this->setName("load");
    this->setInformation("Load matrix from file");
}

void OperatorLoadMatrix::exec(){
    string str  = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
    Mat2F64* m = new Mat2F64;
    m->load(str.c_str());
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}



COperator * OperatorLoadMatrix::clone(){
    return new OperatorLoadMatrix();
}
