#include "OperatorConvertToTableMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataTable.h>

OperatorConvertToTableMatrix::OperatorConvertToTableMatrix(){
    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugOut(DataTable::KEY,"t.table");

    this->path().push_back("Data");
    this->path().push_back("Matrix");
    this->path().push_back("Convert");
    this->setKey("OperatorConvertToTableMatrix");
    this->setName("toTable");
    this->setInformation("t(row,col)=A(i,j)");
}

void OperatorConvertToTableMatrix::exec(){
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();

    Table * t = new Table(m->sizeJ(),m->sizeI());
    for(int i=0;i<m->sizeI();i++)
        for(int j=0;j<m->sizeJ();j++){
            double d = m->operator ()(i,j);
            t->operator ()(j,i) = UtilityString::Any2String(d);
        }

    dynamic_cast<DataTable *>(this->plugOut()[0]->getData())->setData(shared_ptr<Table>(t));
}



COperator * OperatorConvertToTableMatrix::clone(){
    return new OperatorConvertToTableMatrix();
}
