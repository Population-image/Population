#include "OperatorConvertFromTableMatrix.h"


#include<CData.h>
#include<DataMatrix.h>
#include<DataTable.h>

OperatorConvertFromTableMatrix::OperatorConvertFromTableMatrix(){

    this->structurePlug().addPlugIn(DataTable::KEY,"t.table");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"A.m");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
    this->path().push_back("Convert");
    this->setKey("OperatorConvertFromTableMatrix");
    this->setName("fromTable");
    this->setInformation("A(i,j)=t(col,raw)");
}

void OperatorConvertFromTableMatrix::exec(){
    shared_ptr<Table> t = dynamic_cast<DataTable *>(this->plugIn()[0]->getData())->getData();

    Mat2F64* m = new Mat2F64(t->sizeRow(),t->sizeCol());
    for(int i=0;i<m->sizeI();i++)
        for(int j=0;j<m->sizeJ();j++)
        {

            string str = t->operator ()(j,i);
            UtilityString::String2Any(str,m->operator ()(i,j) );
        }

    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}



COperator * OperatorConvertFromTableMatrix::clone(){
    return new OperatorConvertFromTableMatrix();
}
