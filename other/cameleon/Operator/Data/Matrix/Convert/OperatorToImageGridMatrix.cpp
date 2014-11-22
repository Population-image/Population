#include "OperatorToImageGridMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataImageGrid.h>

OperatorConvertToMatNMatrix::OperatorConvertToMatNMatrix(){
    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugOut(DataMatN::KEY,"t.");

    this->path().push_back("Data");
    this->path().push_back("Matrix");
    this->path().push_back("Convert");
    this->setKey("OperatorConvertToImageGridMatrix");
    this->setName("toImageGrid");
    this->setInformation("img(x(j,i))=A(i,j= with a Float Pixel type");
}

void OperatorConvertToMatNMatrix::exec(){
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();

    VecN<2,int> x(m->sizeJ(),m->sizeI());
    MatN<2,pop::F64> * t = new MatN<2,pop::F64>(x);
    for(int i=0;i<m->sizeI();i++)
        for(int j=0;j<m->sizeJ();j++){
            t->operator ()(j,i) = m->operator ()(i,j);
        }

    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(t));
}



COperator * OperatorConvertToMatNMatrix::clone(){
    return new OperatorConvertToMatNMatrix();
}
