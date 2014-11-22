#include "OperatorFromImageGridMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataImageGrid.h>

OperatorConvertFromMatNMatrix::OperatorConvertFromMatNMatrix(){

    this->structurePlug().addPlugIn(DataMatN::KEY,"t.");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"A.m");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
    this->path().push_back("Convert");
    this->setKey("OperatorConvertFromImageGridMatrix");
    this->setName("fromImageGrid");
    this->setInformation("A(i,j)=img(x(j,i))");
}

void OperatorConvertFromMatNMatrix::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();


    foo func;
    BaseMatN * fc= f.get();
    typedef  FilterKeepTlistTlist<TListImgGridScalar,0,Loki::Int2Type<2> >::Result ListFilter;

    Mat2F64* m ;
    try{Dynamic2Static<ListFilter>::Switch(func,fc,m,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be scalar type and dim=2");
        return;
    }




    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}
COperator * OperatorConvertFromMatNMatrix::clone(){
    return new OperatorConvertFromMatNMatrix();
}
