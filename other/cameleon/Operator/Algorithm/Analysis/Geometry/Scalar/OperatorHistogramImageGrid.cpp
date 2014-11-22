#include "OperatorHistogramImageGrid.h"

#include<DataImageGrid.h>
#include<DataMatrix.h>
OperatorHistogramMatN::OperatorHistogramMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Scalar");
    this->setKey("PopulationOperatorHistogramImageGrid");
    this->setName("histogram");
    this->setInformation("A(i,0)=i and A(i,1) = |Sigma$_i$|/|Sigma| where Sigma$_i$={x in Sigma :f(x)=i}");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"A.m");
}

void OperatorHistogramMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    Mat2F64* m;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,m,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be scalar type (used ConvertColor2GreyMatN or ConvertColor2RGBMatN for color image)");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}

COperator * OperatorHistogramMatN::clone(){
    return new OperatorHistogramMatN();
}
