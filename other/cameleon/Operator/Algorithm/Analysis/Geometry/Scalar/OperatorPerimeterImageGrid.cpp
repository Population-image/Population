#include "OperatorPerimeterImageGrid.h"

#include<DataImageGrid.h>
#include<DataMatrix.h>
OperatorPerimeterMatN::OperatorPerimeterMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Scalar");
    this->setKey("PopulationOperatorPerimeterImageGrid");
    this->setName("perimeter");
    this->setInformation("V(i) = |border(Sigma$_i$)| where Sigma$_i$={x:f(x)=i}\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"P.m");
}

void OperatorPerimeterMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
        Mat2F64* m;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,m,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}

COperator * OperatorPerimeterMatN::clone(){
    return new OperatorPerimeterMatN();
}
