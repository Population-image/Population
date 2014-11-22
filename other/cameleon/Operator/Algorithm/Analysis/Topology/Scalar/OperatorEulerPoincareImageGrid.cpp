#include "OperatorEulerPoincareImageGrid.h"
#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataString.h>
#include<DataMatrix.h>
OperatorEulerPoincareMatN::OperatorEulerPoincareMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Topology");
    this->path().push_back("Scalar");
    this->setKey("PopulationOperatorEulerPoincareImageGrid");
    this->setName("eulerPoincare");
    this->setInformation("Gaussian curvature of the binary image");
    this->structurePlug().addPlugIn(DataMatN::KEY,"bin.pgm");
    this->structurePlug().addPlugIn(DataString::KEY,"eulertab.file");
    this->structurePlug().addPlugOut(DataNumber::KEY,"euler.num");
}

void OperatorEulerPoincareMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    string file = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    double number;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,fc1,file,number,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(number);
}

COperator * OperatorEulerPoincareMatN::clone(){
    return new OperatorEulerPoincareMatN();
}
