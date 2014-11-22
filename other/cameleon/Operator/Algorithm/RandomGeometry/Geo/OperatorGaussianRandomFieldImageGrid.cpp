#include "OperatorGaussianRandomFieldImageGrid.h"

#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataMatrix.h>
#include<DataMatrix.h>
#include"algorithm/RandomGeometry.h"
OperatorGaussianRandomFieldMatN::OperatorGaussianRandomFieldMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("FitGeometry");
    this->setKey("PopulationOperatorGaussianRandomField");
    this->setName("gaussianRandomField");
    this->setInformation("binary is a realisation of the gaussian random field with  domain=x and the 2-point correlation function corr (used correlation operator) \n");
    this->structurePlug().addPlugIn(DataPoint::KEY,"x.v");
    this->structurePlug().addPlugIn(DataMatrix::KEY,"corr.m");
    this->structurePlug().addPlugOut(DataMatN::KEY,"binary.num");
}
void OperatorGaussianRandomFieldMatN::exec(){

    VecF64   v= dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();


    shared_ptr<Mat2F64>  m= dynamic_cast<DataMatrix*>(this->plugIn()[1]->getData())->getData();

    if(v.size()==2){
        Vec2F64 x;
        x = v;
        Mat2UI8 * bin = new Mat2UI8;
        Mat2F64 gaussian;
        * bin = RandomGeometry::gaussianThesholdedRandomField(*m.get(), Vec2I32(x),gaussian);
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(bin));
    }else{
        Vec3F64 x;
        x= v;
        Mat3UI8 * bin = new Mat3UI8;
        Mat3F64 gaussian;
        * bin = RandomGeometry::gaussianThesholdedRandomField(*m.get(), Vec3I32(x),gaussian);
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(bin));
    }
}


COperator * OperatorGaussianRandomFieldMatN::clone(){
    return new OperatorGaussianRandomFieldMatN();
}
