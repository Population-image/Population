#include "OperatorRandomStructureImageGrid.h"

#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataMatrix.h>
#include<DataMatrix.h>
#include"algorithm/RandomGeometry.h"
OperatorRandomStructureMatN::OperatorRandomStructureMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("FitGeometry");
    this->setKey("PopulationOperatorRandomStructure");
    this->setName("randomStructure");
    this->setInformation("generate a random structure at the given volume fraction \n");
    this->structurePlug().addPlugIn(DataPoint::KEY,"domain.v");
    this->structurePlug().addPlugIn(DataMatrix::KEY,"volumefraction.m");
    this->structurePlug().addPlugOut(DataMatN::KEY,"binary.num");
}
void OperatorRandomStructureMatN::exec(){

    VecF64   v= dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();


    shared_ptr<Mat2F64>  m= dynamic_cast<DataMatrix*>(this->plugIn()[1]->getData())->getData();

    if(v.size()==2){
        Vec2F64 x;
        x=v;
        Mat2UI8 * bin = new Mat2UI8;
        * bin = RandomGeometry::randomStructure( Vec2I32(x),*m.get());
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(bin));
    }else{
        Vec3F64 x;
        x=v;
        Mat3UI8 * bin = new Mat3UI8;
        * bin = RandomGeometry::randomStructure(Vec3I32(x),*m.get());
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(bin));
    }
}


COperator * OperatorRandomStructureMatN::clone(){
    return new OperatorRandomStructureMatN();
}
