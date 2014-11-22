#include "OperatorGenerate3DRotationMatrix.h"

#include<DataMatrix.h>
#include<DataNumber.h>
#include"algorithm/LinearAlgebra.h"
OperatorGenerate3DRotationMatrix::OperatorGenerate3DRotationMatrix(){


    this->path().push_back("Algorithm");
    this->path().push_back("LinearAlgebra");
    this->setKey("OperatorGenerate3DRotationMatrix");
    this->setName("generate3DRotation");
    this->setInformation("Generate a rotation matrix with a given angle in radian and a given coordinate see http://en.wikipedia.org/wiki/Rotation\\_matrix");
    this->structurePlug().addPlugIn(DataNumber::KEY,"angle.num (radian)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"coordinate.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"R.m");

}

void OperatorGenerate3DRotationMatrix::exec(){
    double angle = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    int coordinate = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    try{
        Mat2F64* m = new Mat2F64;
        *m = LinearAlgebra::generate3DRotation(angle,coordinate);
        dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>( m) );
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorGenerate3DRotationMatrix::clone(){
    return new OperatorGenerate3DRotationMatrix();
}
