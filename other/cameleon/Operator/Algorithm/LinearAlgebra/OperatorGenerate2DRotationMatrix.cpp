#include "OperatorGenerate2DRotationMatrix.h"

#include<DataMatrix.h>
#include<DataNumber.h>

#include"algorithm/LinearAlgebra.h"
OperatorGenerate2DRotationMatrix::OperatorGenerate2DRotationMatrix(){


    this->path().push_back("Algorithm");
    this->path().push_back("LinearAlgebra");
    this->setKey("OperatorGenerate2DRotationMatrix");
    this->setName("generate2DRotation");
    this->setInformation("Generate a rotation matrix with a given angle in radian  see http://en.wikipedia.org/wiki/Rotation\\_matrix");
    this->structurePlug().addPlugIn(DataNumber::KEY,"angle.num (radian)");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"R.m");

}

void OperatorGenerate2DRotationMatrix::exec(){
    double angle = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    try{
        Mat2F64* m = new Mat2F64;
        *m = LinearAlgebra::generate2DRotation(angle);
        dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>( m) );
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorGenerate2DRotationMatrix::clone(){
    return new OperatorGenerate2DRotationMatrix();
}
