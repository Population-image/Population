#include "OperatorToMatrixDistribution.h"

#include"algorithm/Statistics.h"
#include<DataDistribution.h>
#include<DataMatrix.h>
#include<DataNumber.h>
OperatorConvertMatrixDistribution::OperatorConvertMatrixDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Convert");
    this->setKey("PopulationOperatorConvertMatrixDistribution");
    this->setName("toMatrix");
    this->setInformation("M(i,0)=xmin+i*step  and M(i,1)=f(xmin+i*step) ");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmax.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"step.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"M.m");
}
void OperatorConvertMatrixDistribution::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugIn()[2]->setState(CPlug::EMPTY);
    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorConvertMatrixDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();


    double xmin = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double xmax = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    double step;
    if(this->plugIn()[3]->isDataAvailable()==true){
        step = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
        if(step<=0)
            this->error("step must be superior to 0");
    }
    else{

        step = 0.01;

    }
    Mat2F64*m = new Mat2F64;
    *m =  Statistics::toMatrix(f, xmin, xmax, step);
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}

COperator * OperatorConvertMatrixDistribution::clone(){
    return new OperatorConvertMatrixDistribution();
}
