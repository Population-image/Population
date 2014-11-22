#include "OperatorAnnealingSimulatedImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include"algorithm/RandomGeometry.h"
OperatorAnnealingSimulatedMatN::OperatorAnnealingSimulatedMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("FitGeometry");
    this->setKey("PopulationOperatorAnnealingSimulated");
    this->setName("annealingSimulated");
    this->setInformation("annealing simulated method for 3d reconstruction based on correlation function\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"reference.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"model.pgm(at time t=0)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"length correlation (by default 150)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"number of permutations by pixel/voxel (by default 8)");
    this->structurePlug().addPlugOut(DataMatN::KEY,"model.pgm(at end time)");
}

void OperatorAnnealingSimulatedMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);

    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);

    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorAnnealingSimulatedMatN::exec(){

    shared_ptr<BaseMatN> ref= dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> model= dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();

    double lengthcor=-1;
    if(this->plugIn()[2]->isDataAvailable()==true)
        lengthcor = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    double permut=8;
    if(this->plugIn()[2]->isDataAvailable()==true)
        permut = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();



    if(Mat2UI8* refcast =   dynamic_cast<Mat2UI8*>(ref.get())){
        if(Mat2UI8* modelcast =   dynamic_cast<Mat2UI8*>(model.get())){
            RandomGeometry::annealingSimutated(*modelcast, * refcast,lengthcor,permut);
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(model);
        }else if(Mat3UI8* modelcast =   dynamic_cast<Mat3UI8*>(model.get())){
            RandomGeometry::annealingSimutated(*modelcast, * refcast,lengthcor,permut);
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(model);
        }
    }else if(Mat3UI8* refcast =   dynamic_cast<Mat3UI8*>(ref.get())){
        if(Mat2UI8* modelcast =   dynamic_cast<Mat2UI8*>(model.get())){
            RandomGeometry::annealingSimutated(*modelcast, * refcast,lengthcor,permut);
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(model);
        }else if(Mat3UI8* modelcast =   dynamic_cast<Mat3UI8*>(model.get())){
            RandomGeometry::annealingSimutated(*modelcast, * refcast,lengthcor,permut);
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(model);
        }
    }
}


COperator * OperatorAnnealingSimulatedMatN::clone(){
    return new OperatorAnnealingSimulatedMatN();
}
