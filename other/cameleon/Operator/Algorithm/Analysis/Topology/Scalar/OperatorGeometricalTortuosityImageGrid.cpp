#include "OperatorGeometricalTortuosityImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataString.h>
#include<DataMatrix.h>
OperatorGeometricalTortuosityMatN::OperatorGeometricalTortuosityMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Topology");
    this->path().push_back("Scalar");
    this->setKey("PopulationOperatorGeometricalTortuosityImageGrid");
    this->setName("geometricalTortuosity");
    this->setInformation("Geometrical Tortuosity of the binary image work with the norm 1 and infinite (n=0) ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"bin.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"n.num(by default 1)");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"tortuosity.m");
}
void OperatorGeometricalTortuosityMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorGeometricalTortuosityMatN::exec(){
    shared_ptr<BaseMatN> topo = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int norm;
    if(this->plugIn()[1]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    else
        norm = 1;

    Mat2F64 * m ;
    BaseMatN * topoc= topo.get();
    foo func;
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,topoc,norm,m,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte\n");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}

COperator * OperatorGeometricalTortuosityMatN::clone(){
    return new OperatorGeometricalTortuosityMatN();
}
