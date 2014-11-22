#include "OperatorPercolationErosionImageGrid.h"
#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataMatrix.h>
OperatorPercolationErosionMatN::OperatorPercolationErosionMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Topology");
    this->path().push_back("Scalar");
    this->setKey("PopulationOperatorPercolationErosionImageGrid");
    this->setName("percolationErosion");
    this->setInformation("A(i,0)=i and A(i,1) = r for maximum radius such that the binary still percole for the coordinate i after the erosion with this radius (for a diameter 2*r+1)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"norm.num (default 1)");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"A.m");
}

void OperatorPercolationErosionMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorPercolationErosionMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int norm;
    if(this->plugIn()[1]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    else
        norm =1;
    Mat2F64* m;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,norm, m,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}

COperator * OperatorPercolationErosionMatN::clone(){
    return new OperatorPercolationErosionMatN();
}
