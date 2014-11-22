#include "OperatorThinningAtConstantTopology.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataString.h>
#include<DataMatrix.h>
OperatorThinningAtConstantTopologyMatN::OperatorThinningAtConstantTopologyMatN()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Topology");
    this->path().push_back("Skeleton");
    this->setKey("PopulationOperatorThinningAtConstantTopologyImageGrid");
    this->setName("thinningAtConstantTopology");
    this->setInformation("Thinning the binary image at constant topology\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"bin.pgm");
    this->structurePlug().addPlugIn(DataString::KEY,"topo24.dat");
    this->structurePlug().addPlugOut(DataMatN::KEY,"skeleton.pgm");
}
void OperatorThinningAtConstantTopologyMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    string file = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    foo func;
    BaseMatN * skeleton;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,fc1,file,skeleton,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(skeleton));
}

COperator * OperatorThinningAtConstantTopologyMatN::clone(){
    return new OperatorThinningAtConstantTopologyMatN();
}
