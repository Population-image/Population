#include "OperatorHoleFillingImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataBoolean.h>
OperatorHoleFillingMatN::OperatorHoleFillingMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("RegionGrowing");
    this->setKey("PopulationOperatorHoleFillingImageGrid");
    this->setName("holeFilling");
    this->setInformation("Hole filling");
    this->structurePlug().addPlugIn(DataMatN::KEY,"bin.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"n.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"holefilling.pgm");
}
void OperatorHoleFillingMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorHoleFillingMatN::exec(){
    shared_ptr<BaseMatN> topo = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int norm;
    if(this->plugIn()[1]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    else
        norm = 0;
    BaseMatN *min;
    BaseMatN * topoc= topo.get();
    foo func;
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,topoc,norm,min,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte\n");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(min));

}
COperator * OperatorHoleFillingMatN::clone(){
    return new OperatorHoleFillingMatN();
}
