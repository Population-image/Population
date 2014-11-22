#include "OperatorDynamicImageGrid.h"


#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorDynamicMatN::OperatorDynamicMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("RegionGrowing");
    this->setKey("PopulationOperatorDynamicImageGrid");
    this->setName("dynamic");
    this->setInformation("$f_{i+1}=max(erosion(f_i),g)$ with $f_0(x) = g(x)+val$ and $f_{\\infty}=h$ ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"g.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"d.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"norm.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorDynamicMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorDynamicMatN::exec(){
    try{
        shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
        int num = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
        int norm;
        if(this->plugIn()[2]->isDataAvailable()==true)
            norm = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
        else
            norm = 1;
        BaseMatN * h;
        foo func;
        BaseMatN * fc1= f1.get();
        Dynamic2Static<TListImgGridUnsigned>::Switch(func,fc1,num,norm,h,Loki::Type2Type<MatN<2,int> >());
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
    }
    catch(pexception msg){

        this->error(string("Pixel/voxel type of input images must be unsigned type used operator Convert1Byte\n")+msg.what());
        return;
    }

}
COperator * OperatorDynamicMatN::clone(){
    return new OperatorDynamicMatN();
}
