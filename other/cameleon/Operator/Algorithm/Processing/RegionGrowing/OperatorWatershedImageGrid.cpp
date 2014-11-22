#include "OperatorWatershedImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataBoolean.h>
OperatorWatershedMatN::OperatorWatershedMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("RegionGrowing");
    this->setKey("PopulationOperatorWatershedImageGrid");
    this->setName("watershed");
    this->setInformation("Watershed transformation if bound = true we have watershed line to separate the bassins");
    this->structurePlug().addPlugIn(DataMatN::KEY,"seed.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"topo.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"mask.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"n.num(by default 1)");
    this->structurePlug().addPlugIn(DataBoolean::KEY,"boundary.bool(by default false)");
    this->structurePlug().addPlugOut(DataMatN::KEY,"watershed.pgm");
}
void OperatorWatershedMatN::initState(){
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

    if(this->plugIn()[4]->isConnected()==false)
        this->plugIn()[4]->setState(CPlug::OLD);
    else
        this->plugIn()[4]->setState(CPlug::EMPTY);


    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorWatershedMatN::exec(){
    shared_ptr<BaseMatN> seed = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> topo = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    int norm;
    if(this->plugIn()[3]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
    else
        norm = 1;
    BaseMatN *water;

    BaseMatN * seedc= seed.get();
    BaseMatN * topoc= topo.get();

    bool boundary;
    if(this->plugIn()[4]->isDataAvailable()==true)
        boundary = dynamic_cast<DataBoolean *>(this->plugIn()[4]->getData())->getValue();
    else
        boundary = false;
    if(boundary==false)
    {
        foo func;
        if(this->plugIn()[2]->isDataAvailable()==true){
            shared_ptr<BaseMatN>  mask = dynamic_cast<DataMatN *>(this->plugIn()[2]->getData())->getData();
            BaseMatN * maskc= mask.get();
            try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,seedc,topoc,maskc,norm,water,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg){
                if(msg.what()[0]=='P')
                    this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
                else
                    this->error(msg.what());
                return;
            }
        }
        else{
            try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,seedc,topoc,norm,water,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg){
                if(msg.what()[0]=='P')
                    this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
                else
                    this->error(msg.what());
                return;
            }
        }
    }
    else
    {
        foob func;
        if(this->plugIn()[2]->isDataAvailable()==true){
            shared_ptr<BaseMatN>  mask = dynamic_cast<DataMatN *>(this->plugIn()[2]->getData())->getData();
            BaseMatN * maskc= mask.get();
            try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,seedc,topoc,maskc,norm,water,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg){
                if(msg.what()[0]=='P')
                    this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
                else
                    this->error(msg.what());
                return;
            }
        }
        else{
            try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,seedc,topoc,norm,water,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg){
                if(msg.what()[0]=='P')
                    this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
                else
                    this->error(msg.what());
                return;
            }
        }
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(water));
}

COperator * OperatorWatershedMatN::clone(){
    return new OperatorWatershedMatN();
}
