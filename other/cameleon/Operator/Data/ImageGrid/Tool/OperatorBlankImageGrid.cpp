#include "OperatorBlankImageGrid.h"
#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataString.h>
OperatorBlankMatN::OperatorBlankMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorBlankImageGrid");
    this->setName("blank");
    this->setInformation("h(x)=0 domain(h)=domain and pixel type = 1Byte by default, type otherwise where \n type=P5 for 2d image in unsigned 1Byte,\n type=PA for 2d image in signed 2Bytes,\n type=PB for 2d image in signed 4Bytes\n type=PC for 2d image in float 8Bytes,\n type=P6 or 2d image in Color in 3*1Byte,\n type=PL for 3d image in unsigned 1Byte,\n type=PN for 3d image in signed 2Bytes,\n type=PP for 3d image in signed 4Bytes\n type=PR for 3d image in float 8Bytes,\n type=PT or 3d image in Color in 3*1Byte \n");
    this->structurePlug().addPlugIn(DataPoint::KEY,"domain.v");
    this->structurePlug().addPlugIn(DataString::KEY,"type.str");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorBlankMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorBlankMatN::exec(){
    VecF64  domain =    dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    string type ;
    if(this->plugIn()[1]->isDataAvailable()==true){
        type = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    }else{
        if(domain.size()==1){
            type = Type2Id<MatN<1,unsigned char>  >::id[1];
        }
        else if(domain.size()==2){
            type = Type2Id<MatN<2,unsigned char>  >::id[1];
        }
        else if(domain.size()==3){
            type = Type2Id<MatN<3,unsigned char>  >::id[1];
        }else
        {
            type = Type2Id<MatN<4,unsigned char>  >::id[1];
        }
    }

    BaseMatN * h;
    h =SingletonFactoryMatN::getInstance()->createObject(type);
    foo func;
    try{Dynamic2Static<TListImgGrid>::Switch(func,h, domain,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }


    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}


COperator * OperatorBlankMatN::clone(){
    return new OperatorBlankMatN();
}
