#include "OperatorContrastScaleImageGrid.h"


#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorContrastScaleMatN::OperatorContrastScaleMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Point");
    this->setKey("PopulationOperatorContrastScaleImageGrid");
    this->setName("greylevelScaleContrast");
    this->setInformation("Let sigma the root mean square contrast (RMSC)  of the input image, the RMSC of the ouput is equal to sigma*scale");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"scale.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}


void OperatorContrastScaleMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    double scale = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();


    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,scale,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorContrastScaleMatN::clone(){
    return new OperatorContrastScaleMatN();
}

OperatorContrastScaleColorMatN::OperatorContrastScaleColorMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Point");
    this->setKey("PopulationOperatorContrastScaleColor");
    this->setName("ContrastScaleColor");
    this->setInformation("Let sigma the root mean square contrast (RMSC)  of the input image, the RMSC of the ouput is equal to sigma*scale");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"redscale.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"greenscale.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"bluescale.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}


void OperatorContrastScaleColorMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    double r=1;
    if(this->plugIn()[1]->isDataAvailable()==true)
        r = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    double g=1;
    if(this->plugIn()[2]->isDataAvailable()==true)
        g = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    double b=1;
    if(this->plugIn()[3]->isDataAvailable()==true)
        b = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();


    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridRGB>::Switch(func,fc1,r,g,b,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be color type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorContrastScaleColorMatN::clone(){
    return new OperatorContrastScaleColorMatN();
}
void OperatorContrastScaleColorMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
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
