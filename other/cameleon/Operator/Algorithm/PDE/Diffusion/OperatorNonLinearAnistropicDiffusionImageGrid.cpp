#include "OperatorNonLinearAnistropicDiffusionImageGrid.h"
#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorNonLinearAnistropicDiffusionMatN::OperatorNonLinearAnistropicDiffusionMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("PDE");
    this->setKey("PopulationOperatorNonLinearAnistropicDiffusionImageGrid");
    this->setName("nonLinearAnisotropicDiffusionDericheFast");
    this->setInformation("Non linear anisotropic diffusion (Kappa is little bit smaller than the grey-level dynamic between phases observed in the histogram)");
    this->structurePlug().addPlugIn(DataMatN::KEY,"init.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"stepsnumber.pgm(by default 20)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"Kappa.num(by default value 50)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"Sigma.num(by default value 1)");
    this->structurePlug().addPlugOut(DataMatN::KEY,"out.pgm");
}

void OperatorNonLinearAnistropicDiffusionMatN::exec(){
    shared_ptr<BaseMatN> init = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    int nbrstep=20;
    if(this->plugIn()[1]->isDataAvailable()==true)
        nbrstep = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    double  kappa=50;
    if(this->plugIn()[2]->isDataAvailable()==true)
        kappa = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    double sigma=1;
    if(this->plugIn()[3]->isDataAvailable()==true)
        sigma = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();

    foo func;

    BaseMatN * initc= init.get();
    BaseMatN * end;
    if(MatN<2,RGBUI8> * imgcolor =  dynamic_cast<MatN<2,RGBUI8> *>(initc)){
        MatN<2,pop::UI8> r,g,b;
        Convertor::toRGB(*imgcolor,r,g,b);
        r = PDE::nonLinearAnisotropicDiffusionDericheFast( r,nbrstep,kappa,sigma);
        g = PDE::nonLinearAnisotropicDiffusionDericheFast( g,nbrstep,kappa,sigma);
        b = PDE::nonLinearAnisotropicDiffusionDericheFast( b,nbrstep,kappa,sigma);
        MatN<2,RGBUI8> * endcast= new MatN<2,RGBUI8>(r.getDomain());
        Convertor::fromRGB(r,g,b,*endcast);
        end = endcast;
    }
    else{
        try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,initc,nbrstep,kappa,sigma,end,Loki::Type2Type<MatN<2,int> >());}
        catch(pexception msg){
            this->error("Pixel/voxel type must be scalar type");
            return;
        }
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(end));
}

COperator * OperatorNonLinearAnistropicDiffusionMatN::clone(){
    return new OperatorNonLinearAnistropicDiffusionMatN();
}
void OperatorNonLinearAnistropicDiffusionMatN::initState(){
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
