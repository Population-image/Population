#include "OperatorConvertScalar2ComplexImageGrid.h"

#include<DataImageGrid.h>
OperatorConvertScalar2ComplexMatN::OperatorConvertScalar2ComplexMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvertScalar2ComplexImageGrid");
    this->setName("convertScalarToComplex");
    this->setInformation("complex(x)=real(x) + i img(x) with img(x)=0 for default value\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"real.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"img.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"complex.pgm");
}
void OperatorConvertScalar2ComplexMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorConvertScalar2ComplexMatN::exec(){
    shared_ptr<BaseMatN> real = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    foo func;
    BaseMatN * realc= real.get();
    BaseMatN *complex;

    if(this->plugIn()[1]->isDataAvailable()==true){
        shared_ptr<BaseMatN> img = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
        BaseMatN * imgc= img.get();
        try{Dynamic2Static<TListImgGridScalar>::Switch(func,realc,imgc,complex,Loki::Type2Type<MatN<2,int> >());}
        catch(pexception msg){
            if(msg.what()[0]=='P')
                this->error("Pixel/voxel type of input image must be scalar Byte");
            else
                this->error(msg.what());
            return;
        }
    }
    else {
        try{Dynamic2Static<TListImgGridScalar>::Switch(func,realc,complex,Loki::Type2Type<MatN<2,int> >());}
        catch(pexception msg){
            if(msg.what()[0]=='P')
                this->error("Pixel/voxel type of input image must be scalar Byte");
            else
                this->error(msg.what());
            return;
        }
    }

    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(complex));
}

COperator * OperatorConvertScalar2ComplexMatN::clone(){
    return new OperatorConvertScalar2ComplexMatN();
}
