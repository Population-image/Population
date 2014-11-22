#include "OperatorConvertComplex2ScalarImageGrid.h"

#include<DataImageGrid.h>
OperatorConvertComplex2ScalarMatN::OperatorConvertComplex2ScalarMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvertComplex2ScalarImageGrid");
    this->setName("convertComplex2Scalar");
    this->setInformation("complex(x)=real(x) + i img(x) with img(x)=0 for default value\n");

    this->structurePlug().addPlugIn(DataMatN::KEY,"complex.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"real.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"img.pgm");
}

void OperatorConvertComplex2ScalarMatN::exec(){
    shared_ptr<BaseMatN> complex = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    foo func;
    BaseMatN * complexc= complex.get();
    BaseMatN *real;
    BaseMatN *img;

    try{Dynamic2Static<TListImgGridComplex>::Switch(func,complexc,real,img,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(real));
    dynamic_cast<DataMatN *>(this->plugOut()[1]->getData())->setData(shared_ptr<BaseMatN>(img));
}

COperator * OperatorConvertComplex2ScalarMatN::clone(){
    return new OperatorConvertComplex2ScalarMatN();
}
