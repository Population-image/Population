#include "OperatorConvert4ByteImageGrid.h"

#include<DataImageGrid.h>
OperatorConvert4ByteMatN::OperatorConvert4ByteMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvert4ByteImageGrid");
    this->setName("convert4Byte");
    this->setInformation("h(x)=f(x) with a pixel type coded in four byte for h\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"label.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}



void OperatorConvert4ByteMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalarAndRGB>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorConvert4ByteMatN::clone(){
    return new OperatorConvert4ByteMatN();
}
