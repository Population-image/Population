#include "OperatorConvert1ByteImageGrid.h"
#include "OperatorConvertColor2GreyImageGrid.h"
#include<DataImageGrid.h>
OperatorConvert1ByteMatN::OperatorConvert1ByteMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvert1ByteImageGrid");
    this->setName("convert1Byte");
    this->setInformation("h(x)=f(x) with a pixel type coded in one byte for h\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"label.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorConvert1ByteMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        try{
            OperatorConvertColor2GreyMatN::foo func2;
            Dynamic2Static<TListImgGridRGB>::Switch(func2,fc1,h,Loki::Type2Type<MatN<2,int> >());
        }catch(pexception msg){
            this->error("Pixel/voxel type of input image must be registered type");
            return;
        }
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorConvert1ByteMatN::clone(){
    return new OperatorConvert1ByteMatN();
}

