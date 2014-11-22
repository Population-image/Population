#include "OperatorCropImageGrid.h"


#include "OperatorCropImageGrid.h"


#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorCropMatN::OperatorCropMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorCropImageGrid");
    this->setName("crop");
    this->setInformation("h(x)= f(x+xmin) with domain(h) = (xmax-xmin)+1 \n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataPoint::KEY,"xmin.v");
    this->structurePlug().addPlugIn(DataPoint::KEY,"xmax.v");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorCropMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    VecF64  xmin =    dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    VecF64  xmax =    dynamic_cast<DataPoint *>(this->plugIn()[2]->getData())->getValue();

    BaseMatN * h;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,xmin,xmax,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
        else
            this->error(msg.what());
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorCropMatN::clone(){
    return new OperatorCropMatN();
}
