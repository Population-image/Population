#include "OperatorResizeImageGrid.h"

#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorResizeMatN::OperatorResizeMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorResizeImageGrid");
    this->setName("resize");
    this->setInformation("d = domain(h)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"h.pgm");
    this->structurePlug().addPlugIn(DataPoint::KEY,"d.v");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");


}

void OperatorResizeMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    VecF64  x = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    foo func;
    BaseMatN * fc1= f1.get();
    BaseMatN * h;
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,x,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be registered type");
        else
            this->error(msg.what());
        return;
    }
       dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}


COperator * OperatorResizeMatN::clone(){
    return new OperatorResizeMatN();
}
