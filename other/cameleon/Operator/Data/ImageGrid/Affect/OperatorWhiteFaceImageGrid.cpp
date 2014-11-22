#include "OperatorWhiteFaceImageGrid.h"

#include<DataImageGrid.h>
#include<DataMatrix.h>
#include<DataNumber.h>
OperatorWhiteFaceMatN::OperatorWhiteFaceMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Affect");
    this->setKey("PopulationOperatorWhiteFaceImageGrid");
    this->setName("setWhiteFace");
    this->setInformation("h(x)= f(x) for x does not belong to the face at the given coordinate\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"coordinate.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"LeftRight.num (left or down=0 and right or top=1)");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorWhiteFaceMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int coordinate =    dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    int face =    dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    BaseMatN * h;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1, coordinate,face, h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}


COperator * OperatorWhiteFaceMatN::clone(){
    return new OperatorWhiteFaceMatN();
}

