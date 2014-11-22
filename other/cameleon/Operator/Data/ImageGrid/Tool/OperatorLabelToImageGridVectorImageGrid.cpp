#include "OperatorLabelToImageGridVectorImageGrid.h"

#include "OperatorLabelToImageGridVectorImageGrid.h"

#include<DataImageGrid.h>
#include<DataVector.h>
OperatorLabelToMatNVectorMatN::OperatorLabelToMatNVectorMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorLabelToImageGridVectorImageGrid");
    this->setName("labelToImageVector");
    this->setInformation("Decompose the input image in a vector of image where each image contains a label of the input image,\n vector(h)=$(h_0,...,h_n)$ where $h_i$(x)=255 for f(x)=i, 0 otherwise");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataVector::KEY,"vector(h).vec");
}
void OperatorLabelToMatNVectorMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    vector<BaseMatN *> h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    shared_ptr<vector<shared_ptr<CData> > > v_data (new vector<shared_ptr<CData> >);
    for(int i=0;i<(int)h.size();i++){
        DataMatN * data = new DataMatN;
        data->setData(shared_ptr<BaseMatN>(h[i]));
        v_data->push_back(shared_ptr<CData>(data));
    }
    dynamic_cast<DataVector  *>(this->plugOut()[0]->getData())->setData(v_data);
}

COperator * OperatorLabelToMatNVectorMatN::clone(){
    return new OperatorLabelToMatNVectorMatN();
}
