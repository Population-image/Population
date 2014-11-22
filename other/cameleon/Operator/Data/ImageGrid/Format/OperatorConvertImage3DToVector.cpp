#include "OperatorConvertImage3DToVector.h"

#include<DataImageGrid.h>
#include<DataVector.h>
#include<QColor>
OperatorConvertImage3DToVector::OperatorConvertImage3DToVector()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvertImage3DToPointImageGrid");
    this->setName("convertImage3DToVector");
    this->setInformation("h(x)=f(x)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"in.pgm");
    this->structurePlug().addPlugOut(DataVector::KEY,"out.vec");
}

void OperatorConvertImage3DToVector::exec(){

    shared_ptr<BaseMatN> f     = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    vector<BaseMatN *> h;
    foo func;

    BaseMatN * fc= f.get();
    typedef  FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<3> >::Result ListFilter;
    try{Dynamic2Static<ListFilter>::Switch(func,fc,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    shared_ptr<vector<shared_ptr<CData> > > v(new vector<shared_ptr<CData> >);
    for(int i=0;i<(int)h.size();i++){
        DataMatN * data = new DataMatN;
        data->setData(shared_ptr<BaseMatN>(h[i]));
        shared_ptr<CData> datac (data);
        v->push_back(datac);
    }
    dynamic_cast<DataVector *>(this->plugOut()[0]->getData())->setData(v);
}

COperator * OperatorConvertImage3DToVector::clone(){
    return new OperatorConvertImage3DToVector();
}
