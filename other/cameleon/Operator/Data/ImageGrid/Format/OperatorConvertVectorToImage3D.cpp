#include "OperatorConvertVectorToImage3D.h"

#include<DataImageGrid.h>
#include<DataVector.h>
#include<QColor>
OperatorConvertVectorToImage3D::OperatorConvertVectorToImage3D()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvertVectorToImage3DImageGrid");
    this->setName("convertVectorToImage3D");
    this->setInformation("h(x)=f(x)\n");
    this->structurePlug().addPlugIn(DataVector::KEY,"out.vec");
    this->structurePlug().addPlugOut(DataMatN::KEY,"in.pgm");

}

void OperatorConvertVectorToImage3D::exec(){
    shared_ptr<vector<shared_ptr<CData> > > f     = dynamic_cast<DataVector  *>(this->plugIn()[0]->getData())->getData();
    foo func;
    vector<shared_ptr<BaseMatN>  > v;
    vector<BaseMatN *   > vv;
    for(int i =0;i<(int)f->size();i++){
        if(DataMatN * data =  dynamic_cast<DataMatN *>(f->operator [](i).get())){
            shared_ptr<BaseMatN> tr = data->getData();
            v.push_back(tr);
            vv.push_back(tr.get());
        }
        else{
            this->error("Pixel/voxel type of input image must be registered type");
            return;
        }
    }

    if(v.empty()==false){
        BaseMatN * h;
        typedef  FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<2> >::Result ListFilter;
        try{
            Dynamic2Static<ListFilter>::Switch(func, vv[0], vv,  h,Loki::Type2Type<MatN<2,int> >());
        }
        catch(pexception msg){
            this->error("Pixel/voxel type of input image must be registered type");
            return;
        }
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
    }
    else{
        this->error("Empty input Point");
        return;
    }
}

COperator * OperatorConvertVectorToImage3D::clone(){
    return new OperatorConvertVectorToImage3D();
}
