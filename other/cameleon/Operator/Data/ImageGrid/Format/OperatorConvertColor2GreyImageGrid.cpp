#include "OperatorConvertColor2GreyImageGrid.h"

#include<DataImageGrid.h>
OperatorConvertColor2GreyMatN::OperatorConvertColor2GreyMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvertColor2GreyImageGrid");
    this->setName("convertColor2Grey");
    this->setInformation("grey(x)=luminance(color(x)) http://fr.wikipedia.org/wiki/Luminance\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"color.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"grey.pgm");
}
void OperatorConvertColor2GreyMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * grey;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridRGB>::Switch(func,fc1,grey,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(f1);
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(grey));
}

COperator * OperatorConvertColor2GreyMatN::clone(){
    return new OperatorConvertColor2GreyMatN();
}
