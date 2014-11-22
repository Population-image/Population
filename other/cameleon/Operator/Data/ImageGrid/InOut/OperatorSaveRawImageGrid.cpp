#include "OperatorSaveRawImageGrid.h"

#include<DataImageGrid.h>
#include<DataString.h>
#include<DataBoolean.h>
OperatorSaveRawMatN::OperatorSaveRawMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorSaveRawImageGrid");
    this->setName("saveRaw");
    this->setInformation("SaveRaw image by file");
    this->structurePlug().addPlugIn(DataMatN::KEY,"h.pgm");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataBoolean::KEY,"out.bool");

    this->setInformation("Save the image in raw format in the given file\n");
}
void OperatorSaveRawMatN::exec(){
    shared_ptr<BaseMatN> h = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    string file = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    try{
        foo func;
        BaseMatN *hcast =h.get();
        Dynamic2Static<TListImgGrid>::Switch(func,hcast,file,Loki::Type2Type<MatN<2,int> >());
        dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(true);
    }
    catch(pexception msg){
        this->error(msg.what());
        return;
    }
}

COperator * OperatorSaveRawMatN::clone(){
    return new OperatorSaveRawMatN();
}
