#include "OperatorSaveImageGrid.h"

#include<DataImageGrid.h>
#include<DataString.h>
#include<DataBoolean.h>
#include<QImage>
#include"dependency/ConvertorQImage.h"
OperatorSaveMatN::OperatorSaveMatN()
    :COperator()
{



    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorSaveImageGrid");
    this->setName("save");
    this->setInformation("Save image by file with the format given by the extention.\n For a 3d image the supported format is only pgm and for a 2d image, You should use pgm however  the supported formats are");
    this->structurePlug().addPlugIn(DataMatN::KEY,"h.pgm(or h.bmp)");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataBoolean::KEY,"out.bool");
}

void OperatorSaveMatN::exec(){
    shared_ptr<BaseMatN> h = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    string file = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    try{
        std::string ext( UtilityString::getExtension(file.c_str()));
        if(ext==".pgm")
        {
            h->save(file.c_str());
        }else
        {
            QImage img =ConvertorQImage::toQImage(h.get());
            img.save(file.c_str());
        }

        dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(true);
    }
    catch(pexception msg){
        this->error(msg.what());
        return;
    }
}

COperator * OperatorSaveMatN::clone(){
    return new OperatorSaveMatN();
}
