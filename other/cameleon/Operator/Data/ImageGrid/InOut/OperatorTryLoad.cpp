#include "OperatorTryLoad.h"

#include<DataImageGrid.h>
#include<DataString.h>
#include<DataBoolean.h>
#include<QImage>
#include"dependency/ConvertorQImage.h"
OperatorTryLoad::OperatorTryLoad()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorTryLoad");
    this->setName("tryLoad");
    this->setInformation("try to load image by file. If ok, the result.bool data is set to true, false otherwise.");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataBoolean::KEY,"result.bool");
}

void OperatorTryLoad::exec(){

    string file = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
    DataBoolean * dret = dynamic_cast<DataBoolean *>(this->plugOut()[1]->getData());
    try{
        BaseMatN * h;
        std::string ext( UtilityString::getExtension(file.c_str()));
        if(ext==".pgm")
        {
            h = BaseMatN::create(file);
        }else
        {
            QImage img;
            img.load(file.c_str());
            if(img.isGrayscale()==true){
                Mat2UI8 * hcast = new Mat2UI8;
                *hcast = ConvertorQImage::fromQImage<2,pop::UI8>(img);
                h =hcast;

            }else{
                Mat2RGBUI8 * hcast = new Mat2RGBUI8;
                *hcast = ConvertorQImage::fromQImage<2,pop::RGBUI8>(img);
                h =hcast;
            }


        }
        dynamic_cast<DataString *>(this->plugOut()[0]->getData())->setValue(file);
        dret->setValue(true);
    }
    catch(pexception msg){
        dynamic_cast<DataString *>(this->plugOut()[0]->getData())->setValue(file);
        dret->setValue(false);
    }

}

COperator * OperatorTryLoad::clone(){
    return new OperatorTryLoad();
}
