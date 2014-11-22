#include "OperatorLoadImageGrid.h"

#include<DataImageGrid.h>
#include<DataString.h>
#include<QImage>
#include"dependency/ConvertorQImage.h"
#include"QImageReader"
OperatorLoadMatN::OperatorLoadMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorLoadImageGrid");
    this->setName("load");
    this->setInformation("Load image by file where the supported format: bmp, pgm, png");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorLoadMatN::exec(){

    string file = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
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
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
    }
    catch(pexception msg){
        QList<QByteArray>  list =QImageReader::supportedImageFormats ();
        QString  str;
        for(int i=0;i<list.size();i++){
            str+="- "+list[i]+"\n";
        }
        this->error("Cannot read this file image" +file+" \n"+"The supported formats are:\n"+str.toStdString());
    }

}

COperator * OperatorLoadMatN::clone(){
    return new OperatorLoadMatN();
}

