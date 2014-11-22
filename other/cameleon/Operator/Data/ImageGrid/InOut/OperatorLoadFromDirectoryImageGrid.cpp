#include "OperatorLoadFromDirectoryImageGrid.h"

#include<DataImageGrid.h>
#include<DataString.h>
#include<QImage>

OperatorLoadFromDirectoryMatN::OperatorLoadFromDirectoryMatN()
    :COperator()
{



    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorLoadFromDirectoryImageGrid");
    this->setName("loadFromDirectory");
    this->setInformation("Load a stack of images from a directory where the supported format are");
    this->structurePlug().addPlugIn(DataString::KEY,"dir.str");
    this->structurePlug().addPlugIn(DataString::KEY,"basefilename.str");
    this->structurePlug().addPlugIn(DataString::KEY,"extension.str");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorLoadFromDirectoryMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);





    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorLoadFromDirectoryMatN::exec(){
    try{
        string pathdir = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
        string basefilename = "";
        if(this->plugIn()[1]->isDataAvailable()==true)
            basefilename =  dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
        string extension = "";
        if(this->plugIn()[2]->isDataAvailable()==true)
            extension =  dynamic_cast<DataString *>(this->plugIn()[2]->getData())->getValue();



        std::vector<std::string> vec =UtilityString::getFilesInDirectory(std::string(pathdir));
        std::string ext =extension;
        if(ext!=""&&ext[0]!='.')
            ext="."+ext;
        for(int i = 0;i<(int)vec.size();i++){
            if(ext!=""&& ext!=(UtilityString::getExtension(vec[i])))
            {
                vec.erase(vec.begin()+i);
                i--;
            }else if(std::string(basefilename)!=""&&vec[i].find(std::string(basefilename))==std::string::npos)
            {
                vec.erase(vec.begin()+i);
                i--;
            }else{
                vec[i]=pathdir+"/"+vec[i];
            }
        }
        BaseMatN * img=NULL;
        ext =  UtilityString::getExtension(vec[0]);
        if(ext==".pgm")
        {
            img = BaseMatN::create(vec[0].c_str());
        }else
        {
            QImage qimg;
            qimg.load(vec[0].c_str());
            if(qimg.isGrayscale()==true){
                Mat2UI8 * hcast = new Mat2UI8;
                *hcast = ConvertorQImage::fromQImage<2,pop::UI8>(qimg);
                img =hcast;

            }else{
                Mat2RGBUI8 * hcast = new Mat2RGBUI8;
                *hcast = ConvertorQImage::fromQImage<2,pop::RGBUI8>(qimg);
                img =hcast;
            }
        }
        BaseMatN * h;
        sort (vec.begin(), vec.end());
        foo func;
        Dynamic2Static<TListImgGrid>::Switch(func,img,vec,h,Loki::Type2Type<MatN<2,int> >());
        delete img;
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
    }
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }








}

COperator * OperatorLoadFromDirectoryMatN::clone(){
    return new OperatorLoadFromDirectoryMatN();
}
