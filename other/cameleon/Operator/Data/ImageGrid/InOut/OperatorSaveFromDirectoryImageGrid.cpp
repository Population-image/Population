#include "OperatorSaveFromDirectoryImageGrid.h"

#include<DataImageGrid.h>
#include<DataString.h>
#include<DataBoolean.h>
OperatorSaveFromDirectoryMatN::OperatorSaveFromDirectoryMatN()
    :COperator()
{


    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorSaveFromDirectoryImageGrid");
    this->setName("saveFromDirectory");
    this->setInformation("Save all slices of the  3D image f in the  directory dir with the given filename and the extenion,\n for instance dir=/home/vincent/Project/ENPC/ROCK/Seg/  filename=seg and extension=.bmp, will save the slices as follows \n  /home/vincent/Project/ENPC/ROCK/Seg/seg0000.bmp, /home/vincent/Project/ENPC/ROCK/Seg/seg0001.bmp, /home/vincent/Project/ENPC/ROCK/Seg/seg0002.bmp and so one.\n The supported formats are");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataString::KEY,"dir.str");
    this->structurePlug().addPlugIn(DataString::KEY,"filename.str");
    this->structurePlug().addPlugIn(DataString::KEY,"extension.str(by default .pgm)");

        this->structurePlug().addPlugOut(DataBoolean::KEY,"out.bool");
}
void OperatorSaveFromDirectoryMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugIn()[2]->setState(CPlug::EMPTY);


    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);

    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorSaveFromDirectoryMatN::exec(){
    shared_ptr<BaseMatN> h = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    string dir      = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    string filename = dynamic_cast<DataString *>(this->plugIn()[2]->getData())->getValue();
    string extension=".pgm";
    if(this->plugIn()[3]->isDataAvailable()==true)
        extension = dynamic_cast<DataString *>(this->plugIn()[3]->getData())->getValue();

    foo f;
    BaseMatN * hc = h.get();
    typedef FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<3> >::Result ListFilter;
    try{
        Dynamic2Static<ListFilter>::Switch(f,hc,dir,filename,extension,Loki::Type2Type<MatN<2,int> >());
        dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(true);
    }
    catch(pexception msg)
    {
        if(msg.what()[0]=='P')
            this->error("Dimension of input image must be 3D");
        else
            this->error(msg.what());
        return;
    }

}

COperator * OperatorSaveFromDirectoryMatN::clone(){
    return new OperatorSaveFromDirectoryMatN();
}
