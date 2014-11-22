#include "ControlMarkerImageGrid.h"
#include<DataImageGrid.h>
#include"algorithm/Visualization.h"
using namespace pop;
#include<DataNumber.h>
#include"dependency/ConvertorQImage.h"
ControlMarkerMatN::ControlMarkerMatN(QWidget *parent)
    :MarkerImage(parent)
{
    this->path().clear();
    this->path().push_back("ImageGrid");
    this->setName("MarkerImageGrid");
    this->setKey("ControlMarkerImageGrid");
    this->setInformation("Create marker image with background image");
    this->structurePlug().plugIn().clear();
    this->structurePlug().plugOut().clear();
    this->structurePlug().addPlugIn(DataMatN::KEY,"background.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"InitialMarker.pgm (optionnal)");
    this->structurePlug().addPlugOut(DataMatN::KEY,"markergreylevel.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"markercolorlevel.pgm");
}
ControlMarkerMatN * ControlMarkerMatN::clone(){
    return new ControlMarkerMatN;
}
void ControlMarkerMatN::apply(){

    if(_test==true)
    {

        try{
            if(this->isPlugOutConnected(0)==true){
                MatN<2,unsigned char> * grey = new  MatN<2,unsigned char>;
                * grey = ConvertorQImage::fromQImage<2,unsigned char>(*imageToSave);
                DataMatN * data = new DataMatN;
                shared_ptr<BaseMatN> st(grey);
                data->setData(st);
                this->sendPlugOutControl(0,data,CPlug::NEW);
            }
            if(this->isPlugOutConnected(1)==true){
                MatN<2,RGBUI8 > * color = new  MatN<2,RGBUI8 >;
                * color = ConvertorQImage::fromQImage<2,RGBUI8 >(*imageToSave);


                DataMatN * data = new DataMatN;
                shared_ptr<BaseMatN> st2(color);
                data->setData(st2);
                this->sendPlugOutControl(1,data,CPlug::NEW);
            }
        }catch(string ){

        }

        _test=false;
    }

}
ControlMarker3DMatN::ControlMarker3DMatN(QWidget *parent)
    :MarkerImage(parent)
{
    this->path().clear();
    this->path().push_back("ImageGrid");
    this->setName("MarkerMatN3d");
    this->setKey("ControlMarker3DImageGrid");
    this->setInformation("Create marker image with background image");
    this->structurePlug().plugIn().clear();
    this->structurePlug().plugOut().clear();
    this->structurePlug().addPlugIn(DataMatN::KEY,"background.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"InitialMarker.pgm (optionnal)");

    this->structurePlug().addPlugOut(DataMatN::KEY,"markergreylevel.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"markercolorlevel.pgm");
//    this->structurePlug().addPlugOut(DataNumber::KEY,"sliceindex.num");

    QHBoxLayout * layhor = new QHBoxLayout;
    valuez = new QLabel;

    indexold =-1;
    __slider = new QSlider;
    __slider->setMinimum(0);
    __slider->setMaximum(100);
    __slider->setSingleStep(1);
    __slider->setOrientation(Qt::Horizontal);
    layhor->addWidget(__slider);
    layhor->addWidget(valuez);
    this->setLayout(layout);
    this->layout->addLayout(layhor);
    if(!QObject::connect(__slider, SIGNAL(valueChanged(int)),this, SLOT(updateImage()),Qt::DirectConnection)){
        //qDebug << "[WARN] Can't connect CDatasEditor and button" ;
    }
}
void ControlMarker3DMatN::updateImage(){

    //save marker
    if(indexold!=-1){
        MatN<2,RGBUI8> plane;
        plane =ConvertorQImage::fromQImage<2,RGBUI8>(*imageToSave);
        if(plane.getDomain()==marker.getPlaneDomain(2)){
            marker.setPlane(2,indexold,plane);
        }
    }


    if(color==1){
        MatN<2,RGBUI8> plane;
        plane = inputcolor.getPlane(2,__slider->value());
        *image=ConvertorQImage::toQImage(plane);
        *imagebackground=*image;

        plane = marker.getPlane(2,__slider->value());
        *imageToSave=ConvertorQImage::toQImage(plane);
    }else{
        MatN<2,unsigned char> plane;
        plane = inputgrey.getPlane(2,__slider->value());
        *image=ConvertorQImage::toQImage(plane);

        MatN<2,RGBUI8> colorplane;
        colorplane = marker.getPlane(2,__slider->value());
        *imageToSave=ConvertorQImage::toQImage(colorplane);
    }
    *imagebackground=*image;
    for(int i=0;image->width()>i;i++){
        for(int j=0;image->height()>j;j++){
            QRgb rgb  = imageToSave->pixel (i, j);
            if(qRed( rgb )!=0||qGreen( rgb ) !=0||qBlue( rgb )!=0 )
                image->setPixel(i,j,rgb);
        }
    }
    valuez->setText(QString::number(__slider->value()));
    indexold =__slider->value();
    setView();
    this->update();
}

ControlMarker3DMatN * ControlMarker3DMatN::clone(){
    return new ControlMarker3DMatN;
}
void ControlMarker3DMatN::apply(){
    if(this->isPlugOutConnected(0)==true&&_test==true){
        updateImage();
        DataMatN * data = new DataMatN;
        MatN<3,unsigned char> * outcast = new MatN<3,unsigned char>(marker.getDomain());
        *outcast = marker;
        shared_ptr<BaseMatN> h(outcast);
        data->setData(h);
        this->sendPlugOutControl(0,data,CPlug::NEW);
        _test = false;
    }
    if(this->isPlugOutConnected(1)==true&&_test==true){
        updateImage();
        DataMatN * data = new DataMatN;
        MatN<3,RGBUI8> * outcast = new MatN<3,RGBUI8>(marker.getDomain());
        *outcast = marker;
        shared_ptr<BaseMatN> h(outcast);
        data->setData(h);
        this->sendPlugOutControl(1,data,CPlug::NEW);
        _test = false;
    }
    if(this->isPlugOutConnected(2)==true&&_test==true){
        DataNumber * data = new DataNumber;
        data->setValue(__slider->value());
        this->sendPlugOutControl(2,data,CPlug::NEW);
    }

}
void ControlMarker3DMatN::clear(){
    marker.fill(0);
    indexold=-1;
    updateImage();
}

void ControlMarker3DMatN::updatePlugInControl(int indexplugin,CData* data){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(data)->getData();
    try{
        if(indexplugin==0 ){

            if(MatN<3,RGBUI8 > *ccolor= dynamic_cast<MatN<3,RGBUI8 > * >(f.get())){
                inputcolor = *ccolor;
                color = 1;
                if(marker.getDomain()!=inputcolor.getDomain()){
                    marker.resize(inputcolor.getDomain());
                    marker .fill( 0);
                }
                __slider->setMaximum(inputcolor.getDomain()(2)-1);
                updateImage();
            }
            else if(MatN<3,pop::UI8 > *cgrey= dynamic_cast<MatN<3,pop::UI8 > * >(f.get())){
                inputgrey = *cgrey;
                color = 0;
                if(marker.getDomain()!=inputgrey.getDomain()){
                    marker.resize(inputgrey.getDomain());
                    marker .fill( 0);
                }
                __slider->setMaximum(inputgrey.getDomain()(2)-1);
                updateImage();

            }
        }
        else{
            if(MatN<3,RGBUI8 > *ccolor= dynamic_cast<MatN<3,RGBUI8 > * >(f.get())){
                marker = *ccolor;
                updateImage();
            }
            else{
                this->error("Input marker image must be color ImageGrid");
            }
        }
    }catch(pexception msg){
        //        this->error(msg.what());
    }
}
