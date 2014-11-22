#include "ControlViewImageGrid.h"
#include<DataImageGrid.h>
#include<OperatorColorRandomFromLabelImageGrid.h>
#include<DataNumber.h>
#include"dependency/ConvertorQImage.h"
ControlViewMatN::ControlViewMatN(QWidget *parent)
    :ViewImage(parent)
{
    this->path().clear();
    this->path().push_back("ImageGrid");
    this->setName("ViewMatN2D");
    this->setKey("ControlImageGrid");
    this->setInformation("View 2D ImageGrid");
    this->structurePlug().plugIn().clear();
    this->structurePlug().addPlugIn(DataMatN::KEY,"in2d.pgm");
}
CControl * ControlViewMatN::clone(){
    return new ControlViewMatN;
}
void ControlViewMatN::updatePlugInControl(int state,CData* data){

    if(state==CPlug::NEW){
        shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(data)->getData();
        try{
            *image= toQImage(f.get());
            if(image->isNull()==false){
                setView();
                this->update();
            }
            else{
                this->error("Cannot read input image");
            }
        }catch(pexception msg){
            this->error(msg.what());
        }
    }
}
ControlViewLabelMatN::ControlViewLabelMatN(QWidget *parent)
    :ViewImage(parent)
{
    this->path().clear();
    this->path().push_back("ImageGrid");
    this->setName("ViewMatNLabel2D");
    this->setKey("ControlMatNLabel2D");
    this->setInformation("View the labels of a 2D ImageGrid");
    this->structurePlug().plugIn().clear();
    this->structurePlug().addPlugIn(DataMatN::KEY,"label2d.pgm");
}
CControl * ControlViewLabelMatN::clone(){
    return new ControlViewLabelMatN;
}

void ControlViewLabelMatN::updatePlugInControl(int ,CData* data){

    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(data)->getData();
    try{
        if(!dynamic_cast<MatN<2,RGBUI8 > * >(f.get())){

            OperatorColorRandomFromLabelMatN::foo func;
            BaseMatN * fc1= f.get();
            BaseMatN * h;
            typedef FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<2> >::Result ListFilter;
            try{Dynamic2Static<ListFilter>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg){
                this->error("Only 2 dimensionnal image");
                return;
            }
            f = shared_ptr<BaseMatN>(h);

        }
        *image= toQImage(f.get());
        if(image->isNull()==false){
            setView();
            this->update();
        }
        else{
            this->error("Cannot read input image");
        }
    }catch(pexception msg){
        this->error(msg.what());
    }

}

ControlView3DMatN::ControlView3DMatN(QWidget * parent)
    :ViewImage(parent){
    this->path().clear();
    this->path().push_back("ImageGrid");
    this->setName("ViewMatN3D");
    this->setKey("ControlViewMatN3D");
    this->setInformation("View 3D ImageGrid");
    this->structurePlug().plugIn().clear();
    this->structurePlug().addPlugIn(DataMatN::KEY,"in3d.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"sliceindex.pgm");
    this->structurePlug().addPlugOut(DataNumber::KEY,"sliceindex.pgm");


    slider = new QSlider;
    slider->setMinimum(0);
    slider->setMaximum(100);
    slider->setSingleStep(1);
    slider->setOrientation(Qt::Horizontal);

    current = new QLabel;
    current->setText("50");

    slider->setValue(50);

    QHBoxLayout *layouth =new QHBoxLayout;
    layouth->addWidget(slider);
    layouth->addWidget(current);
    layout->removeWidget(view);
    layout->addLayout(layouth);
    layout->addWidget(view);
    color =-1;
    if(!QObject::connect(slider, SIGNAL( valueChanged(int)),this, SLOT(sliderMove(int)))){
        //qDebug << "[WARN] Can't connect CDatasEditor and button" ;
    }
}
void ControlView3DMatN::sliderMove(int m){
    current->setText(QString::number(m));
    current->update();
    if(color==0){
        if(slider->maximum()== img_UC.getDomain()(2)-1){
            MatN<2,unsigned char> plane;
            plane = img_UC.getPlane(2,m);
            *image= ConvertorQImage::toQImage(plane);
            if(image->isNull()==false){
                setView();
                this->update();
            }
        }else{
            slider->setMaximum(img_UC.getDomain()(2)-1);
            if(m<=slider->maximum()){
                sliderMove(slider->value());
            }
        }
    }else if (color==1) {
        if(slider->maximum()== img_color.getDomain()(2)-1){
            MatN<2,RGBUI8 > plane;
            plane = img_color.getPlane(2,m);
            *image= ConvertorQImage::toQImage(plane);
            if(image->isNull()==false){
                setView();
                this->update();
            }
        }else{
            slider->setMaximum(img_color.getDomain()(2)-1);
            if(m<=slider->maximum()){
                sliderMove(slider->value());
            }
        }
    }
}
void ControlView3DMatN::updatePlugInControl(int index ,CData* data){

    if(index==0){
        shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(data)->getData();
        try{
            typedef FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<3> >::Result ListFilter;
            foo func;
            BaseMatN * fc= f.get();
            Dynamic2Static<ListFilter>::Switch(func,fc,img_color,img_UC,color, Loki::Type2Type<MatN<2,int> >());
            sliderMove(slider->value());
        }catch(pexception msg){
            this->error("Dimension of the input image must be 3D");
        }
    }else{
        int index = dynamic_cast<DataNumber *>(data)->getValue();
        slider->setValue(index);
        sliderMove(index);
    }

}

CControl * ControlView3DMatN::clone(){
    return new ControlView3DMatN;
}
ControlView3DMatNLabel::ControlView3DMatNLabel(QWidget * parent)
    :ControlView3DMatN(parent){
    this->path().clear();
    this->path().push_back("ImageGrid");
    this->setName("ViewMatNLabel3D");
    this->setKey("ControlViewMatNLabel3D");
    this->setInformation("View 3D labeled ImageGrid");
    this->structurePlug().plugIn().clear();
    this->structurePlug().addPlugIn(DataMatN::KEY,"label3d.pgm");
}

CControl * ControlView3DMatNLabel::clone(){
    return new ControlView3DMatNLabel;
}

void ControlView3DMatNLabel::updatePlugInControl(int ,CData* data){

    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(data)->getData();
    try{
        if(MatN<3,RGBUI8 > *ccolor= dynamic_cast<MatN<3,RGBUI8 > * >(f.get())){
            img_color = *ccolor;
            color = 1;
            sliderMove(slider->value());
        }
        else{
            OperatorColorRandomFromLabelMatN::foo func;
            BaseMatN * fc1= f.get();
            BaseMatN * h;
            typedef FilterKeepTlistTlist<TListImgGridUnsigned,0,Loki::Int2Type<3> >::Result ListFilter;
            try{Dynamic2Static<ListFilter>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg){
                this->error("Only 3 dimensionnal image");
                return;
            }
            MatN<3,RGBUI8 > *ccolor2= dynamic_cast<MatN<3,RGBUI8 > * >(h);
            img_color = *ccolor2;
            delete ccolor2;
            color = 1;
            sliderMove(slider->value());
        }
    }catch(pexception msg){
        this->error(msg.what());
    }

}
