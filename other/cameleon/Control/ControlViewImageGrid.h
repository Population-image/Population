#ifndef CONTROLVIEWMatN_H
#define CONTROLVIEWMatN_H
#include "ViewImage.h"
#include"data/mat/MatN.h"
#include"dependency/ConvertorQImage.h"
using namespace pop;
class ControlViewMatN : public ViewImage
{
public:
    ControlViewMatN(QWidget * parent=0);
    virtual CControl * clone();
    virtual void updatePlugInControl(int indexplugin,CData* data);
};
class ControlViewLabelMatN : public ViewImage
{
public:
    ControlViewLabelMatN(QWidget * parent=0);
    virtual CControl * clone();
    virtual void updatePlugInControl(int indexplugin,CData* data);
};
class ControlView3DMatN : public ViewImage
{
    Q_OBJECT
public:
    ControlView3DMatN(QWidget * parent=0);
    virtual CControl * clone();
    virtual void updatePlugInControl(int indexplugin,CData* data);
public slots:
    void sliderMove(int m);
protected:
    QSlider * slider;
    QLabel* current;
    int color;
    MatN<3,unsigned char> img_UC;
    MatN<3,RGBUI8 > img_color;
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> *in,MatN<3,RGBUI8 >& ,MatN<3,unsigned char> & img_grey, int &type){
            img_grey = *in;
            type = 0;
        }
        template<int DIM>
        void operator()(MatN<DIM,RGBUI8 > *in,MatN<3,RGBUI8 >& img_color,MatN<3,unsigned char> & , int& type){
            img_color = *in;
            type = 1;
        }
    };
};

class ControlView3DMatNLabel : public ControlView3DMatN
{
public:
    ControlView3DMatNLabel(QWidget * parent=0);
    virtual CControl * clone();
    virtual void updatePlugInControl(int indexplugin,CData* data);
};
#endif // CONTROLVIEWMatN_H
