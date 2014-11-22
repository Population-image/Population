#ifndef CONTROLMARKERMatN_H
#define CONTROLMARKERMatN_H
#include "MarkerImage.h"
#include"data/mat/MatN.h"
using namespace pop;
class ControlMarkerMatN : public MarkerImage
{
public:
    ControlMarkerMatN(QWidget * parent=0);
    virtual ControlMarkerMatN * clone();
    virtual void apply();
};


class ControlMarker3DMatN : public MarkerImage
{
    Q_OBJECT
public:
    ControlMarker3DMatN(QWidget * parent=0);
    virtual ControlMarker3DMatN * clone();
    virtual void apply();
    virtual void updatePlugInControl(int indexplugin,CData* data);
    QSlider * slider;

public slots:
    void updateImage();
        virtual void clear();
private:
    QSlider * __slider;
    MatN<3,unsigned char>  inputgrey;
    MatN<3,RGBUI8 >  inputcolor;
    MatN<3,RGBUI8 >  marker;
    int color;
    int indexold;
    QLabel *valuez;
};

#endif // CONTROLMARKERMatN_H
