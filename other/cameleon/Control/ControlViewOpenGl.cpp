#include "ControlViewOpenGl.h"
#include <QtGui>
#include <QtOpenGL>
#include <cmath>
#include<DataOpenGl.h>
#include<DataImageGrid.h>
#include"dependency/ConvertorQImage.h"
GLWidget2::GLWidget2(QWidget *parent)
    : QGLWidget(parent),scale_global(1),_f(NULL)
{

    xRot = 0;
    yRot = 0;
    zRot = 0;

    ra=0.4;
    ga=0.4;
    ba=0.4;
    this->setWindowFlags(Qt::Widget);
    this->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    //    this->setGeometry(0,0,600,600);

}
GLWidget2::~GLWidget2()
{
    //    makeCurrent();

}



void GLWidget2::initializeGL()
{
    //    GLfloat lightSpecular[] = {1.00, 1.00, 1.00, 1.00};

    //    GLfloat matSpecular[]={1,1,1};
    //    GLfloat matShininess[]={100.0};



    if(this->_f==NULL)
    {
        GLfloat lightAmbient[]={0.5,0.5,0.5};
        GLfloat lightDiffuse[]={0.5,0.5,0.5};
        glDepthFunc(GL_LEQUAL);
        glLightfv(GL_LIGHT0,GL_DIFFUSE,lightDiffuse);
        glLightfv(GL_LIGHT0,GL_AMBIENT,lightAmbient);

    }else{
        //        GLfloat lightAmbient[]={0.5,0.5,0.5,1.0};
        //        GLfloat lightDiffuse[]={0.5,0.5,0.5,1.0};
        GLfloat lightAmbient[3];
        lightAmbient[0]=this->_f->getAmbient().r();
        lightAmbient[1]=this->_f->getAmbient().g();
        lightAmbient[2]=this->_f->getAmbient().b();
        GLfloat lightDiffuse[3];
        lightDiffuse[0]=this->_f->getDiffuse().r();
        lightDiffuse[1]=this->_f->getDiffuse().g();
        lightDiffuse[2]=this->_f->getDiffuse().b();
        glDepthFunc(GL_LEQUAL);
        glLightfv(GL_LIGHT0,GL_DIFFUSE,lightDiffuse);
        glLightfv(GL_LIGHT0,GL_AMBIENT,lightAmbient);
    }

    //    glFrontFace(GL_FRONT_AND_BACK);

    //definit les proprit-A-As du matriaux-b-b
    //    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, matSpecular);
    //    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, matShininess);

    //permet de faire capter la lumi-A-Are a toutes les facettes (2 cots)-b-b
    //    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);


    //    glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER,local_view);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);


    //With transparency
    //    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    if(this->_f==NULL){

        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        //        glEnable(GL_DEPTH_TEST);
        //        glAlphaFunc ( GL_GREATER, 0.1 ) ;
        //        glEnable ( GL_ALPHA_TEST ) ;
    }
    // no transparent mode
    else if(this->_f->getTransparentMode()==0){
        glBlendFunc(GL_SRC_ALPHA,GL_ONE);
        glEnable(GL_BLEND);     // Turn Blending On
        glDisable(GL_DEPTH_TEST);
    }
    else{
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        //        glAlphaFunc ( GL_GREATER, 0.1 ) ;
        //        glEnable ( GL_ALPHA_TEST ) ;
    }
    glClearColor(0.0F,0.0F,0.0F,0.0F);
}
double anglealpha =0;


void GLWidget2::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    //glColorMaterial(GL_FRONT_AND_BACK, GL_EMISSION);
    glPushMatrix();
    double scale=1;
    if(this->_max(0)-this->_min(0)!=0)
        scale = min(scale,1./(this->_max(0)-this->_min(0)));
    if(this->_max(1)-this->_min(1)!=0)
        scale = min(scale,1./(this->_max(1)-this->_min(1)));
    if(this->_max(2)-this->_min(2)!=0)
        scale = min(scale,1./(this->_max(2)-this->_min(2)));

    scale *=10;
    glScalef(scale*scale_global, scale*scale_global, scale*scale_global);




    // Draw the triangles

    VecN<3,pop::F64> trans;
    glRotated(-90, 1.0, 0.0, 0.0);
    glRotated(xRot, 1.0, 0.0, 0.0);
    glRotated(yRot , 0.0, 1.0, 0.0);
    glRotated(zRot, 0.0, 0.0, 1.0);
    trans = (this->_max-this->_min)/2+this->_min;
    glTranslatef(-trans(0),-trans(1),-trans(2));


    if(this->_f!=NULL)
    {
        for(int i =0;i<(int)this->_f->vfigure.size();i++)
        {
            if(i==0)
            {
                this->_f->vfigure[i]->callBeginMode();
            }
            else if(typeid(*(this->_f->vfigure[i]))!=typeid(*(this->_f->vfigure[i-1])))
            {
                this->_f->vfigure[i]->callBeginMode();
            }
            this->_f->vfigure[i]->draw();
            if(i==(int)this->_f->vfigure.size()-1)
            {
                this->_f->vfigure[i]->callEndMode();
            }
            else if(typeid(*(this->_f->vfigure[i]))!=typeid(*(this->_f->vfigure[i+1])))
            {
                this->_f->vfigure[i]->callEndMode();
            }
        }
    }
    glPopMatrix();
    emit paintCall();
}

void GLWidget2::resizeGL(int width, int height)
{
    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, side, side);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1.0, +1.0, -1.0, 1.0, 5.0, 60.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslated(0.0, 0.0, -40.0);
}
void GLWidget2::rotateBy(int xAngle, int yAngle, int zAngle)
{
    xRot = xAngle;
    yRot = yAngle;
    zRot = zAngle;
    //    updateGL();
}

void GLWidget2::setScene(Scene3d * f)
{
    _f = f;
    _max = -numeric_limits<pop::F64>::max();

    _min = numeric_limits<pop::F64>::max();
    for(int i =0;i<(int)this->_f->vfigure.size();i++)
    {
        _max = std::max(_max, this->_f->vfigure[i]->getMax());
        _min = std::min(_min, this->_f->vfigure[i]->getMin());
    }
    this->updateGL();
}

QPixmap GLWidget2::getPixmap(){
    return buffer;
}

ControlViewOpenGL::ControlViewOpenGL()
{
    this->path().push_back("OpenGl");
    this->setKey("ViewOpenGL");
    this->setName("ViewOpenGL");
    this->structurePlug().addPlugIn(DataOpenGl::KEY,"in.opengl");
    this->structurePlug().addPlugOut(DataMatN::KEY,"screenshot.pgm");


    scrollbarx = new QScrollBar ;
    scrollbarx->setMinimum(0);
    scrollbarx->setValue(20);
    scrollbarx->setMaximum(360);
    scrollbary = new QScrollBar ;
    scrollbary->setMinimum(0);
    scrollbary->setMaximum(360);
    scrollbarz = new QScrollBar ;
    scrollbarz->setMinimum(0);
    scrollbarz->setValue(45);
    scrollbarz->setMaximum(360);


    lab = new QLabel();
    this->glwidget = new  GLWidget2();
    lab->setPixmap(glwidget->renderPixmap());
    this->glwidget->update();
    lab->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    //    vlayout->addWidget(lab);

    //    if(!connect(glwidget,SIGNAL(paintCall()),this,SLOT(bufferUpdated())))
    //    {
    //        //qDebug << "[WARN] Can't connect CDatasEditor and button" ;
    //    }
    hlayout1 = new QHBoxLayout();
    hlayout1->addWidget(lab);


        hlayout2 = new QHBoxLayout();
        hlayout2->addLayout(hlayout1);
        hlayout2->addWidget(scrollbarx);
        hlayout2->addWidget(scrollbary);
        hlayout2->addWidget(scrollbarz);
        hlayout2->setMargin(5);
        hlayout2->setSpacing(5);
    this->setLayout(hlayout2);

    if(!connect(scrollbarx,SIGNAL(valueChanged(int)),this,SLOT(rotate())))
    {
        //qDebug << "[WARN] Can't connect CDatasEditor and button" ;
    }
    if(!connect(scrollbary,SIGNAL(valueChanged(int)),this,SLOT(rotate())))
    {
        //qDebug << "[WARN] Can't connect CDatasEditor and button" ;
    }

    if(!connect(scrollbarz,SIGNAL(valueChanged(int)),this,SLOT(rotate())))
    {
        //qDebug << "[WARN] Can't connect CDatasEditor and button" ;
    }
    this->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    this->setMinimumSize(300,300);
}
void ControlViewOpenGL::resizeEvent ( QResizeEvent * event ){
    CControl::resizeEvent(event);
    QRect rec(0,0,event->size().width()-70,event->size().height());
    //    //    hlayout1->setGeometry(rec);

    //    hlayout1->
    glwidget->setGeometry(rec);
    //    //    glwidget->update();

    QMetaObject::invokeMethod(this, "bufferUpdated");

}

void ControlViewOpenGL::wheelEvent(QWheelEvent *e){
    if(e->delta()>0)
        this->glwidget->scale_global*=1.1;
    else
        this->glwidget->scale_global*=0.9;
    this->bufferUpdated();

}
void ControlViewOpenGL::bufferUpdated(){
    //qDebug << "void ControlViewOpenGL::bufferUpdated()";
    pix = glwidget->renderPixmap();
    lab->setPixmap(pix);
    lab->setMinimumSize(0,0);
    if(this->isPlugOutConnected(0)==true){
        MatN<2,RGBUI8 > *img = new MatN<2,RGBUI8 >;
         *img = ConvertorQImage::fromQImage<2,RGBUI8 >(pix.toImage());
        DataMatN * b = new DataMatN;
        b->setData(shared_ptr<BaseMatN>(img));
        this->sendPlugOutControl(0,b,CPlug::NEW);
    }
}

void ControlViewOpenGL::updatePlugInControl(int ,CData* data)
{

    //qDebug << "void ControlViewOpenGL::updatePlugInControl()";
    f = dynamic_cast<DataOpenGl*>(data)->getData();
    this->glwidget->setScene(f.get());
    rotate();
    update();
    //    this->glwidget->update();



}


void ControlViewOpenGL::rotate()
{
    glwidget->rotateBy(scrollbarx->value(),scrollbary->value(),scrollbarz->value());
    glwidget->update();
    this->bufferUpdated();
}
CControl * ControlViewOpenGL::clone(){
    return new ControlViewOpenGL;
}
