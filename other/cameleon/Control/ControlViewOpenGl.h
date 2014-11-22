#ifndef CONTROLVIEWOPENGL_H
#define CONTROLVIEWOPENGL_H

#include "iostream"
#include <vector>
using namespace std;
#include <QObject>
#include <QtGui>
#include "CData.h"
#include "CPlug.h"




#include <CControl.h>
#include <QGLWidget>
#include <CDataByFile.h>
#include"data/3d/GLFigure.h"
using namespace pop;
class GLWidget2 : public QGLWidget
{
    Q_OBJECT
signals:
    void paintCall();

public:
    int xRot;
    int yRot;
    int zRot;
    GLWidget2(QWidget *parent = 0);
    ~GLWidget2();
    void setScene(Scene3d * f);
       void rotateBy(int xAngle, int yAngle, int zAngle);

          double scale_global;
protected:
    //void keyPressEvent(QKeyEvent * event);


   void setAmbiant(GLfloat ra, GLfloat ga,GLfloat ba);
   void setDiffuse(GLfloat rd, GLfloat gd,GLfloat bd);
   void setSpecular(GLfloat rs, GLfloat gs,GLfloat bs);
   QPixmap getPixmap();
protected:
    virtual void initializeGL();
    virtual void paintGL();
    void resizeGL(int width, int height);
private:
   GLfloat ra; GLfloat ga;GLfloat ba;
   GLfloat rd; GLfloat gd;GLfloat bd;
   GLfloat rs; GLfloat gs;GLfloat bs;
   Scene3d * _f;
   VecN<3,pop::F32> _min;
   VecN<3,pop::F32> _max;

   QPixmap buffer;
};



class ControlViewOpenGL: public CControl
{
    Q_OBJECT
public slots:
    void rotate();
    void bufferUpdated();
public:
    ControlViewOpenGL();
    virtual CControl * clone();

    void updatePlugInControl(int indexplugin,CData* data);
    GLWidget2 *glwidget;
    QScrollBar * scrollbarx;
    QScrollBar * scrollbary;
    QScrollBar * scrollbarz;
    shared_ptr<Scene3d > f;
protected:
    void resizeEvent ( QResizeEvent * event );
    void wheelEvent(QWheelEvent *e);
    QLabel* lab;
    QPixmap pix;
    QImage img;
    QHBoxLayout * hlayout1;
    QHBoxLayout * hlayout2;
};
#endif // CONTROLVIEWOPENGL_H
