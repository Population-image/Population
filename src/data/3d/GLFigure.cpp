#include"data/3d/GLFigure.h"
#include <algorithm>
#include<cmath>
#include"data/utility/BasicUtility.h"
#include"algorithm/LinearAlgebra.h"
#include"data/mat/MatNInOut.h"
#include"data/mat/MatNDisplay.h"
#include"data/utility/BasicUtility.h"
#include"algorithm/GeometricalTransformation.h"
#include"PopulationConfig.h"
#if defined(HAVE_OPENGL)
#include<typeinfo>
#if Pop_OS==1
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#define HAVE_GLUT
#endif
#if Pop_OS==2
#define WINDOWSOPENGL
#define UNICODE 1
#include <tchar.h>
#include <windows.h>
#include <GL/GL.h>
#include <GL/GLU.h>
#define UNICODE 1
#endif
#include"3rdparty/tinythread.h"
#endif
namespace pop{
void GeometricalFigure::setTransparent(UI8  transparent){_transparant=transparent;}
UI8 GeometricalFigure::getTransparent() const{return _transparant;}
void GeometricalFigure::setRGB(const RGBUI8 &RGB){_RGB =RGB;}
RGBUI8 GeometricalFigure::getRGB() const{return _RGB;}
GeometricalFigure::~GeometricalFigure()
{}
GeometricalFigure::GeometricalFigure()
    :_transparant(255),_RGB(255,255,255)
{}

void GeometricalFigure::callBeginMode(){}
void GeometricalFigure::callEndMode(){}



void FigurePolygon::draw()
{
#if defined(HAVE_OPENGL)
    glBegin(GL_POLYGON);
    glPolygonMode(GL_FRONT_AND_BACK,GL_POLYGON);
    {
        glColor4ub(this->_RGB.r(),this->_RGB.g(),this->_RGB.b(),this->_transparant);

        glNormal3f(normal[0],normal[1],normal[2]);
        for(int i =0;i<(int)vpos.size();i++)
        {
            glVertex3f(vpos[i][0],vpos[i][1],vpos[i][2]);
        }
    }
    glEnd();
#endif
}
VecN<3,F32> FigurePolygon::getMax()const
{
    VecN<3,F32> _max(-100000);

    for(int i =0;i<(int)vpos.size();i++)
    {
        if(vpos[i].allSuperior(_max))
            _max = vpos[i];
    }
    return _max;
}
VecN<3,F32> FigurePolygon::getMin()const
{
    VecN<3,F32> _min(100000);
    for(int i =0;i<(int)vpos.size();i++)
    {
        if(vpos[i].allInferior(_min))
            _min = vpos[i];
    }
    return _min;
}


void FigurePolygon::translate(VecN<3,F32> trans){
    for(unsigned int i =0;i<vpos.size();i++)
    {
        vpos[i]+=trans;
    }
}

FigurePolygon* FigurePolygon::createTriangle(const VecN<3,F32>& x1,const VecN<3,F32>& x2,const VecN<3,F32>& x3,const RGBUI8 & c, UI8 transparence)
{

    VecN<3,F32> n = productVectoriel(x2-x1,x3-x1);
    n/=n.norm();
    FigurePolygon * t = new FigurePolygon;
    t->normal=n;
    t->vpos.push_back(x1);
    t->vpos.push_back(x2);
    t->vpos.push_back(x3);
    t->setRGB(c);
    t->setTransparent(transparence);
    return t;
}
FigurePolygon::FigurePolygon(){
    //   key = FigurePolygon::KEY;
}

FigurePolygon * FigurePolygon::clone()const
{
    FigurePolygon * poly= new FigurePolygon();
    poly->vpos.resize(this->vpos.size());
    std::copy ( this->vpos.begin(),this->vpos.end(), poly->vpos.begin() );
    poly->normal =this->normal;
    poly->_RGB =this->_RGB;
    poly->_transparant =this->_transparant;
    return poly;
}




void FigureTriangle::draw()
{
#if defined(HAVE_OPENGL)
    glColor4ub(this->_RGB.r(),this->_RGB.g(),this->_RGB.b(),this->_transparant);
    glNormal3f(normal[0],normal[1],normal[2]);
    glVertex3f(x[0],x[1],x[2]);
#endif
}





FigureTriangle::FigureTriangle(){
    //   key = FigureTriangle::KEY;
}
GeometricalFigure * FigureTriangle::clone()const
{
    FigureTriangle * triangle = new FigureTriangle();
    triangle->x =this->x;
    triangle->normal =this->normal;
    triangle->_RGB =this->_RGB;
    triangle->_transparant =this->_transparant;
    return triangle;
}




VecN<3,F32> FigureTriangle::getMax()const
{
    return this->x;
}
VecN<3,F32> FigureTriangle::getMin()const
{
    return this->x;
}
void FigureTriangle::callBeginMode()
{
#if defined(HAVE_OPENGL)
    glBegin(GL_TRIANGLES);
#endif
}
void FigureTriangle::callEndMode()
{
#if defined(HAVE_OPENGL)
    glEnd();
#endif
}






void FigureLine::draw()
{
#if defined(HAVE_OPENGL)
    glLineWidth(width);
    glBegin(GL_LINES);
    glColor4ub(this->_RGB.r(),this->_RGB.g(),this->_RGB.b(),this->_transparant);
    glVertex3f(x1[0],x1[1],x1[2]);
    glVertex3f(x2[0],x2[1],x2[2]);
    glEnd();
#endif
}




FigureLine::FigureLine()
    :width(2)
{
    this->setTransparent(255);
    //   key = FigureLine::KEY;
}
GeometricalFigure * FigureLine::clone()const
{
    FigureLine * Line = new FigureLine();
    Line->width = this->width;
    Line->x1 =this->x1;
    Line->x2 =this->x2;
    Line->_RGB =this->_RGB;
    Line->_transparant =this->_transparant;
    return Line;
}


VecN<3,F32> FigureLine::getMax()const
{
    return maximum(this->x1,this->x2);

}
VecN<3,F32> FigureLine::getMin()const
{
    return minimum(this->x1,this->x2);
}
void FigureLine::callBeginMode()
{

}
void FigureLine::callEndMode()
{

}





//static GLuint sphereList;

//GLfloat vBLUE[]    = {0.0, 0.0, 1.0};
#if defined(HAVE_OPENGL)
GLfloat __X = .525731112119133606f;
GLfloat __Z = .850650808352039932f;

static GLfloat vdata[12][3] = {
    {-__X, 0.0, __Z}, {__X, 0.0, __Z}, {-__X, 0.0, -__Z}, {__X, 0.0, -__Z},
    {0.0, __Z, __X}, {0.0, __Z, -__X}, {0.0, -__Z, __X}, {0.0, -__Z, -__X},
    {__Z, __X, 0.0}, {-__Z, __X, 0.0}, {__Z, -__X, 0.0}, {-__Z, -__X, 0.0}
};
static GLuint tindices[20][3] = {
    {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
    {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
    {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
    {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} };

void normalize(GLfloat *a) {
    GLfloat d=std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
    a[0]/=d; a[1]/=d; a[2]/=d;
}

void drawtri(GLfloat *a, GLfloat *b, GLfloat *c, int div, float r) {
    if (div<=0) {
        glNormal3fv(a); glVertex3f(a[0]*r, a[1]*r, a[2]*r);
        glNormal3fv(b); glVertex3f(b[0]*r, b[1]*r, b[2]*r);
        glNormal3fv(c); glVertex3f(c[0]*r, c[1]*r, c[2]*r);
    } else {
        GLfloat ab[3], ac[3], bc[3];
        for (int i=0;i<3;i++) {
            ab[i]=(a[i]+b[i])/2;
            ac[i]=(a[i]+c[i])/2;
            bc[i]=(b[i]+c[i])/2;
        }
        normalize(ab); normalize(ac); normalize(bc);
        drawtri(a, ab, ac, div-1, r);
        drawtri(b, bc, ab, div-1, r);
        drawtri(c, ac, bc, div-1, r);
        drawtri(ab, bc, ac, div-1, r);  //<--Comment this line and sphere looks really cool!
    }
}
void drawsphere(int ndiv, float radius=1.0) {
    glBegin(GL_TRIANGLES);
    for (int i=0;i<20;i++)
        drawtri(vdata[tindices[i][0]], vdata[tindices[i][1]], vdata[tindices[i][2]], ndiv, radius);
    glEnd();
}

#endif
void FigureSphere::draw()
{
#if defined(HAVE_OPENGL)
    glTranslatef(_x[0], _x[1], _x[2]);

    glColor4ub(this->_RGB.r(),this->_RGB.g(),this->_RGB.b(),this->_transparant);
    drawsphere(0,_radius);
    glTranslatef(-_x[0], -_x[1], -_x[2]);
#endif
}





FigureSphere::FigureSphere()
{
    //   key = FigureSphere::KEY;
}

GeometricalFigure * FigureSphere::clone()const
{
    FigureSphere * Sphere = new FigureSphere();
    Sphere->_x =this->_x;
    Sphere->_radius =this->_radius;
    Sphere->_RGB =this->_RGB;
    Sphere->_transparant =this->_transparant;
    return Sphere;
}


VecN<3,F32> FigureSphere::getMax()const
{

    return _x;
}
VecN<3,F32> FigureSphere::getMin()const
{

    return _x;
}
void FigureSphere::callBeginMode()
{

}
void FigureSphere::callEndMode()
{

}


void FigureUnitSquare::draw()
{
#if defined(HAVE_OPENGL)
    glBegin(GL_QUADS); //Begin quadrilateral coordinates
    glColor4ub(this->_RGB.r(),this->_RGB.g(),this->_RGB.b(),this->_transparant);
    //Trapezoid
    if(direction==0){
        glNormal3f(1*way,0,0);
        glVertex3f(x[0], x[1], x[2]);
        glVertex3f(x[0], x[1]+1, x[2]);
        glVertex3f(x[0], x[1]+1, x[2]+1);
        glVertex3f(x[0], x[1], x[2]+1);
    }else if(direction==1){
        glNormal3f(0,1*way,0);
        glVertex3f(x[0] , x[1], x[2]);
        glVertex3f(x[0]+1,x[1], x[2]);
        glVertex3f(x[0]+1,x[1], x[2]+1);
        glVertex3f(x[0] , x[1], x[2]+1);
    }else{
        glNormal3f(0,0,1*way);
        glVertex3f(x[0] , x[1], x[2]);
        glVertex3f(x[0]+1,x[1], x[2]);
        glVertex3f(x[0]+1,x[1]+1, x[2]);
        glVertex3f(x[0] , x[1]+1, x[2]);
    }
    glEnd();
#endif

}



FigureUnitSquare::FigureUnitSquare(){
    //   key =  FigureUnitSquare::KEY;
}
GeometricalFigure * FigureUnitSquare::clone()const
{
    FigureUnitSquare * UnitSquare = new FigureUnitSquare();
    UnitSquare->x =this->x;
    UnitSquare->direction =this->direction;
    UnitSquare->_RGB =this->_RGB;
    UnitSquare->_transparant =this->_transparant;
    return UnitSquare;
}


VecN<3,F32> FigureUnitSquare::getMax()const
{
    return this->x;
}
VecN<3,F32> FigureUnitSquare::getMin()const
{
    return this->x;
}
void FigureUnitSquare::callBeginMode()
{

}
void FigureUnitSquare::callEndMode()
{

}




void FigureCone::draw()
{
#if defined(HAVE_OPENGL)
    //this is to tell OpenGL 3 consecutive vertex's to be the vertices of triangle
    glBegin(GL_TRIANGLE_FAN);
    glColor4ub(this->_RGB.r(),this->_RGB.g(),this->_RGB.b(),this->_transparant);
    glVertex3f(x(0)+dir(0)*h,x(1)+dir(1)*h,x(2)+dir(2)*h);
    Mat2x33F64 m= GeometricalTransformation::rotationFromVectorToVector(Vec3F64(0,0,1),dir);
    for(double angle=0.0f;angle<=(2*PI);angle+=(PI/10.0))
    {
        Vec3F64 xrot(r*std::sin(angle),r*std::cos(angle),0);
        xrot= m*xrot;
        glNormal3f(xrot(0),xrot(1),xrot(2));
        xrot = x+xrot;
        glVertex3f(xrot(0),xrot(1),xrot(2));
    }
    glEnd();
#endif


}



FigureCone::FigureCone(){
    //   key =  FigureUnitSquare::KEY;
}
FigureCone * FigureCone::clone()const
{
    FigureCone * cone = new FigureCone();
    cone->x =this->x;
    cone->dir=this->dir;
    cone->h=this->h;
    cone->r=this->r;
    cone->_RGB =this->_RGB;
    cone->_transparant =this->_transparant;
    return cone;
}


VecN<3,F32> FigureCone::getMax()const
{
    return this->x;
}
VecN<3,F32> FigureCone::getMin()const
{
    return this->x;
}
void FigureCone::callBeginMode()
{
}
void FigureCone::callEndMode()
{
}



void FigureArrow::draw()
{

    cone.draw();
    line.draw();

}

void FigureArrow::setArrow(const Vec3F64 x1,const Vec3F64 x2,double heigh_peak){
    line.x1=x1;
    line.x2=x2;
    cone.x = x2;
    cone.dir = (x2-x1)/(x2-x1).norm();
    cone.h = heigh_peak;
    cone.r = heigh_peak/2;
}
void FigureArrow::setTransparent(UI8  transparent){
    line.setTransparent(transparent);
    cone.setTransparent(transparent);
}

void FigureArrow::setRGB(const RGBUI8& RGB){
    line.setRGB(RGB);
    cone.setRGB(RGB);
}


FigureArrow::FigureArrow(){
    //      key =  FigureArrow::KEY;
}
FigureArrow * FigureArrow::clone()const
{
    FigureArrow * cone = new FigureArrow();
    cone->cone =this->cone;
    cone->line =this->line;
    cone->_RGB =this->_RGB;
    cone->_transparant =this->_transparant;
    return cone;
}


VecN<3,F32> FigureArrow::getMax()const
{
    return line.getMax();
}
VecN<3,F32> FigureArrow::getMin()const
{
    return line.getMin();
}
void FigureArrow::callBeginMode()
{
}
void FigureArrow::callEndMode()
{
}
Vec<GeometricalFigure*> FigureArrow::referencielEuclidean(pop::RGBUI8 color){
    Vec<GeometricalFigure*> v;
    FigureArrow * arrow =  new FigureArrow;
    arrow->setRGB(color);arrow->setArrow(Vec3F64(0,0,0),Vec3F64(1,0,0),0.1);v.push_back(arrow);
    arrow =  new FigureArrow;
    arrow->setRGB(color);arrow->setArrow(Vec3F64(0,0,0),Vec3F64(0,1,0),0.1);v.push_back(arrow);
    arrow =  new FigureArrow;
    arrow->setRGB(color);arrow->setArrow(Vec3F64(0,0,0),Vec3F64(0,0,1),0.1);v.push_back(arrow);
    return v;
}




Scene3d * op=NULL;
double scale_global=0.75;
double xRot=25;
double yRot=0;
double zRot=25;
#ifdef HAVE_OPENGL
tthread::mutex  mutex_draw;
tthread::thread *thread_draw;
#endif
#ifdef HAVE_GLUT


/* ASCII code for the escape key. */
#define ESCAPE 27

/* The number of our GLUT window */
int window;

/* rotation angle for the triangle. */
float rtri = 0.0f;

/* rotation angle for the quadrilateral. */
float rquad = 0.0f;

/* A general OpenGL initialization function.  Sets all of the initial parameters. */
void InitGL(int , int )            // We call this right after our OpenGL window is created.
{

    if(op!=NULL){
        GLfloat lightAmbient[3];
        lightAmbient[0]=op->getAmbient().r();
        lightAmbient[1]=op->getAmbient().g();
        lightAmbient[2]=op->getAmbient().b();
        GLfloat light_diffuse[3];
        light_diffuse[0]=op->getDiffuse().r();
        light_diffuse[1]=op->getDiffuse().g();
        light_diffuse[2]=op->getDiffuse().b();
        glDepthFunc(GL_LEQUAL);
        glLightfv(GL_LIGHT0,GL_DIFFUSE,light_diffuse);
        glLightfv(GL_LIGHT0,GL_AMBIENT,lightAmbient);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);

        if(op->getTransparentMode()==false){
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
        }
        else{
            glBlendFunc(GL_SRC_ALPHA,GL_ONE);
            glEnable(GL_BLEND);     // Turn Blending On
            glDisable(GL_DEPTH_TEST);
        }


        glClearColor(0.0F,0.0F,0.0F,0.0F);
        //        glClearRGB(0.0f, 0.0f, 0.0f, 0.0f);        // This Will Clear The Background Color To Black
        //        glClearDepth(1.0);                // Enables Clearing Of The Depth Buffer
        //        glDepthFunc(GL_LESS);                    // The Type Of Depth Test To Do
        //        glEnable(GL_DEPTH_TEST);                // Enables Depth Testing
        //        glShadeModel(GL_SMOOTH);            // Enables Smooth Color Shading

        //        glMatrixMode(GL_PROJECTION);
        //        glLoadIdentity();                // Reset The Projection Matrix

        //    gluPerspective(45.0f,(GLfloat)Width/(GLfloat)Height,0.1f,100.0f);    // Calculate The Aspect Ratio Of The Window

        glMatrixMode(GL_MODELVIEW);
    }
}

/* The function called when our window is resized (which shouldn't happen, because we're fullscreen) */
void ReSizeGLScene(int width, int height)
{

    int side = minimum(width, height);
    glViewport((width - side) / 2, (height - side) / 2, side, side);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1.0, +1.0, -1.0, 1.0, 5.0, 60.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslated(0.0, 0.0, -40.0);
}


/* The main drawing function. */
void DrawGLScene()
{

    mutex_draw.lock();
    if(op!=NULL){

        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);    // Clear The Screen And The Depth Buffer
        //glLoadIdentity();                // Reset The View

        glEnable(GL_NORMALIZE);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);


        glPushMatrix();
        VecN<3,F32> _max = -100000;

        VecN<3,F32> _min = 100000;
        for(int i =0;i<(int)op->_v_figure.size();i++)
        {
            _max = maximum(_max, op->_v_figure[i]->getMax());
            _min = minimum(_min, op->_v_figure[i]->getMin());
        }

        double scale=1;
        if(_max(0)-_min(0)!=0)
            scale = minimum(scale,1./(_max(0)-_min(0)));
        if(_max(1)-_min(1)!=0)
            scale = minimum(scale,1./(_max(1)-_min(1)));
        if(_max(2)-_min(2)!=0)
            scale = minimum(scale,1./(_max(2)-_min(2)));

        scale *=10;
        glScalef(scale*scale_global, scale*scale_global, scale*scale_global);
        glRotated(-90, 1.0, 0.0, 0.0);
        glRotated(xRot, 1.0, 0.0, 0.0);
        glRotated(yRot , 0.0, 1.0, 0.0);
        glRotated(zRot, 0.0, 0.0, 1.0);
        VecN<3,F32> trans;
        trans = (_max-_min)/2+_min;
        glTranslatef(-trans(0),-trans(1),-trans(2));
        for(int i =0;i<(int)op->_v_figure.size();i++)
        {
            if(i==0)
            {
                op->_v_figure[i]->callBeginMode();
            }
            else if(typeid(*(op->_v_figure[i]))!=typeid(*(op->_v_figure[i-1])))
            {
                op->_v_figure[i]->callBeginMode();
            }
            op->_v_figure[i]->draw();
            if(i==(int)op->_v_figure.size()-1)
            {
                op->_v_figure[i]->callEndMode();
            }
            else if(typeid(*(op->_v_figure[i]))!=typeid(*(op->_v_figure[i+1])))
            {
                op->_v_figure[i]->callEndMode();
            }
        }
        glPopMatrix();

        glutSwapBuffers();
        op->_snapshot();
    }
    mutex_draw.unlock();
    //    }
}
int vv=0;
/* The function called whenever a key is pressed. */
void keyPressed(unsigned char key, int, int )
{
    /* avoid thrashing this call */
    /* If escape is pressed, kill everything. */
    if (key == ESCAPE)
    {
        glutDestroyWindow(window);

    }else if(key ==(unsigned char) '1'){
        xRot+=5;
    }
    else if(key ==(unsigned char) '2'){
        xRot-=5;
    }
    else if(key ==(unsigned char) '4'){
        yRot+=5;
    }
    else if(key ==(unsigned char) '5'){
        yRot-=5;
    }
    else if(key ==(unsigned char) '7'){
        zRot+=5;
    }
    else if(key ==(unsigned char) '8'){
        zRot-=5;
    }
    else if(key ==(unsigned char) '+'){
        scale_global*=1.1;
    }
    else if(key ==(unsigned char) '-'){
        scale_global*=0.9;
    }
    else if(key ==(unsigned char) 's'){
        std::cout<<"snapshot"<<std::endl;
        op->_shot=true;
    }

}





//static void * toto(void * p_data)

void drawglut(void *)
{
    /* Initialize GLUT state - glut will take any command line arguments that pertain to it or
   X Windows - look at its documentation at http://reality.sgi.com/mjk/spec3/spec3.html */
    int   argc =1;

    char *argv[1];
    *argv = new char[26];

    glutInit(&argc,argv);


    /* Select type of Display mode:
   Double buffer
   RGBA RGB
   Alpha components supported
   Depth buffered for automatic clipping */
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);

    /* get a 640 x 480 window */
    glutInitWindowSize(640, 480);

    /* the window starts at the upper left corner of the screen */
    glutInitWindowPosition(0, 0);

    /* Open a window */

    window = glutCreateWindow(op->title.c_str());

    /* Register the function to do all our OpenGL drawing. */
    glutDisplayFunc(&DrawGLScene);

    /* Go fullscreen.  This is as soon as possible. */
    //  glutFullScreen();

    /* Even if there are no events, redraw our gl scene. */
    glutIdleFunc(&DrawGLScene);
    //    glutIdleFunc(MyIdleFunc);
    /* Register the function called when our window is resized. */
    glutReshapeFunc(&ReSizeGLScene);

    /* Register the function called when the keyboard is pressed. */
    glutKeyboardFunc(&keyPressed);

    /* Initialize our window. */
    InitGL(640, 480);

    /* Start Event Processing Engine */
    try
    {
        glutMainLoop();
    }
    catch (const char* msg)
    {
        std::cout<<msg<<std::endl;
    }

}
#endif

#ifdef WINDOWSOPENGL

//Fonctions d'allocation et suppression
char * StrAllocThrowA(size_t cchSize)
{
    return new char[cchSize];
}
wchar_t * StrAllocThrowW(size_t cchSize)
{
    return new wchar_t[cchSize];
}
void StrFreeA(char * s)
{
    delete[] s;
}
void StrFreeW(wchar_t * s)
{
    delete[] s;
}

//Surcharge C++, plus simple.
inline void StrFree( char   * s) { return StrFreeA(s); }
inline void StrFree(wchar_t * s) { return StrFreeW(s); }

//Fonctions de conversion
wchar_t * ctow(char const *sczA)
{
    size_t const cchLenA = strlen(sczA);
    size_t const cchLenW = mbstowcs(NULL, sczA, cchLenA+1);
    wchar_t * szW = StrAllocThrowW(cchLenW+1);
    mbstowcs(szW, sczA, cchLenA+1);
    return szW;
}

HDC            hDC=NULL;        // Private GDI Device Context
HGLRC        hRC=NULL;        // Permanent Rendering Context
HWND        hWnd=NULL;        // Holds Our Window Handle
HINSTANCE    hInstance;        // Holds The Instance Of The Application

bool    keys[256];            // Array Used For The Keyboard Routine
bool    active=TRUE;        // Window Active Flag Set To TRUE By Default
bool    fullscreen=FALSE;    // Fullscreen Flag Set To Fullscreen Mode By Default

GLfloat    rtri;                // Angle For The Triangle ( NEW )
GLfloat    rquad;                // Angle For The Quad ( NEW )

LRESULT    CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);    // Declaration For WndProc

void ReSizeGLScene(GLsizei width, GLsizei height)        // Resize And Initialize The GL Window
{
    int side;
    if(width>height)
        side = width;
    else
        side = height;
    glViewport((width - side) / 2, (height - side) / 2, side, side);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1.0, +1.0, -1.0, 1.0, 5.0, 60.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslated(0.0, 0.0, -40.0);                                // Reset The Modelview Matrix
}

int InitGL()                                        // All Setup For OpenGL Goes Here
{
    if(op!=NULL){
        GLfloat lightAmbient[3];
        lightAmbient[0]=op->getAmbient().r();
        lightAmbient[1]=op->getAmbient().g();
        lightAmbient[2]=op->getAmbient().b();
        GLfloat light_diffuse[3];
        light_diffuse[0]=op->getDiffuse().r();
        light_diffuse[1]=op->getDiffuse().g();
        light_diffuse[2]=op->getDiffuse().b();
        glDepthFunc(GL_LEQUAL);
        glLightfv(GL_LIGHT0,GL_DIFFUSE,light_diffuse);
        glLightfv(GL_LIGHT0,GL_AMBIENT,lightAmbient);
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);

        if(op->getTransparentMode()==false){
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
        }
        else{
            glBlendFunc(GL_SRC_ALPHA,GL_ONE);
            glEnable(GL_BLEND);     // Turn Blending On
            glDisable(GL_DEPTH_TEST);
        }
        glClearColor(0.0F,0.0F,0.0F,0.0F);
        //        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);        // This Will Clear The Background Color To Black
        //        glClearDepth(1.0);                // Enables Clearing Of The Depth Buffer
        //        glDepthFunc(GL_LESS);                    // The Type Of Depth Test To Do
        //        glEnable(GL_DEPTH_TEST);                // Enables Depth Testing
        //        glShadeModel(GL_SMOOTH);            // Enables Smooth Color Shading

        //        glMatrixMode(GL_PROJECTION);
        //        glLoadIdentity();                // Reset The Projection Matrix

        //    gluPerspective(45.0f,(GLfloat)Width/(GLfloat)Height,0.1f,100.0f);    // Calculate The Aspect Ratio Of The Window

        glMatrixMode(GL_MODELVIEW);
    }
    return TRUE;                                        // Initialization Went OK
}

int DrawGLScene()                                    // Here's Where We Do All The Drawing
{
    mutex_draw.lock();
    if(op!=NULL){
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);    // Clear The Screen And The Depth Buffer
        //    glLoadIdentity();                // Reset The View

        glEnable(GL_NORMALIZE);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);


        glPushMatrix();
        VecN<3,F32> _max = -10000;

        VecN<3,F32> _min = 10000;
        for(int i =0;i<(int)op->_v_figure.size();i++)
        {
            VecN<3,F32> v =op->_v_figure[i]->getMax();
            for(int j =0;j<3;j++){
                if(v(j)>_max(j))
                    _max(j)=v(j);
            }
            v =op->_v_figure[i]->getMin();
            for(int j =0;j<3;j++)
                if(v(j)<_min(j))
                    _min(j)=v(j);

        }
        double scale=1;
        if(_max(0)-_min(0)!=0)
            scale = std::min<double>(scale,1./(_max(0)-_min(0)));
        if(_max(1)-_min(1)!=0)
            scale = std::min<double>(scale,1./(_max(1)-_min(1)));
        if(_max(2)-_min(2)!=0)
            scale = std::min<double>(scale,1./(_max(2)-_min(2)));

        scale *=10;
        glScalef(scale*scale_global, scale*scale_global, scale*scale_global);

        glRotated(-90, 1.0, 0.0, 0.0);

        glRotated(xRot, 1.0, 0.0, 0.0);
        glRotated(yRot , 0.0, 1.0, 0.0);
        glRotated(zRot, 0.0, 0.0, 1.0);
        VecN<3,F32> trans;
        trans = (_max-_min)/2+_min;
        glTranslatef(-trans(0),-trans(1),-trans(2));
        for(int i =0;i<(int)op->_v_figure.size();i++)
        {
            if(i==0)
            {
                op->_v_figure[i]->callBeginMode();
            }
            else if(typeid(*(op->_v_figure[i]))!=typeid(*(op->_v_figure[i-1])))
            {
                op->_v_figure[i]->callBeginMode();
            }
            op->_v_figure[i]->draw();
            if(i==(int)op->_v_figure.size()-1)
            {
                op->_v_figure[i]->callEndMode();
            }
            else if(typeid(*(op->_v_figure[i]))!=typeid(*(op->_v_figure[i+1])))
            {
                op->_v_figure[i]->callEndMode();
            }
        }
        glPopMatrix();
        op->_snapshot();
        mutex_draw.unlock();
    }
    return TRUE;                                        // Keep Going
}

void KillGLWindow()                                // Properly Kill The Window
{
    ctow("toto");
    if (fullscreen)                                        // Are We In Fullscreen Mode?
    {
        ChangeDisplaySettings(NULL,0);                    // If So Switch Back To The Desktop
        ShowCursor(TRUE);                                // Show Mouse VecNer
    }

    if (hRC)                                            // Do We Have A Rendering Context?
    {
        if (!wglMakeCurrent(NULL,NULL))                    // Are We Able To Release The DC And RC Contexts?
        {
            MessageBox(NULL,ctow("Release Of DC And RC Failed."),ctow("SHUTDOWN ERROR"),MB_OK | MB_ICONINFORMATION);
        }

        if (!wglDeleteContext(hRC))                        // Are We Able To Delete The RC?
        {
            MessageBox(NULL,ctow("Release Rendering Context Failed."),ctow("SHUTDOWN ERROR"),MB_OK | MB_ICONINFORMATION);
        }
        hRC=NULL;                                        // Set RC To NULL
    }

    if (hDC && !ReleaseDC(hWnd,hDC))                    // Are We Able To Release The DC
    {
        MessageBox(NULL,ctow("Release Device Context Failed."),ctow("SHUTDOWN ERROR"),MB_OK | MB_ICONINFORMATION);
        hDC=NULL;                                        // Set DC To NULL
    }

    if (hWnd && !DestroyWindow(hWnd))                    // Are We Able To Destroy The Window?
    {
        MessageBox(NULL,ctow("Could Not Release hWnd."),ctow("SHUTDOWN ERROR"),MB_OK | MB_ICONINFORMATION);
        hWnd=NULL;                                        // Set hWnd To NULL
    }

    if (!UnregisterClass(ctow("OpenGL"),hInstance))            // Are We Able To Unregister Class
    {
        MessageBox(NULL,ctow("Could Not Unregister Class."),ctow("SHUTDOWN ERROR"),MB_OK | MB_ICONINFORMATION);
        hInstance=NULL;                                    // Set hInstance To NULL
    }
}

/*    This Code Creates Our OpenGL Window.  Parameters Are:                    *
 *    title            - Title To Appear At The Top Of The Window                *
 *    width            - Width Of The GL Window Or Fullscreen Mode                *
 *    height            - Height Of The GL Window Or Fullscreen Mode            *
 *    bits            - Number Of Bits To Use For COLOR (8/16/24/32)            *
 *    fullscreenflag    - Use Fullscreen Mode (TRUE) Or Windowed Mode (FALSE)    */

BOOL CreateGLWindow(char* title, int width, int height, int bits, bool fullscreenflag)
{
    GLuint        PixelFormat;            // Holds The Results After Searching For A Match
    WNDCLASS    wc;                        // Windows Class Structure
    DWORD        dwExStyle;                // Window Extended Style
    DWORD        dwStyle;                // Window Style
    RECT        WindowRect;                // Grabs Rectangle Upper Left / Lower Right Values
    WindowRect.left=(long)0;            // Set Left Value To 0
    WindowRect.right=(long)width;        // Set Right Value To Requested Width
    WindowRect.top=(long)0;                // Set Top Value To 0
    WindowRect.bottom=(long)height;        // Set Bottom Value To Requested Height

    fullscreen=fullscreenflag;            // Set The Global Fullscreen Flag

    hInstance            = GetModuleHandle(NULL);                // Grab An Instance For Our Window
    wc.style            = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;    // Redraw On Size, And Own DC For Window.
    wc.lpfnWndProc        = (WNDPROC) WndProc;                    // WndProc Handles Messages
    wc.cbClsExtra        = 0;                                    // No Extra Window Data
    wc.cbWndExtra        = 0;                                    // No Extra Window Data
    wc.hInstance        = hInstance;                            // Set The Instance
    wc.hIcon            = LoadIcon(NULL, IDI_WINLOGO);            // Load The Default Icon
    wc.hCursor            = LoadCursor(NULL, IDC_ARROW);            // Load The Arrow VecNer
    wc.hbrBackground    = NULL;                                    // No Background Required For GL
    wc.lpszMenuName        = NULL;                                    // We Don't Want A Menu
    wc.lpszClassName    = ctow("OpenGL");                                // Set The Class Name

    if (!RegisterClass(&wc))                                    // Attempt To Register The Window Class
    {
        MessageBox(NULL,ctow("Failed To Register The Window Class."),ctow("ERROR"),MB_OK|MB_ICONEXCLAMATION);
        return FALSE;                                            // Return FALSE
    }

    if (fullscreen)                                                // Attempt Fullscreen Mode?
    {
        DEVMODE dmScreenSettings;                                // Device Mode
        memset(&dmScreenSettings,0,sizeof(dmScreenSettings));    // Makes Sure Memory's Cleared
        dmScreenSettings.dmSize=sizeof(dmScreenSettings);        // Size Of The Devmode Structure
        dmScreenSettings.dmPelsWidth    = width;                // Selected Screen Width
        dmScreenSettings.dmPelsHeight    = height;                // Selected Screen Height
        dmScreenSettings.dmBitsPerPel    = bits;                    // Selected Bits Per Pixel
        dmScreenSettings.dmFields=DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT;

        // Try To Set Selected Mode And Get Results.  NOTE: CDS_FULLSCREEN Gets Rid Of Start Bar.
        if (ChangeDisplaySettings(&dmScreenSettings,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL)
        {
            // If The Mode Fails, Offer Two Options.  Quit Or Use Windowed Mode.
            if (MessageBox(NULL,ctow("The Requested Fullscreen Mode Is Not Supported By\nYour Video Card. Use Windowed Mode Instead?"),ctow("NeHe GL"),MB_YESNO|MB_ICONEXCLAMATION)==IDYES)
            {
                fullscreen=FALSE;        // Windowed Mode Selected.  Fullscreen = FALSE
            }
            else
            {
                // Pop Up A Message Box Letting User Know The Program Is Closing.
                MessageBox(NULL,ctow("Program Will Now Close."),ctow("ERROR"),MB_OK|MB_ICONSTOP);
                return FALSE;                                    // Return FALSE
            }
        }
    }

    if (fullscreen)                                                // Are We Still In Fullscreen Mode?
    {
        dwExStyle=WS_EX_APPWINDOW;                                // Window Extended Style
        dwStyle=WS_POPUP;                                        // Windows Style
        ShowCursor(FALSE);                                        // Hide Mouse VecNer
    }
    else
    {
        dwExStyle=WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;            // Window Extended Style
        dwStyle=WS_OVERLAPPEDWINDOW;                            // Windows Style
    }

    AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);        // Adjust Window To True Requested Size

    // Create The Window
    if (!(hWnd=CreateWindowEx(    dwExStyle,                            // Extended Style For The Window
                                  ctow("OpenGL"),                            // Class Name
                                  ctow(title),                                // Window Title
                                  dwStyle |                            // Defined Window Style
                                  WS_CLIPSIBLINGS |                    // Required Window Style
                                  WS_CLIPCHILDREN,                    // Required Window Style
                                  0, 0,                                // Window Position
                                  WindowRect.right-WindowRect.left,    // Calculate Window Width
                                  WindowRect.bottom-WindowRect.top,    // Calculate Window Height
                                  NULL,                                // No Parent Window
                                  NULL,                                // No Menu
                                  hInstance,                            // Instance
                                  NULL)))                                // Dont Pass Anything To WM_CREATE
    {
        KillGLWindow();                                // Reset The Display
        MessageBox(NULL,ctow("Window Creation Error."),ctow("ERROR"),MB_OK|MB_ICONEXCLAMATION);
        return FALSE;                                // Return FALSE
    }

    static    PIXELFORMATDESCRIPTOR pfd=                // pfd Tells Windows How We Want Things To Be
    {
            sizeof(PIXELFORMATDESCRIPTOR),                // Size Of This Pixel Format Descriptor
            1,                                            // Version Number
            PFD_DRAW_TO_WINDOW |                        // Format Must Support Window
            PFD_SUPPORT_OPENGL |                        // Format Must Support OpenGL
            PFD_DOUBLEBUFFER,                            // Must Support Double Buffering
            PFD_TYPE_RGBA,                                // Request An RGBA Format
            (BYTE)bits,                                        // Select Our RGB Depth
            0, 0, 0, 0, 0, 0,                            // RGB Bits Ignored
            0,                                            // No Alpha Buffer
            0,                                            // Shift Bit Ignored
            0,                                            // No Accumulation Buffer
            0, 0, 0, 0,                                    // Accumulation Bits Ignored
            16,                                            // 16Bit Z-Buffer (Depth Buffer)
            0,                                            // No Stencil Buffer
            0,                                            // No Auxiliary Buffer
            PFD_MAIN_PLANE,                                // Main Drawing Layer
            0,                                            // Reserved
            0, 0, 0                                        // Layer Masks Ignored
};

if (!(hDC=GetDC(hWnd)))                            // Did We Get A Device Context?
{
    KillGLWindow();                                // Reset The Display
    MessageBox(NULL,ctow("Can't Create A GL Device Context."),ctow("ERROR"),MB_OK|MB_ICONEXCLAMATION);
    return FALSE;                                // Return FALSE
}

if (!(PixelFormat=ChoosePixelFormat(hDC,&pfd)))    // Did Windows Find A Matching Pixel Format?
{
    KillGLWindow();                                // Reset The Display
    MessageBox(NULL,ctow("Can't Find A Suitable PixelFormat."),ctow("ERROR"),MB_OK|MB_ICONEXCLAMATION);
    return FALSE;                                // Return FALSE
}

if(!SetPixelFormat(hDC,PixelFormat,&pfd))        // Are We Able To Set The Pixel Format?
{
    KillGLWindow();                                // Reset The Display
    MessageBox(NULL,ctow("Can't Set The PixelFormat."),ctow("ERROR"),MB_OK|MB_ICONEXCLAMATION);
    return FALSE;                                // Return FALSE
}

if (!(hRC=wglCreateContext(hDC)))                // Are We Able To Get A Rendering Context?
{
    KillGLWindow();                                // Reset The Display
    MessageBox(NULL,ctow("Can't Create A GL Rendering Context."),ctow("ERROR"),MB_OK|MB_ICONEXCLAMATION);
    return FALSE;                                // Return FALSE
}

if(!wglMakeCurrent(hDC,hRC))                    // Try To Activate The Rendering Context
{
    KillGLWindow();                                // Reset The Display
    MessageBox(NULL,ctow("Can't Activate The GL Rendering Context."),ctow("ERROR"),MB_OK|MB_ICONEXCLAMATION);
    return FALSE;                                // Return FALSE
}

ShowWindow(hWnd,SW_SHOW);                        // Show The Window
SetForegroundWindow(hWnd);                        // Slightly Higher Priority
SetFocus(hWnd);                                    // Sets Keyboard Focus To The Window
ReSizeGLScene(width, height);                    // Set Up Our Perspective GL Screen

if (!InitGL())                                    // Initialize Our Newly Created GL Window
{
    KillGLWindow();                                // Reset The Display
    MessageBox(NULL,ctow("Initialization Failed."),ctow("ERROR"),MB_OK|MB_ICONEXCLAMATION);
    return FALSE;                                // Return FALSE
}

return TRUE;                                    // Success
}

LRESULT CALLBACK WndProc(    HWND    hWnd,            // Handle For This Window
                             UINT    uMsg,            // Message For This Window
                             WPARAM    wParam,            // Additional Message Information
                             LPARAM    lParam)            // Additional Message Information
{
    switch (uMsg)                                    // Check For Windows Messages
    {
    case WM_ACTIVATE:                            // Watch For Window Activate Message
    {
        // LoWord Can Be WA_INACTIVE, WA_ACTIVE, WA_CLICKACTIVE,
        // The High-Order Word Specifies The Minimized State Of The Window Being Activated Or Deactivated.
        // A NonZero Value Indicates The Window Is Minimized.
        if ((LOWORD(wParam) != WA_INACTIVE) && !((BOOL)HIWORD(wParam)))
            active=TRUE;                        // Program Is Active
        else
            active=FALSE;                        // Program Is No Longer Active

        return 0;                                // Return To The Message Loop
    }

    case WM_SYSCOMMAND:                            // Intercept System Commands
    {
        switch (wParam)                            // Check System Calls
        {
        case SC_SCREENSAVE:                    // Screensaver Trying To Start?
        case SC_MONITORPOWER:                // Monitor Trying To Enter Powersave?
            return 0;                            // Prevent From Happening
        }
        break;                                    // Exit
    }

    case WM_CLOSE:                                // Did We Receive A Close Message?
    {
        PostQuitMessage(0);                        // Send A Quit Message
        return 0;                                // Jump Back
    }

    case WM_KEYDOWN:                            // Is A Key Being Held Down?
    {
        keys[wParam] = TRUE;                    // If So, Mark It As TRUE
        return 0;                                // Jump Back
    }

    case WM_KEYUP:                                // Has A Key Been Released?
    {
        keys[wParam] = FALSE;                    // If So, Mark It As FALSE
        return 0;                                // Jump Back
    }

    case WM_SIZE:                                // Resize The OpenGL Window
    {
        ReSizeGLScene(LOWORD(lParam),HIWORD(lParam));  // LoWord=Width, HiWord=Height
        return 0;                                // Jump Back
    }
    }

    // Pass All Unhandled Messages To DefWindowProc
    return DefWindowProc(hWnd,uMsg,wParam,lParam);
}
void drawwindow(void *    )            // Window Show State
{

    MSG        msg;                                    // Windows Message Structure
    BOOL    done=FALSE;                                // Bool Variable To Exit Loop

    //    // Ask The User Which Screen Mode They Prefer
    //    if (MessageBox(NULL,(WCHAR*)"Would You Like To Run In Fullscreen Mode?", (WCHAR*)"Start FullScreen?",MB_YESNO|MB_ICONQUESTION)==IDNO)
    //    {
    //        fullscreen=FALSE;                            // Windowed Mode
    //    }

    // Create Our OpenGL Window
    char  * c = const_cast<char*>(op->title.c_str());
    if (!CreateGLWindow(c,640,480,16,fullscreen))
    {
        return;                                    // Quit If Window Was Not Created
    }

    while(!done)                                    // Loop That Runs While done=FALSE
    {
        if (PeekMessage(&msg,NULL,0,0,PM_REMOVE))    // Is There A Message Waiting?
        {
            if (msg.message==WM_QUIT)                // Have We Received A Quit Message?
            {
                done=TRUE;                            // If So done=TRUE
            }
            else                                    // If Not, Deal With Window Messages
            {
                TranslateMessage(&msg);                // Translate The Message
                DispatchMessage(&msg);                // Dispatch The Message
            }
        }
        else                                        // If There Are No Messages
        {
            if(keys[VK_NUMPAD1]){
                xRot+=5;
            }
            else if(keys[VK_NUMPAD2]){
                xRot-=5;
            }
            else if(keys[VK_NUMPAD4]){
                yRot+=5;
            }
            else if(keys[VK_NUMPAD5]){
                yRot-=5;
            }
            else if(keys[VK_NUMPAD7]){
                zRot+=5;
            }
            else if(keys[VK_NUMPAD8]){
                zRot-=5;
            }
            else if(keys[VK_ADD]){
                scale_global*=1.1;
            }
            else if(keys[VK_SUBTRACT]){
                scale_global*=0.9;
            }
            else if(keys['S']){
                op->_shot=true;
            }

            // Draw The Scene.  Watch For ESC Key And Quit Messages From DrawGLScene()
            if ((active && !DrawGLScene()) || keys[VK_ESCAPE])    // Active?  Was There A Quit Received?
            {
                done=TRUE;                            // ESC or DrawGLScene Signalled A Quit
            }
            else                                    // Not Time To Quit, Update Screen
            {
                SwapBuffers(hDC);                    // Swap Buffers (Double Buffering)
            }

            if (keys[VK_F1])                        // Is F1 Being Pressed?
            {
                keys[VK_F1]=FALSE;                    // If So Make Key FALSE
                KillGLWindow();                        // Kill Our Current Window
                fullscreen=!fullscreen;                // Toggle Fullscreen / Windowed Mode
                // Recreate Our OpenGL Window
                if (!CreateGLWindow(const_cast<char*>(op->title.c_str()),640,480,16,fullscreen))
                {
                    return ;                        // Quit If Window Was Not Created
                }
            }
        }
    }

    // Shutdown
    KillGLWindow();                                    // Kill The Window
    return;                            // Exit The Program
}
#endif




void Scene3d::display(bool stopprocess)
{

    op=this;
#ifdef HAVE_GLUT
    if(_instance==false){
        _instance=true;
        if(stopprocess==false)
            thread_draw = new  tthread::thread(drawglut,NULL);
        else
            drawglut(NULL);

        int milliseconds =600;

        struct timespec tv;
        tv.tv_sec = milliseconds/1000;
        tv.tv_nsec = (milliseconds%1000)*1000000;
        nanosleep(&tv,0);
    }
#endif
#ifdef WINDOWSOPENGL
    if(stopprocess==false)
        thread_draw = new  tthread::thread(drawwindow,NULL);
    else
        drawwindow(NULL);
#endif
#ifdef HOMEMADETRACING
    stopprocess =true;
    drawMyImplementation();
#endif
    if(stopprocess ==true)
        stopprocess=false;
}
bool Scene3d::_instance=false;
void Scene3d::_snapshot(){
#if defined(HAVE_OPENGL)
    if(_shot==true){

        int nSize = 640* 480 ;
        GLubyte *pixelred = new GLubyte [nSize];
        glReadPixels(0, 0, 640, 480, GL_RED,
                     GL_UNSIGNED_BYTE, pixelred);
        GLubyte *pixelgreen = new GLubyte [nSize];
        glReadPixels(0, 0, 640, 480, GL_GREEN,
                     GL_UNSIGNED_BYTE, pixelgreen);
        GLubyte *pixelblue = new GLubyte [nSize];
        glReadPixels(0, 0, 640, 480, GL_BLUE,
                     GL_UNSIGNED_BYTE, pixelblue);
        Mat2RGBUI8 img(480,640);

        for(unsigned int i =0;i<img.sizeI();i++)
            for(unsigned int j=0;j<img.sizeJ();j++)
            {
                img(img.sizeI()-1-i,j).r() = pixelred[(j+i*img.sizeJ())];
                img(img.sizeI()-1-i,j).g() = pixelgreen[(j+i*img.sizeJ())];
                img(img.sizeI()-1-i,j).b() = pixelblue[(j+i*img.sizeJ())];
            }
        img.save(_file.c_str());
        delete pixelred;
        delete pixelgreen;
        delete pixelblue;
        _shot=false;
    }
#endif

}

void Scene3d::snapshot(const char * file){

    _shot=true;
    _file=file;
}

void Scene3d::lock(){
#if defined(HAVE_OPENGL)

    mutex_draw.lock();
#endif
}

void Scene3d::unlock(){
#if defined(HAVE_OPENGL)
    mutex_draw.unlock();

    int milliseconds =5;
#if Pop_OS==1
    struct timespec tv;
    tv.tv_sec = milliseconds/1000;
    tv.tv_nsec = (milliseconds%1000)*1000000;
    nanosleep(&tv,0);
#elif Pop_OS==2
    Sleep(milliseconds);
#endif
#endif
}



void Scene3d::clear(){
    for(int i =0;i<(int)_v_figure.size();i++){
        delete _v_figure[i];
    }
    _v_figure.clear();
}

Scene3d::~Scene3d()
{
    lock();
    op=NULL;
    for(int i =0;i<(int)_v_figure.size();i++){
        delete _v_figure[i];
    }
    _v_figure.clear();
    unlock();
}


void Scene3d::addGeometricalFigure(const Vec<GeometricalFigure*> figures){
    lock();
    for(unsigned int i=0;i<figures.size();i++)
        _v_figure.push_back(figures(i));
    unlock();
}


Scene3d::Scene3d()
    :_shot(false),_file("snapshot.png"),_ambient(0.5,0.5,0.5),_diffuse(0.5,0.5,0.5),_transparencymode(false)
{
    title = "keys= +(-) zoom, 1(2)xrot, 4(5)yrot, 7(8)zrot, s(snapshot in working directory)";
}

Scene3d& Scene3d::operator =(const Scene3d& g){
    this->_ambient=g._ambient;
    this->_diffuse=g._diffuse;
    this->_transparencymode=g._transparencymode;

    for(int i=0; i <(int)_v_figure.size();i++)
    {
        delete this->_v_figure[i];
    }
    this->_v_figure.clear();
    for(int i =0;i<(int)g._v_figure.size();i++)
    {
        this->_v_figure.push_back(g._v_figure[i]->clone());
    }
    return *this;
}
Scene3d& Scene3d::merge(const Scene3d& g){
    for(int i=0; i <(int)g._v_figure.size();i++)
    {
        this->_v_figure.push_back(g._v_figure[i]->clone());
    }
    return *this;
}

void Scene3d::rotateX(F64 angle)
{
    xRot+=angle;
}

void Scene3d::rotateY(F64 angle)
{
    yRot+=angle;
}
void Scene3d::rotateZ(F64 angle)
{
    zRot+=angle;
}
void Scene3d::setColorAllGeometricalFigure(const RGBUI8 & value){
    for(unsigned int i=0;i<this->_v_figure.size();i++){
        GeometricalFigure* figure = this->_v_figure[i];
        figure->setRGB(value);
    }
}

void Scene3d::setTransparencyAllGeometricalFigure( UI8  value){
    for(unsigned int i=0;i<this->_v_figure.size();i++)
        this->_v_figure[i]->setTransparent(value);
}
void Scene3d::setAmbient(const pop::RGBF64 &ambient){
    _ambient = ambient;
}

pop::RGBF64 Scene3d::getAmbient()const{
    return _ambient;
}
void Scene3d::setDiffuse(const pop::RGBF64 &diffuse){
    _diffuse = diffuse;
}

pop::RGBF64 Scene3d::getDiffuse()const{
    return _diffuse;
}
void Scene3d::setTransparentMode(bool istranspararent){
    _transparencymode = istranspararent;
}

bool Scene3d::getTransparentMode()const{
    return _transparencymode;
}
}


