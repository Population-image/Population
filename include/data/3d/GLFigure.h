/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software and for any writing
public or private that has resulted from the use of the software population,
the reference of this book "Population library, 2012, Vincent Tariel" shall
be included in it.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other dealings
in the Software.
\***************************************************************************/
#ifndef GEOMETRICALFIGURE_H
#define GEOMETRICALFIGURE_H
#define UNICODE 1



#include <cstring>
#include <queue>
#include <iostream>
#include <fstream>

#include"data/typeF/TypeF.h"
#include"data/typeF/RGB.h"
#include"data/vec/VecN.h"
namespace pop
{
/*! \ingroup Data
* \defgroup Other Other
* \brief usefull classes
*/

/*! \ingroup Other
* \defgroup Scene3d Scene3d
* \brief 3d scene in opengl
*/


class POP_EXPORTS GeometricalFigure
{
    /*!
     * \class pop::GeometricalFigure
     * \brief abtract interface for gl geometrical figure
     * \author Tariel Vincent
     *
     * very old code...
    */
protected:
    UI8  _transparant;
    RGBUI8 _RGB;
public:
    GeometricalFigure();
    virtual ~GeometricalFigure();
    virtual void draw(){}
    virtual GeometricalFigure * clone()const=0;
    virtual VecN<3,F32> getMax()const=0;
    virtual VecN<3,F32> getMin()const=0;
    virtual void callBeginMode();
    virtual void callEndMode();
    virtual void setTransparent(UI8  transparent);
    UI8  getTransparent()const;

    virtual void setRGB(const RGBUI8& RGB);
    RGB<UI8 > getRGB()const;
};
class POP_EXPORTS FigurePolygon :public GeometricalFigure
{
public:

    Vec<VecN<3,F32> > vpos;
    VecN<3,F32> normal;
    FigurePolygon();
    void draw();
    virtual FigurePolygon * clone()const;
    virtual VecN<3,F32> getMax()const;
    virtual VecN<3,F32> getMin()const;
    static FigurePolygon* createTriangle(const VecN<3,F32>& x1,const VecN<3,F32>& x2,const VecN<3,F32>& x3, const RGBUI8 &c=RGBUI8(255,255,255), UI8 transparence=255);
    void translate(VecN<3,F32> trans);

};

class POP_EXPORTS FigureTriangle :public GeometricalFigure
{
public:
    VecN<3,F32>  x;
    VecN<3,F32> normal;
    FigureTriangle();
    void draw();
    virtual GeometricalFigure * clone()const;
    virtual void callBeginMode();
    virtual void callEndMode();
    virtual VecN<3,F32> getMax()const;
    virtual VecN<3,F32> getMin()const;


};

class POP_EXPORTS FigureLine :public GeometricalFigure
{
public:

    F32 width;
    VecN<3,F32>  x1;
    VecN<3,F32>  x2;
    FigureLine();
    void draw();

    virtual GeometricalFigure * clone()const;
    virtual void callBeginMode();
    virtual void callEndMode();
    virtual VecN<3,F32> getMax()const;
    virtual VecN<3,F32> getMin()const;

};



class POP_EXPORTS FigureSphere :public GeometricalFigure
{
public:

    F32 _radius;
    VecN<3,F32>  _x;
    FigureSphere();
    void draw();

    virtual GeometricalFigure * clone()const;
    virtual void callBeginMode();
    virtual void callEndMode();
    virtual VecN<3,F32> getMax()const;
    virtual VecN<3,F32> getMin()const;

};

class POP_EXPORTS FigureUnitSquare :public GeometricalFigure
{
public:

    VecN<3,int>  x;
    int way;
    int direction;
    FigureUnitSquare();
    void draw();

    virtual GeometricalFigure * clone()const;
    virtual void callBeginMode();
    virtual void callEndMode();
    virtual VecN<3,F32> getMax()const;
    virtual VecN<3,F32> getMin()const;

};

class POP_EXPORTS FigureCone :public GeometricalFigure
{
public:

    Vec3F32  x;
    Vec3F32 dir;
    F32 h;
    F32 r;
    FigureCone();
    void draw();

    virtual FigureCone * clone()const;
    virtual void callBeginMode();
    virtual void callEndMode();
    virtual VecN<3,F32> getMax()const;
    virtual VecN<3,F32> getMin()const;

};

class POP_EXPORTS FigureArrow:public GeometricalFigure
{
public:

    FigureCone _cone;
    FigureLine _line;
    FigureArrow();
    void setArrow(const Vec3F32 x1,const Vec3F32 x2,F32 heigh_peak);
     void setTransparent(UI8  transparent);
     void setRGB(const RGBUI8& RGB);
    void draw();
    virtual FigureArrow * clone()const;
    virtual void callBeginMode();
    virtual void callEndMode();
    virtual VecN<3,F32> getMax()const;
    virtual VecN<3,F32> getMin()const;
    static Vec<GeometricalFigure*> referencielEuclidean(pop::RGBUI8 color=pop::RGBUI8(255,0,0));

};
class POP_EXPORTS Scene3d
{
    /*!
      *  \class pop::Scene3d
      *  \brief contain the list of glfigue
      *  \ingroup Scene3d
      *  \author Tariel Vincent
      *
      * This simple class contains a vector of elementary glfigures. This class does not aim to concurrennce VTK or other top-level OpenGL library. The aim is to have
      * a simple opengl visualisation for material science. When you close the display, the program exist in Linux thanks to glut.     You have many information in these tutorials \ref  pagevisu2d \ref  pagevisu3d.\n
      * For instance, this code produces this following image
      * \code
        Mat3UI8 img;
        img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/rock3d.pgm").c_str());
        //img = img(Vec3I32(0,0,0),Vec3I32(64,64,64)); //uncomment these lines for the full core sample
        Mat3UI8 imgfilter= Processing::median(img,2);
        Mat3UI8 grain= Processing::threshold(imgfilter,155);
        Mat3UI8 oil = Processing::threshold(imgfilter,70,110);
        oil = Processing::openingRegionGrowing(oil,2);//To remove the interface artefact
        Mat3UI8 air = Processing::threshold(imgfilter,0,40);
        Mat3UI8 seed = Processing::labelMerge(grain,oil);
        seed = Processing::labelMerge(seed,air);
        Mat3UI8 gradient = Processing::gradientMagnitudeDeriche(img,1.5);
        Mat3UI8 water = Processing::watershed(seed,gradient);
        grain = Processing::labelFromSingleSeed(water,grain);
        grain=Mat3F32(grain)*0.75;
        oil = Processing::labelFromSingleSeed(water,oil);
        oil = Mat3F32(oil)*0.4;
        Scene3d scene;
        Visualization::marchingCube(scene,grain);
        Visualization::marchingCube(scene,oil);
        Visualization::lineCube(scene,img);
        scene.display();
      * \endcode
      * \image html marchingcube.png
      * \sa pop::Visualization
      */
public:
    bool _shot;
private:
    static bool _instance;
     std::string _file;
     pop::RGBF32 _ambient;
     RGBF32 _diffuse;
     /*! \var _transparencymode
      *  if transparency_mode=1 no transparent, otherwise yes
      */
     bool _transparencymode;
public:



    /*! \var title
     *  windows title
     */
     std::string title;
    /*! \var _v_figure
     *  std::vector of elementary figues
     */
    std::vector<GeometricalFigure* > _v_figure;

    /*!
    \fn Scene3d();
    *
    * default constructor
    */
    Scene3d();
    /*!
    \fn ~Scene3d()
    *
    * default destructor
    */
    ~Scene3d();

    /*!
    \fn Scene3d& operator =(const Scene3d& g)
    *
    * assignement operator
    */
    Scene3d& operator =(const Scene3d& g);

    /*!
    \fn Scene3d& merge(const Scene3d& g);
    *
    * Merge two scenes
    */
    Scene3d& merge(const Scene3d& g);


    /*!
    * \param stopprocess stop the process or not
    *
    * display the 3d scene in a windows
    */
    void display(bool stopprocess=true);


    void addGeometricalFigure(const Vec<GeometricalFigure*> figures);
    void clear();
    void lock();
    void unlock();
    void snapshot(const char * file);
    void _snapshot();
    void rotateX(F32 angle);
    void rotateY(F32 angle);
    void rotateZ(F32 angle);
    void setColorAllGeometricalFigure(const RGBUI8 & value);
    void setTransparencyAllGeometricalFigure(UI8 value);

    void setAmbient(const pop::RGBF32 &ambient);
    pop::RGBF32 getAmbient()const;
    void setDiffuse(const pop::RGBF32 &diffuse);
    pop::RGBF32 getDiffuse()const;
    void setTransparentMode(bool istranspararent);
    bool getTransparentMode()const;


};
}
#endif // GEOMETRICALFIGURE_H
