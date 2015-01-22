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
#ifndef VISUALIZATION_H
#define VISUALIZATION_H
#include"data/typeF/TypeF.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

#include"data/3d/GLFigure.h"
#include"data/distribution/DistributionAnalytic.h"

#include"data/notstable/graph/Graph.h"
#include"algorithm/Draw.h"
namespace pop
{

/*!
\defgroup Visualization Visualization
\ingroup Algorithm
\brief Matrix In -> Matrix Out for 2D or OpenGl scene for 3D

- visualization of the result of a 2d process (segmentation or distance function)
\image html outdistance.png "Distance function"
- 3d visualization
\image html Graph.gif "Topological graph"

*/

struct POP_EXPORTS Visualization
{

    /*!
        \class pop::Visualization
        \ingroup Visualization
        \brief matrix visualization with opengl layer
        \author Tariel Vincent

         This class provides some algorithms to visualize an matrix.\n
         Note : when you display an opengl scene, its closure causes the end of the algorithms due to Glut in Linux.
      \sa MatN
    */
    //-------------------------------------
    //
    //! \name Label matrix
    //@{
    //-------------------------------------
    /*!
     * \brief affect a random color for each label
     * \param f input label matrix
     * \return  color matrix
     *
     * affect a random RGB at each label value
     *
     * \code
     * Mat2UI8 img;
     * img.load("../image/outil.bmp");
     * Mat2UI32 label = pop::Processing::clusterToLabel(img);
     * Mat2RGBUI8 RGB = Visualization::labelToRandomRGB(label);
     * \endcode
     * \image html outilcluster.png
    */
    template<int DIM,typename TypePixel>
    static MatN<DIM,RGBUI8> labelToRandomRGB(const MatN<DIM,TypePixel> & f )
    {
        MatN<DIM,RGBUI8> fRGB(f.getDomain());
        typename MatN<DIM,TypePixel>::IteratorEDomain it(f.getIteratorEDomain());

        I32 value1=234;
        I32 value2=27;
        I32 value3=33;


        it.init();
        while(it.next())
        {
            if(normValue(f(it.x()))>0 )
            {
                RGBUI8 v;
                I32 r = 0;
                I32 g = 0;
                I32 b = 0;
                int i=normValue(f(it.x()));
                if(i==1)v=RGBUI8(255,0,0);
                else if (i==2)v=RGBUI8(0,255,0);
                else if (i==3)v=RGBUI8(0,0,255);
                else if (i==255)v=RGBUI8(120,120,120);
                else
                {
                    r = absolute((value1*i+56));
                    g = absolute((value2*i+53));
                    b = absolute((value3*i+11));
                    v=RGBUI8(r%255,g%255,b%255);
                }
                fRGB(it.x())=v;
            }
            else
                fRGB(it.x())=0;
        }
        return fRGB;
    }

    /*!
     * \brief affect a graduaded color for each label
     * \param f input label matrix
     * \param cmin mininum RGB
     * \param cmax maximum RGB
     * \return  RGB matrix
     *
     * affect a color at each label value with a gradation from mininum value associated to the blue value to maximum value associated to the red value
     *
     * \code
     * Mat2UI8 img;
     * img.load("../image/outil.bmp");
     * Mat2UI32 label;
     * label = pop::Processing::distanceEuclidean(img);
     * Mat2RGBUI8 RGB = Visualization::labelToRGBGradation(label);
     * \endcode
     * \image html outdistance.png
    */
    template<int DIM,typename TypePixel>
    static MatN<DIM,RGBUI8> labelToRGBGradation (const MatN<DIM,TypePixel> & f,RGBUI8 cmin=RGBUI8(0,0,255),RGBUI8 cmax=RGBUI8(255,0,0)  )
    {
        MatN<DIM,RGBUI8> fRGB(f.getDomain());
        typename MatN<DIM,TypePixel>::IteratorEDomain it(f.getIteratorEDomain());
        typename MatN<DIM,TypePixel>::F maxi = NumericLimits<typename MatN<DIM,TypePixel>::F>::minimumRange();
        typename MatN<DIM,TypePixel>::F mini = NumericLimits<typename MatN<DIM,TypePixel>::F>::maximumRange();
        while(it.next()){
            if(f(it.x())!=0){
                maxi = maximum(maxi,f(it.x()));
                mini = minimum(mini,f(it.x()));
            }
        }
        //        std::cout<<maxi<<std::endl;
        //        std::cout<<mini<<std::endl;
        RGBUI8 v_init(0);
        std::vector< RGBUI8 > v(maxi+1-mini, v_init);
        RGBF32 cminf,cmaxf;
        cminf = cmin;
        cmaxf = cmax;

        for(I32 i=0;i<=(I32)maxi-(I32)mini;i++)
        {
            F32 dist = i*1.0/(maxi-mini);
            v[i]=dist*(cmaxf-cminf)+cminf;
        }
        it.init();
        while(it.next())
        {
            if(f(it.x())!=0)
                fRGB(it.x())=v[f(it.x())-mini];
        }
        return fRGB;
    }

    /*!
     * \brief affect the mean value in the label area
     * \param label input label matrix
     * \param img matrix grey-level matrix
     * \return RGB matrix
     *
     * out(x)= average(img(x)) where average is the mean value of img in each label
     *
     * \code
     * Mat3UI8 img;
     * img.load("../image/rock3d.pgm");
     * img = pop::Processing::median(img,3,2);

     * Mat3UI8 seed1 = pop::Processing::threshold(img,0,50);
     * Mat3UI8 seed2 = pop::Processing::threshold(img,110,140);
     * Mat3UI8 seed3 = pop::Processing::threshold(img,165,255);

     * seed1 = pop::Processing::labelMerge(seed1,seed2);
     * seed1 = pop::Processing::labelMerge(seed1,seed3);

     * Mat3UI8 gradient = pop::Processing::gradientMagnitudeDeriche(img,1);
     * Mat3UI8 basins =  pop::Processing::watershed(seed1,gradient);

     * img= Visualization::labelAverageRGB(basins,img);
     * Scene3d scene;
     * Visualization::cube(scene,img);
     * scene.display();
     * \endcode
     * \image html average.png
    */
    template<int DIM,typename TypePixel1,typename TypePixel2>
    static MatN<DIM,TypePixel2> labelAverageRGB(const MatN<DIM,TypePixel1> & label,const MatN<DIM,TypePixel2> & img)
    {
        FunctionAssert(label,img,"Visualization::labelAverageRGB");
        typename MatN<DIM,TypePixel1>::IteratorEDomain it (img.getIteratorEDomain());
        MatN<DIM,typename FunctionTypeTraitsSubstituteF<TypePixel2 ,F32>::Result> in(img);
        std::vector<typename FunctionTypeTraitsSubstituteF<TypePixel2 ,F32>::Result> vaverage;
        std::vector<UI32> voccurence;
        it.init();
        while(it.next())
        {
            if((UI32)vaverage.size()<=(UI32)label(it.x()))
            {
                vaverage.resize(label(it.x())+1);
                voccurence.resize(label(it.x())+1);
            }
            vaverage[label(it.x())]=vaverage[label(it.x())]+in(it.x());
            voccurence[label(it.x())]++;
        }
        MatN<DIM,TypePixel2> average(label.getDomain());
        it.init();
        while(it.next())
        {
            F32 div = (1./voccurence[label(it.x())]);
            typename FunctionTypeTraitsSubstituteF<typename MatN<DIM,TypePixel2>::F ,F32>::Result value(vaverage[label(it.x())]);
            value = value*div;
            average(it.x())=value;
        }
        return average;
    }



    /*!
     * \brief add a random color at the label boundary to input image (img)
     * \param label input label matrix
     * \param img matrix grey-level matrix
     * \param norm 1 = 4-neighborhood in 2d and 6 in 3d, and  0=8-neighborhood in 2d and 26 in 3d
     * \param width boundary width
     * \return RGB matrix
     *
     * out(x)=img(x) for label_c(x)=0; img(x)*ratio+label_c(x)*(1-ratio) otherwise, where label_c is random RGB boundary label
     *
     * \code
     * Mat3UI8 img;
     * img.load("../image/rock3d.pgm");
     * img = pop::Processing::median(img,3,2);

     * Mat3UI8 seed1 = pop::Processing::threshold(img,0,50);
     * Mat3UI8 seed2 = pop::Processing::threshold(img,110,140);
     * Mat3UI8 seed3 = pop::Processing::threshold(img,165,255);

     * seed1 = pop::Processing::labelMerge(seed1,seed2);
     * seed1 = pop::Processing::labelMerge(seed1,seed3);

     * Mat3UI8 gradient = pop::Processing::gradientMagnitudeDeriche(img,1);
     * Mat3UI8 basins =  pop::Processing::watershed(seed1,gradient);

     * Mat3RGBUI8 RGB = Visualization::labelForegroundBoundary(basins,img,0.5);
     * Scene3d scene;
     * Visualization::cube(scene,RGB);
     * scene.display();
     * \endcode
     * \image html foregroundboundary.png
    */
    template<int DIM,typename TypePixel1,typename TypePixel2>
    static MatN<DIM,RGBUI8> labelForegroundBoundary(const MatN<DIM,TypePixel1> & label,const MatN<DIM,TypePixel2> & img,int width=1,int norm=1)
    {
        MatN<DIM,TypePixel1>  labelb(label.getDomain());
        typename MatN<DIM,TypePixel1>::IteratorEDomain it (label.getIteratorEDomain());
        typename MatN<DIM,TypePixel1>::IteratorENeighborhood itn(label.getIteratorENeighborhood(width,norm));
        while(it.next()){
            typename MatN<DIM,TypePixel1>::F l = label(it.x());
            itn.init(it.x());
            bool diff=false;
            while(itn.next()){
                if(label(itn.x())!=l)
                    diff=true;
            }
            if(diff==true)
                labelb(it.x())=l;
        }




        FunctorF::FunctorAccumulatorMin<typename MatN<DIM,TypePixel1>::F > funcmini;
        it.init();
        I32 min = forEachFunctorAccumulator(labelb,funcmini,it);
        FunctorF::FunctorAccumulatorMax<typename MatN<DIM,TypePixel1>::F > funcmaxi;
        it.init();
        I32 max = forEachFunctorAccumulator(labelb,funcmaxi,it);


        if(min<0){
            std::cerr<<"In Vizualization::labelForegroundBoundary,the label matrix must be possitive";
        }
        std::vector<RGB<F32> > v(max+1);

        srand ( 1 );

        for(I32 i=0;i<(I32)v.size();i++)
        {
            if(i==1)v[i]=RGB<F32>(255,0,0);
            else if(i==2)v[i]=RGB<F32>(0,255,0);
            else if(i==3)v[i]=RGB<F32>(0,0,255);
            else
                v[i]=RGB<F32>(rand()%256,rand()%256,rand()%256);//255*dist.randomVariable(),255*dist.randomVariable(),255*dist.randomVariable());

        }
        MatN<DIM,RGBUI8> foreground(label.getDomain());


        it.init();
        while(it.next())
        {
            if(labelb(it.x())==0)
            {
                RGB<F32> value(img(it.x()));
                foreground(it.x())=value;
            }
            else
            {
                foreground(it.x())=v[labelb(it.x())];
            }
        }
        return foreground;
    }

    /*!
     * \brief ratio value between a random color for each label and the image value
     * \param label input label matrix
     * \param img matrix grey-level matrix
     * \param ratio ratio between label and grey-level matrixs
     * \return RGB matrix
     *
     * out(x)=img(x) for label(x)=0; img(x)*ratio+label_c(x)*(1-ratio) otherwise, where label_c is random RGB label
     *
     * \code
     * Mat3UI8 img;
     * img.load("../image/rock3d.pgm");
     * img = pop::Processing::median(img,3,2);

     * Mat3UI8 seed1 = pop::Processing::threshold(img,0,50);
     * Mat3UI8 seed2 = pop::Processing::threshold(img,110,140);
     * Mat3UI8 seed3 = pop::Processing::threshold(img,165,255);

     * seed1 = pop::Processing::labelMerge(seed1,seed2);
     * seed1 = pop::Processing::labelMerge(seed1,seed3);

     * Mat3UI8 gradient = pop::Processing::gradientMagnitudeDeriche(img,1);
     * Mat3UI8 basins =  pop::Processing::watershed(seed1,gradient);

     * Mat3RGBUI8 RGB = Visualization::labelForeground(basins,img,0.5);
     * Scene3d scene;
     * Visualization::cube(scene,RGB);
     * scene.display();
     * \endcode
     * \image html foreground.png
    */

    template<int DIM,typename TypePixel1,typename TypePixel2>
    static MatN<DIM,RGBUI8> labelForeground(const MatN<DIM,TypePixel1> & label,const MatN<DIM,TypePixel2> & img,F32 ratio=0.5)
    {
        FunctionAssert(label,img,"Visualization::labelForeground");
        typename MatN<DIM,TypePixel1>::IteratorEDomain it (label.getIteratorEDomain());

        FunctorF::FunctorAccumulatorMin<typename MatN<DIM,TypePixel1>::F > funcmini;
        it.init();
        I32 min = forEachFunctorAccumulator(label,funcmini,it);
        FunctorF::FunctorAccumulatorMax<typename MatN<DIM,TypePixel1>::F > funcmaxi;
        it.init();
        I32 max = forEachFunctorAccumulator(label,funcmaxi,it);


        if(min<0){
            std::cerr<<"In Vizualization::labelForeground, the label matrix must be possitive";
        }
        std::vector<RGB<F32> > v(max+1);

        srand ( 1 );

        for(I32 i=0;i<(I32)v.size();i++)
        {
            if(i==1)v[i]=RGB<F32>(255,0,0);
            else if(i==2)v[i]=RGB<F32>(0,255,0);
            else if(i==3)v[i]=RGB<F32>(0,0,255);
            else
                v[i]=RGB<F32>(rand()%256,rand()%256,rand()%256);//255*dist.randomVariable(),255*dist.randomVariable(),255*dist.randomVariable());

        }
        MatN<DIM,RGBUI8> foreground(label.getDomain());


        it.init();
        while(it.next())
        {
            if(label(it.x())==0)
            {
                RGB<F32> value(img(it.x()));
                foreground(it.x())=value;
            }
            else
            {
                RGB<F32> value(img(it.x()));
                RGB<F32> value2 =  value*ratio+ v[label(it.x())]*(1-ratio);
                foreground(it.x())=value2;
            }
        }
        return foreground;
    }

    //@}
    //-------------------------------------
    //
    //! \name OpenGl
    //@{
    //-------------------------------------

    /*!
     * \brief  add the faces of the input 3d matrix to the opengl scene
     * \param scene input/output opengl scene
     * \param m input 3d matrix
     *
     * extract the faces of the 3d matrix and add their to the opengl scene
     *
     * \code
     * Mat3UI8 img;
     * img.load("../image/rock3d.pgm");
     * Scene3d scene;
     * Visualization::cube(scene,img);
     * scene.display();
     * \endcode
     * \image html cube.png
    */
    template<typename TypePixel>
    static void cube(Scene3d& scene,const MatN<3,TypePixel> & m)
    {


        Visualization::plane(scene,m,0,0,-1);
        Visualization::plane(scene,m,0,1,-1);
        Visualization::plane(scene,m,0,2,-1);
        Visualization::plane(scene,m,m.getDomain()(0)-1,0,1,Vec3F32(1,0,0));
        Visualization::plane(scene,m,m.getDomain()(1)-1,1,1,Vec3F32(0,1,0));
        Visualization::plane(scene,m,m.getDomain()(2)-1,2,1,Vec3F32(0,0,1));
    }
    /*!
     * \brief  add red lines to the opengl scene
     * \param scene input/output opengl scene
     * \param img input 3d matrix
     * \param width line width
     * \param RGB line RGB
     *
     * extract the line of cube and add their to the opengl scene
     *
     * \code
     * Mat3UI8 img;
     * img.load("../image/rock3d.pgm");
     * Scene3d scene;
     * Visualization::cube(scene,img);
     * Visualization::lineCube(scene,img);
     * scene.display();
     * \endcode
     * \image html cubeline.png
    */
    template<typename TypePixel>
    static void lineCube(Scene3d& scene,const MatN<3,TypePixel> & img,F32 width=2,RGBUI8 RGB=RGBUI8(255,0,0))
    {

        int d0 = img.getDomain()(0);
        int d1 = img.getDomain()(1);
        int d2 = img.getDomain()(2);

        VecN<3,F32> x1,x2;
        x1(0)=0;x1(1)=0;x1(2)=0;

        x2(0)=d0;x2(1)=0;x2(2)=0;

        FigureLine * line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;

        scene._v_figure.push_back(line);
        x2(0)=0;x2(1)=d1;x2(2)=0;


        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;


        scene._v_figure.push_back(line);

        x2(0)=0;x2(1)=0;x2(2)=d2;
        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;


        scene._v_figure.push_back(line);

        x1(0)=d0;x1(1)=d1;x1(2)=0;
        x2(0)=d0;x2(1)=0;x2(2)=0;

        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;


        scene._v_figure.push_back(line);

        x2(0)=0;x2(1)=d1;x2(2)=0;

        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;


        scene._v_figure.push_back(line);

        x2(0)=d0;x2(1)=d1;x2(2)=d2;

        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;


        scene._v_figure.push_back(line);


        x1(0)=0;x1(1)=d1;x1(2)=d2;

        x2(0)=d0;x2(1)=d1;x2(2)=d2;

        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;



        scene._v_figure.push_back(line);

        x2(0)=0;x2(1)=0;x2(2)=d2;

        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;



        scene._v_figure.push_back(line);

        x2(0)=0;x2(1)=d1;x2(2)=0;

        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;



        scene._v_figure.push_back(line);


        x1(0)=d0;x1(1)=0;x1(2)=d2;

        x2(0)=d0;x2(1)=d1;x2(2)=d2;

        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;



        scene._v_figure.push_back(line);

        x2(0)=0;x2(1)=0;x2(2)=d2;

        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;



        scene._v_figure.push_back(line);

        x2(0)=d0;x2(1)=0;x2(2)=0;

        line = new FigureLine();
        line->x1 = x1;
        line->x2 = x2;
        line->setRGB(RGB);
        line->width=width;


        scene._v_figure.push_back(line);
    }
    /*!
     * \brief add the marching cube mesh of the input label image
     * \param scene input/output opengl scene
     * \param img input 3d matrix
     *
     * Apply the marching cube algorithm to the boundary defined between two neightborhood voxels with one equal to 0 and another different to zero (the face take the color of this value)
     *
     * \code
    Mat3UI8 img;
    img.load("../image/rock3d.pgm");
//    img = img(Vec3I32(0,0,0),Vec3I32(64,64,64));

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
    Mat3UI8 grain_oil = grain + oil;
    Scene3d scene;
    Visualization::marchingCube(scene,grain_oil);
    Visualization::lineCube(scene,grain_oil);
    scene.display();
     * \endcode
     * \image html marchingcube.png
    */
    template<typename TypePixel>
    static void marchingCube(Scene3d& scene,const MatN<3,TypePixel> & img )
    {

        MatN<3,RGBUI8 >   binc;
        binc = img;
        //        if(dynamic_cast< MatN<3,RGBUI8 >  * >(&img)){
        //            binc = img;
        //        }else {
        //            Convertor::fromRGB(img,img,img,binc);
        //        }
        std::vector<std::pair<_vertex,RGBUI8 > > vertices = _runMarchingCubes2(binc);
        while(vertices.empty()==false)
        {
            _vertex vert = vertices.back().first;
            RGBUI8 RGB = vertices.back().second;
            vertices.pop_back();
            FigureTriangle * triangle = new FigureTriangle();
            triangle->normal(0)=vert.normal_x;triangle->normal(1)=vert.normal_y;triangle->normal(2)=vert.normal_z;
            triangle->x(0)=vert.x-2;triangle->x(1)=vert.y-2;triangle->x(2)=vert.z-2;

            triangle->setRGB(RGB);
            scene._v_figure.push_back(triangle);
        }
    }




    /*!
     * \brief add the marching cube mesh of the 0-level set of a continious field
     * \param scene input/output opengl scene
     * \param phasefied input  phasefied
     *
     * Marching cube on the level set 0 of the input continious  field
     * \code
        //Generate a random field
        Mat3F32 img_time_t(100,100,100);
        DistributionSign sgn;
        ForEachDomain3D(x,img_time_t){
            img_time_t(x)=sgn.randomVariable();
        }
        //evolution governed by the diffusion equation
        int dimension = 3;
        unsigned int time=20;
        F32 D = 0.25/dimension;//for stability
        Mat3F32 img_time_t_plus_1(img_time_t);
        FunctorPDE::Laplacien<> laplacien;
        for(unsigned int i =0;i<time;i++){
            std::cout<<"time "<<i<<std::endl;
            ForEachDomain3D(xx,img_time_t){
                img_time_t_plus_1(xx)=  img_time_t(xx)+  D*  laplacien(img_time_t,xx);//finite difference diffusion equation
            }
            img_time_t = img_time_t_plus_1;
        }
    Scene3d scene;
        Visualization::marchingCubeLevelSet(scene, (img_time_t));
        Visualization::lineCube(scene,img_time_t);
    scene.display();
     * \endcode
     * \image html spinodal.png
    */

    static inline void marchingCubeLevelSet(Scene3d& scene,const MatN<3,F32> & phasefied)
    {


        std::vector<_vertex> vertices = _runMarchingCubes2(phasefied,0);
        //        MatN<3,RGBUI8 >::IteratorENeighborhood itn (maRGB.getIteratorENeighborhood(1,0));
        while(vertices.empty()==false)
        {
            _vertex vert = vertices.back();
            vertices.pop_back();
            FigureTriangle * triangle = new FigureTriangle();
            triangle->normal(0)=vert.normal_x;triangle->normal(1)=vert.normal_y;triangle->normal(2)=vert.normal_z;
            triangle->x(0)=vert.x-2;triangle->x(1)=vert.y-2;triangle->x(2)=vert.z-2;

            //            if(maRGB.getDomain()(0)==0){
            RGBUI8 c(200,200,200);
            triangle->setRGB(c);
            //            }else
            //            {
            //                itn.init(triangle->x);
            //                RGBUI8 c;
            //                while(itn.next()){
            //                    c = max(c,maRGBUI8(itn.x()));
            //                }
            //                triangle->setRGB(c);
            //            }
            scene._v_figure.push_back(triangle);
        }
    }
    /*!
     * \brief  add the marching cube mesh of the 0-level set of a continious field with colored mesh
     * \param scene input/output opengl scene
     * \param phasefied input  phasefied
     * \param RGBfield input  RGB fied
     *
     * Marching cube on the level set 0 of the input phase field with the color map to dress the mesh
     *
     * \code
     * Mat3UI8 img;
     * img.load("../image/rock3d.pgm");

     * Mat3UI8 imgfilter= Processing::median(img,2);
     * Mat3UI8 grain= Processing::threshold(imgfilter,155);
     * Mat3UI16 dist = Processing::distanceEuclidean(grain.opposite())*4;
     * //dynamic filter to avoid over-partition
     * dist = Processing::dynamic(dist.opposite()-20,4,0);
     * Mat3UI32 minima = pop::Processing::minimaRegional(dist,0);
     * Mat3UI32 water = pop::Processing::watershed(minima,dist,grain,1);
     * Mat3F32 phasefield = PDE::allenCahn(grain,5);
     * phasefield = PDE::getField(grain,phasefield,1,6);

     * Scene3d scene;
     * Visualization::marchingCubeLevelSet(scene,phasefield,Visualization::labelToRandomRGB(water));
     * Visualization::lineCube(scene,phasefield);
     * scene.display();
     * \endcode
     * \image html graindecomposition.png
    */

    static inline void marchingCubeLevelSet(Scene3d& scene,const MatN<3,F32> & phasefied,const MatN<3,RGBUI8>  & RGBfield)
    {
        std::vector<_vertex> vertices = _runMarchingCubes2(phasefied,0);
        while(vertices.empty()==false)
        {
            _vertex vert = vertices.back();
            vertices.pop_back();
            FigureTriangle * triangle = new FigureTriangle();
            triangle->normal(0)=vert.normal_x;triangle->normal(1)=vert.normal_y;triangle->normal(2)=vert.normal_z;
            triangle->x(0)=vert.x-2;triangle->x(1)=vert.y-2;triangle->x(2)=vert.z-2;
            MatN<3,F32>::E x;
            x= triangle->x;

            MatN<3,RGBUI8>::IteratorENeighborhood itn(RGBfield.getIteratorENeighborhood(1,0));
            if(RGBfield.isValid(x)){
                RGBUI8 fRGB(RGBfield(x));
                if((fRGB==RGBUI8(0,0,0))){
                    itn.init(x);
                    while(itn.next()){
                        RGBUI8 fRGB_neight(RGBfield(itn.x()));
                        fRGB = maximum(fRGB,fRGB_neight);
                    }
                }
                triangle->setRGB(fRGB);
            }
            scene._v_figure.push_back(triangle);
        }

    }


    /*!
     * \brief surface voxel boundary
     * \param scene input/output opengl scene
     * \param img input 3d matrix
     *
     * Extract the voxel boundary defined between the voxels of value 0 and the others
     *
     * \code
        Mat3UI8 img;
        img.load("../image/rock3d.pgm");
        img = img(Vec3I32(0,0,0),Vec3I32(100,100,100));
        Mat3UI8 imgfilter= Processing::median(img,2);
        Mat3UI8 pore_space = Processing::threshold(imgfilter,0,155);
        pore_space=   pop::Processing::holeFilling(pore_space);
        Mat3UI8 skeleton= Analysis::thinningAtConstantTopology(pore_space,"../file/topo24.dat");
        Scene3d scene;
        pop::Visualization::voxelSurface(scene,skeleton);
        pop::Visualization::lineCube(scene,skeleton);
        scene.display();
     * \endcode
     * \image html cube.png "initial image"
     * \image html spinodal_skeleton.png "Topological skeleton"
    */
    template<typename TypePixel>
    static void voxelSurface(Scene3d & scene , const MatN<3,TypePixel> & img)
    {
        MatN<3,TypePixel> f(img.getDomain()+2);
        typename MatN<3,TypePixel>::IteratorEDomain it (img.getIteratorEDomain());

        while(it.next())
        {
            VecN<3,int> x  =it.x()+1;
            x=x+1;
            f(x)=img(it.x());
        }

        typename MatN<3,TypePixel>::IteratorEDomain itg(f.getIteratorEDomain());
        typename MatN<3,TypePixel>::IteratorENeighborhood itn(f.getIteratorENeighborhood(1,1));

        while(itg.next()){
            if(f(itg.x())!=typename MatN<3,TypePixel>::F(0) ){
                itn.init(itg.x());
                while(itn.next()){
                    if(f(itn.x())==typename MatN<3,TypePixel>::F(0)){
                        FigureUnitSquare * square = new FigureUnitSquare();
                        RGBUI8 c(f(itg.x()));
                        square->setRGB(c);
                        square->setTransparent(255);
                        square->x = itg.x()-1;
                        for(int i = 0;i<=2;i++ ){
                            if(itg.x()(i)!=itn.x()(i)){
                                square->direction = i;
                                square->way = (itn.x()(i)-itg.x()(i));
                                if((itn.x()(i)-itg.x()(i))>0)
                                    square->x(i)++;
                            }
                        }
                        scene._v_figure.push_back(square);
                    }
                }
            }
        }
    }


    /*!
     * \brief add coordinate axes (red arrow =x, green arrow=y,  blue arrow =z)
     * \param scene input/output opengl scene
     * \param length length of the arrows
     * \param width width of the arrows
     * \param trans_minus tanslated the axes from the origin (optional)
     *
     *
     * \code
     * Mat3UI8 img("../image/rock3d.pgm");
     * Scene3d scene;
     * Visualization::cubeExtruded(scene,img);//add the extruded cube surfaces to the scene
     * Visualization::lineCube(scene,img);//add the border red lines to the scene to the scene
     * Visualization::axis(scene,40);//add axis
     * scene.display();//display the scene
     * \endcode
     * \image html cubeExtruded.png
    */

    static void axis(Scene3d &scene, F32 length =20,F32 width=3,F32 trans_minus=2);



    /*!
     * \brief add the extruded cube to the scene
     * \param scene input/output opengl scene
     * \param m input 3d matrix
     * \param extrusion extrusion binary matrix (optional)
     *
     * extract the faces of the 3d input matrix inside the binary extrusion  matrix
     *
     * With the default  binary extrusion  matrix
     * \code
     * Mat3UI8 img("../image/rock3d.pgm");
     * Scene3d scene;
     * Visualization::cubeExtruded(scene,img);//add the extruded cube surfaces to the scene
     * Visualization::lineCube(scene,img);//add the border red lines to the scene to the scene
     * Visualization::axis(scene,40);//add axis
     * scene.display();//display the scene
     * \endcode
     * \image html cubeExtruded.png
     * With my own binary extrusion  matrix
     * \code
     * Mat3UI8 img("../image/rock3d.pgm");
     * Scene3d scene;
     * Mat3UI8 extruded(img.getDomain());
     * int radius=img.getDomain()(0)/2;
     * Vec3I32 x1(0,0,0);
     * Vec3I32 x2(img.getDomain());
     * ForEachDomain3D(x,extruded){
     *  if((x-x1).norm(2)<radius||(x-x2).norm(2)<radius)
     *      extruded(x)=0;
     *   else
     *       extruded(x)=255;
     * }
     * Visualization::cubeExtruded(scene,img,extruded);//add the cube surfaces to the scene
     * Visualization::lineCube(scene,img);//add the border red lines to the scene to the scene
     * Visualization::axis(scene,40);//add axis
     * scene.display();//display the scene
     * \endcode
     * \image html cubeExtruded2.png
    */

    template<typename TypeVoxel1>
    static void cubeExtruded(Scene3d & scene, const MatN<3,TypeVoxel1> & m,const MatN<3,UI8>& extrusion =MatN<3,UI8>());


    /*!
     * \brief add slice of the 3d matrix to the scene
     * \param scene input/output opengl scene
     * \param img input 3d matrix
     * \param slice slice index
     * \param direction coordinate
     * \param normal_way opengl parameter
     * \param trans translation factor of all elements
     *
     * add a plane of the matrix at a given slice and direrection to the opengl scene
     *
     * \code
     * Mat3UI8 img;
     * img.load("../image/rock3d.pgm");
     * Scene3d scene;
     * Visualization::plane(scene,img,50,2);
     * Visualization::plane(scene,img,50,1);
     * Visualization::plane(scene,img,200,0);
     * Visualization::lineCube(scene,img);

     * scene.display();
     * \endcode
     * \image html planegl.png
    */

    template<typename TypePixel>
    static void  plane(Scene3d &scene, const MatN<3,TypePixel> & img,int slice=0, int direction=2,int normal_way=1,Vec3F32 trans = Vec3F32())
    {




        if(direction<0||direction>=3)
            direction=2;
        if(slice>=img.getDomain()(direction))
            slice =img.getDomain()(direction)-1;
        if(slice<0)
            slice =0;


        MatN<2,TypePixel> hyperff(img.getDomain().removeCoordinate(direction));
        typename MatN<2,TypePixel> ::IteratorEDomain it_plane(hyperff.getIteratorEDomain());
        while(it_plane.next()){
            hyperff(it_plane.x()) = img(it_plane.x().addCoordinate(direction,slice));
        }
        typename MatN<2,TypePixel>::IteratorEDomain it (hyperff.getIteratorEDomain());
        VecN<3,F32 > normal;
        if(normal_way==1)
            normal(direction)=1;
        else
            normal(direction)=-1;
        VecN<3,F32 > add1;
        add1=0;
        add1( (direction+1)%3)=1;
        VecN<3,F32 > add2;
        add2=0;
        add2( (direction+1)%3)=1;add2( (direction+2)%3)=1;
        VecN<3,F32 > add3;
        add3=0;
        add3( (direction+2)%3)=1;

        VecN<3,F32> x;
        VecN<3,F32> y;
        x(direction)=slice;
        while(it.next())
        {
            FigurePolygon * poly = new FigurePolygon();
            RGBUI8 c(hyperff(it.x()));
            poly->setRGB( c);
            poly->setTransparent(255);
            poly->normal= normal;
            for(int i=0;i<3;i++)
            {
                if(i<direction)
                    x(i)=it.x()(i);
                else if(i>direction)
                    x(i)=it.x()(i-1);
            }
            y =x;
            poly->vpos.push_back(x+trans);
            y = x+add1;
            poly->vpos.push_back(y+trans);
            y = x+add2;
            poly->vpos.push_back(y+trans);
            y = x+add3;
            poly->vpos.push_back(y+trans);
            scene._v_figure.push_back(poly);
        }
    }
    /*!
     * \brief add graph to the scene
     * \param scene input/output opengl scene
     * \param g input graph
     *
     *
     * \code
    Mat3UI8 grain;
    grain.load("../image/spinodal.pgm");
    grain = grain(Vec3I32(0,0,0),Vec3I32(100,100,100));
    grain = pop::Processing::greylevelRemoveEmptyValue(grain);//the grain label is now 1 (before 255)
    //TOLOGICAL GRAPH
    Mat3UI8 grain_hole=   pop::Processing::holeFilling(grain);
    Mat3UI8 skeleton= Analysis::thinningAtConstantTopology(grain_hole,"../file/topo24.dat");
    std::pair<Mat3UI8,Mat3UI8> vertex_edge = Analysis::fromSkeletonToVertexAndEdge (skeleton);
    Mat3UI32 verteces = pop::Processing::clusterToLabel(vertex_edge.first,0);
    Mat3UI32 edges = pop::Processing::clusterToLabel(vertex_edge.second,0);
    int tore;
    GraphAdjencyList<Vec3I32> g = Analysis::linkEdgeVertex(verteces,edges,tore);
    Mat3F32 phasefield = PDE::allenCahn(grain,20);
    phasefield = PDE::getField(grain,phasefield,1,6);
    Scene3d scene;
    Visualization::marchingCubeLevelSet(scene,phasefield);
    scene.setTransparencyAllGeometricalFigure(40);
    scene.setTransparentMode(true);
    Visualization::graph(scene,g);
    Visualization::lineCube(scene,edges);
    scene.display(false);

    int i=0;
    while(1==1){
        scene.lock();
        i++;
         std::string file = "Graph"+BasicUtility::IntFixedDigit2String(i,4)+".pgm";
        scene.rotateZ(5);
        std::cout<<i<<std::endl;
        if(i==100)
            return 1;
        scene.unlock();
        scene.snapshot(file.c_str());
    }
    * \endcode
    * \image html Graph.gif
    */
    template<typename Scalar>
    static void  graph(Scene3d &scene,  GraphAdjencyList<VecN<3,Scalar> > & g)
    {
        RGB<unsigned char> c1(255,0,0);
        for ( int i =0; i < (int)g.sizeVertex(); i++ ){
            FigureSphere *  sphere =  new FigureSphere;
            VecF32 v  =g.vertex(i);
            Vec<F32> vv;
            vv = v;
            sphere->_x=vv;
            sphere->_radius=1;
            sphere->setRGB(c1);
            sphere->setTransparent(255);
            scene._v_figure.push_back(sphere);

        }
        RGB<unsigned char> c2(0,255,0);
        for ( int i =0; i < (int)g.sizeEdge(); i++ ){
            FigureLine * line = new FigureLine;
            std::pair<int,int> p  =g.getLink(i);
            VecF32 v1  =g.vertex(p.first);
            Vec<F32> vv;
            vv = v1;
            line->x1= vv;
            VecF32 v2  =g.vertex(p.second);
            vv =v2;
            line->x2=vv;
            line->setTransparent(255);
            line->width=1;
            line->setRGB( c2);
            scene._v_figure.push_back(line);
        }

    }
    /*!
     * \brief add the topographic surface to the scene
     * \param scene input/output opengl scene
     * \param img topographic surface
     *
     * \code
     * DistributionMultiVariate d("-100*(x-0.5)^2*(y-0.5)^2+2","x,y");

     * VecF32 xmin(2),xmax(2);
     * xmin(0)=0;xmin(1)=0;
     * xmax(0)=1;xmax(1)=1;
     * Mat2F32 m = Statistics::toMatrix(d,xmin,xmax,0.1);
     * Scene3d scene;
     * Visualization::topography(scene,m);
     * Visualization::axis(scene,2,0.5,0);
     * scene.display();
     * \endcode
     * \image html tomatrix.jpg
     */

    template<typename Type>
    static void  topography(Scene3d &scene, const MatN<2,Type> & img)
    {
        Type maxi = 0;
        ForEachDomain2D(x,img){
            maxi = maximum(maxi,img(x));
            Vec2I32 x1(x),x2(x),x3(x);
            x2=x;x2(0)--;
            if(img.isValid(x2)){
                x3=x;x3(1)--;
                if(img.isValid(x3)){
                    Vec3F32 xx1,xx2,xx3;
                    xx1(0)=x1(0);xx1(1)=x1(1);xx1(2)=img(x1);
                    xx2(0)=x2(0);xx2(1)=x2(1);xx2(2)=img(x2);
                    xx3(0)=x3(0);xx3(1)=x3(1);xx3(2)=img(x3);
                    FigurePolygon * poly = FigurePolygon::createTriangle(xx1,xx2,xx3);
                    if(poly->normal(2)<0)
                        poly->normal =-poly->normal;
                    scene._v_figure.push_back(poly);
                }
                x3=x;x3(1)++;
                if(img.isValid(x3)){
                    Vec3F32 xx1,xx2,xx3;
                    xx1(0)=x1(0);xx1(1)=x1(1);xx1(2)=img(x1);
                    xx2(0)=x2(0);xx2(1)=x2(1);xx2(2)=img(x2);
                    xx3(0)=x3(0);xx3(1)=x3(1);xx3(2)=img(x3);
                    FigurePolygon * poly = FigurePolygon::createTriangle(xx1,xx2,xx3);
                    if(poly->normal(2)<0)
                        poly->normal =-poly->normal;
                    scene._v_figure.push_back(poly);
                }
            }
            x2=x;x2(0)++;
            if(img.isValid(x2)){
                x3=x;x3(1)--;
                if(img.isValid(x3)){
                    Vec3F32 xx1,xx2,xx3;
                    xx1(0)=x1(0);xx1(1)=x1(1);xx1(2)=img(x1);
                    xx2(0)=x2(0);xx2(1)=x2(1);xx2(2)=img(x2);
                    xx3(0)=x3(0);xx3(1)=x3(1);xx3(2)=img(x3);
                    FigurePolygon * poly = FigurePolygon::createTriangle(xx1,xx2,xx3);
                    if(poly->normal(2)<0)
                        poly->normal =-poly->normal;
                    scene._v_figure.push_back(poly);
                }
                x3=x;x3(1)++;
                if(img.isValid(x3)){
                    Vec3F32 xx1,xx2,xx3;
                    xx1(0)=x1(0);xx1(1)=x1(1);xx1(2)=img(x1);
                    xx2(0)=x2(0);xx2(1)=x2(1);xx2(2)=img(x2);
                    xx3(0)=x3(0);xx3(1)=x3(1);xx3(2)=img(x3);
                    FigurePolygon * poly = FigurePolygon::createTriangle(xx1,xx2,xx3);
                    if(poly->normal(2)<0)
                        poly->normal =-poly->normal;
                    scene._v_figure.push_back(poly);
                }
            }

        }
//        std::cout<<maxi<<std::endl;
//        F32 sizepeak=maximum(1.,img.getDomain()(0)*0.05);
//        FigureArrow * arrow = new FigureArrow;
//        arrow->setRGB(RGBUI8(255,0,0));arrow->setArrow(Vec3F32(0,0,0),Vec3F32(img.getDomain()(0)+2,0,0),sizepeak);scene._v_figure.push_back(arrow);
//        arrow = new FigureArrow;
//        arrow->setRGB(RGBUI8(255,0,0));arrow->setArrow(Vec3F32(0,0,0),Vec3F32(0,img.getDomain()(1)+2,0),sizepeak);scene._v_figure.push_back(arrow);
//        arrow = new FigureArrow;
//        arrow->setRGB(RGBUI8(255,0,0));arrow->setArrow(Vec3F32(0,0,0),Vec3F32(0,0,maxi+2),sizepeak);scene._v_figure.push_back(arrow);

    }
    //@}
    //-------------------------------------
    //
    //! \name Vec field
    //@{
    //-------------------------------------

    /*!
     * \brief draw arrow associated to a vector field
     * \param vectorfield input vector field
     * \param cmin min RGB
     * \param cmax max RGB
     * \param step step between two arrows
     * \param length size of the arrow between the fixed one and the variable given but the norm of the vector
     *
     *
    \code
    Mat2UI8 img;
    img.load("../image/outil.bmp");
    img = img.opposite();
    Mat2Vec2F32 vel;
    int dir=0;
    PDE::permeability(img,vel,dir,0.1);
    vel= GeometricalTransformation::scale(vel,Vec2F32(8));
    Mat2RGBUI8 c = Visualization::vectorField2DToArrows(vel);
    c.display("velocity",true,false);
    \endcode
    The matrix is the final matrix of this animation:
    \image html outilvelocity.gif
    */
    template<int DIM,typename TypePixel>
    static MatN<DIM,RGBUI8> vectorField2DToArrows(const MatN<DIM,VecN<DIM,TypePixel> > & vectorfield,RGBUI8 cmin=RGBUI8(0,0,255),RGBUI8 cmax=RGBUI8(255,0,0),int step=30,F32 length=90){
        typename MatN<DIM,TypePixel>::IteratorEDomain it(vectorfield.getIteratorEDomain());
        TypePixel maxi = NumericLimits<TypePixel   >::minimumRange();
        TypePixel mini = NumericLimits<TypePixel   >::maximumRange();
        while(it.next()){
            maxi = maximum(maxi,vectorfield(it.x()).norm());
            mini = minimum(mini,vectorfield(it.x()).norm());
        }
        RGBUI8 v_init(0);
        std::vector< RGBUI8 > v(maxi+1-mini, v_init);
        RGBF32 cminf,cmaxf;
        cminf = cmin;
        cmaxf = cmax;

        for(I32 i=0;i<=maxi-mini;i++)
        {
            F32 dist = i*1.0/(maxi-mini);
            v[i]=dist*(cmaxf-cminf)+cminf;
        }
        MatN<DIM,RGBUI8> fRGB(vectorfield.getDomain());

        it.init();
        while(it.next()){
            if(it.x()(0)%step==0&&it.x()(1)%step==0){
                pop::F32 value = vectorfield(it.x()).norm();
                if(value!=0){
                    VecN<DIM,F32> x1 = it.x();
                    VecN<DIM,F32> x2 = x1 + (value-mini)*1.f/(maxi-mini)*length*vectorfield(it.x())/vectorfield(it.x()).norm();
                    Draw::arrow(fRGB,x1,x2,v[value-mini]);
                }
            }
        }
        return fRGB;
    }



    //@}

private:
    struct _vertex {
        F32 x, y, z, normal_x, normal_y, normal_z;
    };


    struct _Cube {
        _vertex p[8];
        F32 val[8];
    };
    struct _cubeF {
        _vertex p[8];
        F32 val[8];
    };


    static void _processCube(_Cube cube, std::vector<std::pair<_vertex,RGBUI8 > >& vertexList,RGBUI8 value,bool diff=false);
    static void _processCubeIso(_Cube cube, std::vector<std::pair<_vertex,RGBUI8 > >& vertexList,RGBUI8 value,unsigned char isolevel);
    static void _processCube(_cubeF cube, std::vector<_vertex>& vertexList,F32 isolevel =0.5,bool diff=false);
    static std::vector<std::pair<_vertex,RGBUI8 > > _runMarchingCubes2(const MatN<3,RGB<UI8 > > &voxel);
    static std::vector<_vertex > _runMarchingCubes2(const MatN<3,F32 > &phasefield,F32 isosurface) ;
    static std::vector<std::pair<_vertex,RGBUI8 > > _runMarchingCubesSurfaceContact(const MatN<3,RGB<UI8 > > &voxel);
    static Visualization::_vertex _interpolate(Visualization::_vertex p1, Visualization::_vertex p2, F32 p1value=1, F32 p2value=0 , F32 iso=0.5 );
    static F32 _affectRGB(RGBUI8 c1,RGBUI8 c2);

};

template<typename TypeVoxel1>
void Visualization::cubeExtruded(Scene3d & scene, const MatN<3,TypeVoxel1> & m,const MatN<3,UI8>& extrustion ){

    MatN<3,UI8> extrustion1(extrustion);
    if(extrustion1.isEmpty()==true){
        extrustion1.resize(m.getDomain());
        ForEachDomain3D(x,extrustion1){
            if(x(0)<m.getDomain()(0)/2&&x(1)<m.getDomain()(1)/2&&x(2)>=m.getDomain()(2)/2)
                extrustion1(x) = 0;
            else
                extrustion1(x) = 255;
        }
    }


    MatN<3,UI8> extrustion2(extrustion1.getDomain()+2);
    ForEachDomain3D(xx,extrustion1){
        extrustion2(xx+1)=extrustion1(xx);
    }
    typename MatN<3,UI8>::IteratorENeighborhood itn = extrustion2.getIteratorENeighborhood(1,1);
    Vec<GeometricalFigure*> vec;
    ForEachDomain3D(x,extrustion1)
    {

        Vec3I32 xplus = x+1;
        if(extrustion2(xplus)!=0){
            itn.init(xplus);
            while(itn.next()){
                if(extrustion2(itn.x())==0){
                    FigurePolygon * poly = new FigurePolygon();
                    RGBUI8 c(m(x));
                    poly->setRGB( c);
                    poly->setTransparent(255);
                    poly->normal= itn.x()-xplus;

                    Vec3I32 add;
                    if(poly->normal(0)!=0)add(1)=1;if(poly->normal(1)!=0)add(2)=1;if(poly->normal(2)!=0)add(0)=1;
                    Vec3I32 add1;
                    if(poly->normal(0)!=0)add1(2)=1;if(poly->normal(1)!=0)add1(0)=1;if(poly->normal(2)!=0)add1(1)=1;
                    Vec3I32 add_coint;
                    if(poly->normal>0)
                        add_coint=poly->normal;
                    poly->vpos.push_back(x+add_coint);
                    poly->vpos.push_back(x+add+add_coint);
                    poly->vpos.push_back(x+add+add1+add_coint);
                    poly->vpos.push_back(x+add1+add_coint);
                    vec.push_back(poly);
                }
            }
        }
    }
    scene.addGeometricalFigure(vec);

}

}
#endif // VISUALIZATION_H
