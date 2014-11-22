/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012, Tariel Vincent

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
#ifndef RANDOMGEOMETRY_H
#define RANDOMGEOMETRY_H

#include"data/utility/CollectorExecutionInformation.h"
#include"data/typeF/RGB.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

#include"data/distribution/Distribution.h"
#include"data/distribution/DistributionMultiVariate.h"
#include"data/germgrain/GermGrain.h"
#include"data/utility/BasicUtility.h"
#include"data/distribution/DistributionMultiVariateFromDataStructure.h"
#include"algorithm/Statistics.h"
#include"algorithm/Processing.h"
#include"data/mat/MatNDisplay.h"
#include"data/utility/BSPTree.h"
namespace pop
{
/*!
\defgroup RandomGeometry RandomGeometry
\ingroup Algorithm
\brief Germ/Grain framework


The germ-grain model includes four components:
   - a germ, a set of random points, \f$\phi=\{\mathbf{x}_0,\ldots,\mathbf{x}_{n}\}\f$,
   - a color, a set of random color, \f$\Phi=\{(\mathbf{x}_0,c_0),\ldots,(\mathbf{x}_{n},c_n)\}\f$, dressing the germ,
   - a grain, a set of random sets, \f$\Phi=\{X_0(\mathbf{x}_0,c_0),\ldots,X_{n}(\mathbf{x}_{n},c_n)\}\f$, dressing the germ,
   - a model, defining the random structure/function, \f$\varTheta\f$, using the grain.

The ModelGermGrain class represent this model.The aim of the RandomGeometry class is to construct the compenents of this model. For instance, in this
following code, we construct the boolean model with disk at a given expected porosity.

 \code
double porosity=0.4;
double mean=20;
double standard_deviation=7;
Distribution dnormal (mean,standard_deviation,"NORMAL");//Normal distribution for disk radius
double moment_order_2 = pop::Statistics::moment(dnormal,2,0,40);
double surface_expectation = moment_order_2*3.14159265;
Vec2F64 domain(1024);//2d field domain
double lambda=-std::log(porosity)/std::log(2.718)/surface_expectation;
ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process
RandomGeometry::sphere(grain,dnormal);//Dress these germs with sphere
Mat2UI8 lattice = RandomGeometry::continuousToDiscrete(grain);//Convert the continuous model to the lattice
lattice.display();
lattice.save("../doc/image2/boolean.jpg");
Mat2F64 mhisto=  Analysis::histogram(lattice);
std::cout<<"Realization porosity"<<mhisto(0,1)<<std::endl;
\endcode
\image html  boolean.jpg
*/
    namespace Private{
        template<int DIM>
        class IntersectionRectangle
        {
        public:
            VecN<DIM,F64> x;
            VecN<DIM,F64> size;
            bool perhapsIntersection(Germ<DIM> * grain,const VecN<DIM,F64> & domain,MatNBoundaryConditionType boundary_condition=MATN_BOUNDARY_CONDITION_BOUNDED)
            {
                if(boundary_condition==MATN_BOUNDARY_CONDITION_BOUNDED){
                    VecN<DIM,F64> xx =grain->x-x;
                    double width = this->size.norm(0)*0.5+grain->getRadiusBallNorm0IncludingGrain();
                    if( xx.norm(0)< width )
                        return true;
                    else
                        return false;
                }else{
                    double width = this->size.norm(0)*0.5+grain->getRadiusBallNorm0IncludingGrain();
                    VecN<DIM,F64> xx =grain->x-x;
                    if( xx.norm(0)< width )
                        return true;
                    for(unsigned int i=0;i<DIM;i++){
                        xx(i)-=domain(i);
                        if( xx.norm(0)< width )
                            return true;
                        xx(i)+=2*domain(i);
                        if( xx.norm(0)< width )
                            return true;
                        xx(i)-=domain(i);
                    }
                    return false;
                }
            }
        };
        template<int DIM>
        static VecN<DIM,F64> smallestDistanceTore(const VecN<DIM,F64> xtore,const VecN<DIM,F64> xfixed, const VecN<DIM,F64> domain){
            VecN<DIM,F64> x(xtore);
            VecN<DIM,F64> xtemp(xtore);
            double dist = (xtemp-xfixed).normPower(2);
            double distemp;
            for(unsigned int i=0;i<DIM;i++){
                xtemp(i)-=domain(i);
                distemp = (xtemp-xfixed).normPower(2);
                if(distemp<dist){
                    dist=distemp;
                    x = xtemp;
                }
                xtemp(i)+=2*domain(i);
                distemp = (xtemp-xfixed).normPower(2);
                if(distemp<dist){
                    dist=distemp;
                    x = xtemp;
                }
                xtemp(i)-=domain(i);
            }
            return x;
        }
    }

class POP_EXPORTS RandomGeometry
{
private:


public:
    /*!
        \class pop::RandomGeometry
        \ingroup RandomGeometry
        \brief Germ-Grain framework
        \author Tariel Vincent



      \sa ModelGermGrain
    */

    //-------------------------------------
    //
    //! \name Germ
    //@{
    //------------- ------------------------

    /*!
    \fn  static ModelGermGrain<DIM>    poissonPointProcess(VecN<DIM,F64> domain,double lambda);
    * \brief Uniform Poisson point process
    * \param domain domain of definition [0,domain(0)]*[1,domain(1)]...
    * \param lambda intensity of the homogenous field
    * \return point process
    *
    * This method returns a realization of uniform Posssin point process in the given domain and with a given number of VecNs
    *  \code
    Vec2F64 domain(512);//2d field domain
    double lambda= 0.001;// parameter of the Poisson point process

    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process
    Distribution d (1,"DIRAC");//because the Poisson point process has a surface equal to 0, we associate each VecN with mono-disperse sphere to display the result
    RandomGeometry::sphere(grain,d);
    Mat2RGBUI8 img = RandomGeometry::continuousToDiscrete(grain);
    img.display();
    * \endcode
    * \sa ModelGermGrain
    */
    template<int DIM>
    static ModelGermGrain<DIM>    poissonPointProcess(VecN<DIM,F64> domain,double lambda);


    /*!
    * \brief Non-uniform Poisson point process
    * \param lambdafield lambda field
    * \return point process
    *
    * This method returns a realization of non uniform Posssin point process as an union of independant simulation of uniform Poisson point process in each ceil (pixel/voxel)
    * \include PoissonNonUniform.cpp
    * \image html NonUniformLena.png
    * \sa ModelGermGrain
    */
    template<int DIM,typename TYPE>
    static ModelGermGrain<DIM>    poissonPointProcessNonUniform(const MatN<DIM,TYPE> & lambdafield);


    /*!
    \fn  static void  hardCoreFilter( ModelGermGrain<DIM>  & grain, F64 radius,bool btore=true)throw(pexception);
    * \brief Matern filter (Hard Core)
    * \param grain input/output grain
    * \param radius R
    * \param btore boundary condition (true=periodic, false=bounded)
    *
    * The resulting point process is a hard-core point process with minimal inter-VecN distance R. The following code presents an art application
    *  \code
    * Mat2RGBUI8 img;
    * img.load("../image/Lena.bmp");
    * img= GeometricalTransformation::scale(img,Vec2F64(1024./img.getDomain()(0)));
    * double radius=10;
    * Vec2F64 domain;
    * domain= img.getDomain();

    * ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,1);//generate the 2d Poisson point process
    * RandomGeometry::hardCoreFilter(grain,radius*2,false);
    * Distribution d (radius,"DIRAC");//because the Poisson point process has a surface equal to 0, we associate each VecN with mono-disperse sphere to display the result
    * RandomGeometry::sphere(grain,d);
    * RandomGeometry::RGBFromMatrix(grain,img);
    * grain.setModel( DeadLeave);
    * Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain);
    * RGBUI8 mean = Analysis::meanValue(img);
    * std::cout<<mean<<std::endl;
    * Mat2RGBUI8::IteratorEDomain it(img.getIteratorEDomain());
    * while(it.next()){
    *    if(aborigenart(it.x())==RGBUI8(0))
    *         aborigenart(it.x())=mean;
    * }
    * aborigenart.display();
    * \endcode
    * \image html AborigenLena.png
    * \sa ModelGermGrain
    */
    template<int DIM>
    static void  hardCoreFilter( ModelGermGrain<DIM>  & grain, F64 radius,bool btore=true)throw(pexception);

    /*!
    \fn  static void  minOverlapFilter( ModelGermGrain<DIM>  & grain, F64 radius,bool btore=true)throw(pexception);
    * \brief Min-overlap filter
    * \param grain input/output grain
    * \param radius R
    * \param btore boundary condition (true=periodic, false=bounded)
    *
    * The resulting point process is the min-overlap point process where each VecN has at least one neighborhood VecN at distance smaller than R.
    *  \code
    * //Initial field with a local porosity equal to img(x)/255
    * Mat2RGBUI8 img;
    * img.load("../image/Lena.bmp");
    * img= GeometricalTransformation::scale(img,Vec2F64(1024./img.getDomain()(0)));
    * double radius=10;
    * Vec2F64 domain;
    * domain= img.getDomain();

    * ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.1);//generate the 2d Poisson point process
    * RandomGeometry::minOverlapFilter(grain,radius*2,false);
    * Distribution d (radius,"DIRAC");//because the Poisson point process has a surface equal to 0, we associate each VecN with mono-disperse sphere to display the result
    * RandomGeometry::sphere(grain,d);
    * RandomGeometry::RGBFromMatrix(grain,img);
    * grain.setModel( DeadLeave);
    * Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain,false);

    * RGBUI8 mean = Analysis::meanValue(img);
    * Mat2RGBUI8::IteratorEDomain it(img.getIteratorEDomain());
    * while(it.next()){
    *     if(aborigenart(it.x())==RGBUI8(0))
    *         aborigenart(it.x())=mean;
    * }
    * aborigenart.display();
    * \endcode
    * \image html AborigenLenaFloculation.png
    * \sa ModelGermGrain
    */
    template<int DIM>
    static void  minOverlapFilter( ModelGermGrain<DIM>  & grain, F64 radius,bool btore=true)throw(pexception);

    /*!
    \fn static void  intersectionGrainToMask( ModelGermGrain<DIM>  & grain, const MatN<DIM,UI8> & img)throw(pexception);
    * \brief Keep the VecN if intersection
    * \param grain input/output grain
    * \param img input binary matrix
    *
    * The resulting point process is the intersectionpoint process where each VecN has a center x such that img(x) is not equal to zero
    *  \code
    * //Initial field with a local porosity equal to img(x)/255
    * Mat2RGBUI8 img;
    * img.load("../image/Lena.bmp");
    * img= GeometricalTransformation::scale(img,Vec2F64(1024./img.getDomain()(0)));
    * double radius=15;
    * Vec2F64 domain;
    * domain= img.getDomain();
    * ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.001);//generate the 2d Poisson point process
    * img = pop::Processing::gradientMagnitudeDeriche(img,0.5);
    * Mat2UI8 threshold = pop::Processing::threshold(img,10);
    * RandomGeometry::intersectionGrainToMask(grain,threshold);
    * Distribution d (radius,"DIRAC");//because the Poisson point process has a surface equal to 0, we associate each VecN with mono-disperse sphere to display the result
    * RandomGeometry::sphere(grain,d);
    * RandomGeometry::RGBFromMatrix(grain,img);
    * grain.setModel( DeadLeave);
    * Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain,false);

    * aborigenart.display();
    * \endcode
    * \image html AborigenLenaGradient.png
    */
    template<int DIM>
    static void  intersectionGrainToMask( ModelGermGrain<DIM>  & grain, const MatN<DIM,UI8> & img)throw(pexception);
    //@}


    //-------------------------------------
    //
    //! \name Grain
    //@{
    //-------------------------------------

    /*!
    \fn   static void sphere( ModelGermGrain<DIM> &  grain,Distribution &dist);
    * \brief dress the VecNs with sphere with a random radius following the probability distribution dist
    * \param grain input/output grain
    * \param dist radius probability distribution
    *
    * dress the VecNs with sphere with a random radius following the probability distribution dist
    *
    * \code
    * double porosity=0.25;
    *  std::string power_law="1/x^(3.1)";
    * Distribution dpower(power_law.c_str());//Poisson generator
    * dpower = pop::Statistics::toProbabilityDistribution(dpower,10,1024,0.1);

    * double moment_order_2 = pop::Statistics::moment(dpower,2,0,1024);
    * double surface_expectation = moment_order_2*3.14159265;
    * Vec2F64 domain(4048);//2d field domain
    * double lambda=-std::log(porosity)/std::log(2.718)/surface_expectation;
    * ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process
    * RandomGeometry::sphere(grain,dpower);
    * Mat2RGBUI8 img_VecN = RandomGeometry::continuousToDiscrete(grain);
    * Mat2UI8 img_VecN_grey;
    * img_VecN_grey = img_VecN;
    * Mat2F64 m=  Analysis::histogram(img_VecN_grey);
    * std::cout<<"Realization porosity"<<m(0,1)<<std::endl;
    * Mat2UI32 img_label = pop::Processing::clusterToLabel(img_VecN_grey,0);
    * img_VecN = pop::Visualization::labelToRandomRGB(img_label)  ;
    * img_VecN.scale(512,512);
    * \endcode
    * \image html PowerLawCluster.png
    */
    template<int DIM>
    static void sphere( ModelGermGrain<DIM> &  grain,Distribution &dist);


    /*!
    \fn   static void box( ModelGermGrain<DIM> &  grain,const  DistributionMultiVariate & distradius,const DistributionMultiVariate& distangle )throw(pexception);
    * \brief dress the VecNs with box with a random radius and random distribution
    * \param grain input/output grain
    * \param distradius multivariate probability distribution for radius
    * \param distangle multivariate probability distribution for radius
    *
    * Dress the VecNs with box with a random radius following the probability distribution dist
    *
    * \code
    Mat2RGBUI8 img;
    img.load("../image/Lena.bmp");
    Vec2F64 domain;
    domain= img.getDomain();
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.1);//generate the 2d Poisson point process


    Distribution d("1/x^(3.1)");
    Distribution dproba = pop::Statistics::toProbabilityDistribution(d,5,128);
    DistributionMultiVariate d_radius(dproba,dproba);
    DistributionMultiVariate d_angle(Distribution(0,3.14159265*2,"UNIFORMREAL"));


    RandomGeometry::box(grain,d_radius,d_angle);
    RandomGeometry::RGBFromMatrix(grain,img);
    grain.setModel( DeadLeave);
    Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain,false);

    aborigenart.display();
    * \endcode
    * \image html AborigenLenaBox.png
    */
    template<int DIM>
    static void box( ModelGermGrain<DIM> &  grain,const  DistributionMultiVariate & distradius,const DistributionMultiVariate& distangle )throw(pexception);

    /*!
    \fn   static void polyhedra( ModelGermGrain<DIM> &  grain,Vec<Distribution>& distradius,Vec<Distribution>& distnormal,Vec<Distribution>& distangle )throw(pexception);
    * \brief dress the VecNs with polyhedra
    * \param grain input/output grain
    * \param distradius Vec of radius probability distribution
    * \param distnormal Vec of normal probability distribution
    * \param distangle Vec of angle probability distribution
    *
    * Dress the VecNs with polyhedra with a random radius/nomals following the probability distributions. In the following code, we dress with  regular tetrahedron:
    *
    * \code
    double porosity=0.8;
    double radius=7;
    Distribution ddirac_radius(radius,"DIRAC");

    double moment_order3 = pop::Statistics::moment(ddirac_radius,3,0,40);
    //std::sqrt(2)/12*(std::sqrt(24))^3
    double volume_expectation = moment_order3*std::sqrt(2.)/12.*std::pow(std::sqrt(24.),3.);
    Vec3F64 domain(128);//2d field domain
    double lambda=-std::log(porosity)/std::log(2.718)/volume_expectation;
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process

    Vec<Distribution > v_radius;
    v_radius.push_back(ddirac_radius);
    v_radius.push_back(ddirac_radius);
    v_radius.push_back(ddirac_radius);
    v_radius.push_back(ddirac_radius);

    Distribution dplus(1/std::sqrt(3.),"DIRAC");
    Distribution dminus(-1/std::sqrt(3.) ,"DIRAC");
    Vec<Distribution> v_normal;
    v_normal.push_back(dplus);
    v_normal.push_back(dplus);
    v_normal.push_back(dplus);

    v_normal.push_back(dminus);
    v_normal.push_back(dminus);
    v_normal.push_back(dplus);

    v_normal.push_back(dminus);
    v_normal.push_back(dplus);
    v_normal.push_back(dminus);

    v_normal.push_back(dplus);
    v_normal.push_back(dminus);
    v_normal.push_back(dminus);


    Vec<Distribution> v_angle;
    v_angle.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    v_angle.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    v_angle.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));

    RandomGeometry::polyhedra(grain,v_radius,v_normal,v_angle);
    Mat3RGBUI8 img_VecN = RandomGeometry::continuousToDiscrete(grain);
    Mat3UI8 img_VecN_grey;
    img_VecN_grey = img_VecN;

    Mat2F64 m=  Analysis::histogram(img_VecN_grey);
    std::cout<<"Realization porosity"<<m(0,1)<<std::endl;

    img_VecN_grey = pop::Processing::greylevelRemoveEmptyValue(img_VecN_grey);
    Mat3F64 phasefield = PDE::allenCahn(img_VecN_grey,20);
    phasefield = PDE::getField(img_VecN_grey,phasefield,1,3);
    Scene3d scene;
    pop::Visualization::marchingCubeLevelSet(scene,phasefield);
    pop::Visualization::lineCube(scene,img_VecN);
    scene.display();
    * \endcode
    * \image html tetrahedron.png
    */
    template<int DIM>
    static void polyhedra( ModelGermGrain<DIM> &  grain,Vec<Distribution>& distradius,Vec<Distribution>& distnormal,Vec<Distribution>& distangle )throw(pexception);

    /*!
    \fn   static void ellipsoid( ModelGermGrain<DIM> &  grain,Vec<Distribution>& distradius,Vec<Distribution>& distangle )throw(pexception);
    * \brief dress the VecNs with polyhedra
    * \param grain input/output grain
    * \param distradius Vec of radius probability distribution
    * \param distangle Vec of angle probability distribution
    *
    * Dress the VecNs with ellipsoids with a  random radius following the probability distributions.
    *
    * \code
    double porosity=0.8;
    double radiusmin=5;
    double radiusmax=30;
    Distribution duniform_radius(radiusmin,radiusmax,"UNIFORMREAL");

    double moment_order3 = pop::Statistics::moment(duniform_radius,3,0,40);
    //4/3*pi*E^3(R)
    double volume_expectation = moment_order3*4/3.*3.14159265;
    Vec3F64 domain(200);//2d field domain
    double lambda=-std::log(porosity)/std::log(2.718)/volume_expectation;

    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process

    DistributionMultiVariate radius(DistributionMultiVariate(duniform_radius,duniform_radius),duniform_radius );

    DistributionMultiVariate angle(DistributionMultiVariate(Distribution(0,3.14159265,"UNIFORMREAL"),Distribution(0,3.14159265,"UNIFORMREAL")),Distribution(0,3.14159265,"UNIFORMREAL") );

    RandomGeometry::ellipsoid(grain,radius,angle);
    Mat3RGBUI8 img_VecN = RandomGeometry::continuousToDiscrete(grain);
    Mat3UI8 img_VecN_grey;
    img_VecN_grey = img_VecN;

    Mat2F64 m=  Analysis::histogram(img_VecN_grey);
    std::cout<<"Realization porosity"<<m(0,1)<<std::endl;

    img_VecN_grey = pop::Processing::greylevelRemoveEmptyValue(img_VecN_grey);
    Mat3F64 phasefield = PDE::allenCahn(img_VecN_grey,10);
    phasefield = PDE::getField(img_VecN_grey,phasefield,1,3);
    Scene3d scene;
    pop::Visualization::marchingCubeLevelSet(scene,phasefield);
    pop::Visualization::lineCube(scene,img_VecN);
    scene.display();
    * \endcode
    * \image html ellipsoid.png
    */
    template<int DIM>
    static void ellipsoid( ModelGermGrain<DIM> &  grain,const DistributionMultiVariate& distradius,const DistributionMultiVariate& distangle)throw(pexception);

    /*!
    \fn   static static void  rhombohedron( ModelGermGrain3 & grain,Distribution distradius,Distribution distangle, Vec<Distribution>& distorientation )throw(pexception);
    * \brief dress the VecNs with polyhedra
    * \param grain input/output grain
    * \param distradius  radius probability distribution
    * \param distangle  angle probability distribution between plane
    * \param distorientation orientation probability distribution
    *
    * Dress the VecNs with rhombohedron with a random radius following the probability distributions.
    *
    * \code
    double porosity=0.8;
    double radiusmin=5;
    double radiusmax=30;
    Distribution duniform_radius(radiusmin,radiusmax,"UNIFORMREAL");

    double angle=10*3.14159265/180;//10 degre
    Distribution ddirar_angle(angle,"DIRAC");

    double moment_order3 = pop::Statistics::moment(duniform_radius,3,0,40);
    //8*E^3(R)/E^3(std::cos(theta))
    double volume_expectation = moment_order3*8./(std::pow(std::cos(angle),3.0));
    Vec3F64 domain(256);//2d field domain
    double lambda=-std::log(porosity)/std::log(2.718)/volume_expectation;
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process

    DistributionMultiVariate v_orientation (DistributionMultiVariate(Distribution(0,3.14159265,"UNIFORMREAL"),Distribution(0,3.14159265,"UNIFORMREAL")) ,Distribution(0,3.14159265,"UNIFORMREAL"));
    RandomGeometry::rhombohedron(grain,duniform_radius,ddirar_angle,v_orientation);
    Mat3RGBUI8 img_VecN = RandomGeometry::continuousToDiscrete(grain);
    Mat3UI8 img_VecN_grey;
    img_VecN_grey = img_VecN;

    Mat2F64 m=  Analysis::histogram(img_VecN_grey);
    std::cout<<"Realization porosity"<<m(0,1)<<std::endl;

    img_VecN_grey = pop::Processing::greylevelRemoveEmptyValue(img_VecN_grey);
    Mat3F64 phasefield = PDE::allenCahn(img_VecN_grey,15);
    phasefield = PDE::getField(img_VecN_grey,phasefield,1,3);
    Scene3d scene;
    pop::Visualization::marchingCubeLevelSet(scene,phasefield);
    pop::Visualization::lineCube(scene,img_VecN);
    scene.display();
    * \endcode
    * \image html rhombohedron.png
    */
    static void  rhombohedron( pop::ModelGermGrain3 & grain,const Distribution& distradius,const Distribution& distangle, const DistributionMultiVariate& distorientation )throw(pexception);

    /*!
    \fn   static void  cylinder( ModelGermGrain3  & grain,Distribution  distradius,Distribution  distheight, Vec<Distribution>& distorientation )throw(pexception);
    * \brief dress the VecNs with cylinder
    * \param grain input/output grain
    * \param distradius  radius probability distribution
    * \param distheight  height probability distribution
    * \param distorientation orientation probability distribution
    *
    * Dress the VecNs with cylinder with a random radius/height following the probability distributions.
    *
    * \code
    double porosity=0.8;
    double radius=5;
    Distribution ddirac_radius(radius,"DIRAC");

    double heightmix=40;
    double heightmax=70;
    Distribution duniform_height(heightmix,heightmax,"UNIFORMREAL");

    double moment_order2 = pop::Statistics::moment(ddirac_radius,2,0,40);
    double moment_order1 = pop::Statistics::moment(duniform_height,1,0,100);
    //8*E^2(R)/E^3(std::cos(theta))
    double volume_expectation = 3.14159265*moment_order2*moment_order1;
    Vec3F64 domain(256);//2d field domain
    double lambda=-std::log(porosity)/std::log(2.718)/volume_expectation;
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process

    Vec<Distribution> v_orientation;
    v_orientation.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    v_orientation.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    v_orientation.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    RandomGeometry::cylinder(grain,ddirac_radius,duniform_height,v_orientation);
    Mat3RGBUI8 img_VecN = RandomGeometry::continuousToDiscrete(grain);
    Mat3UI8 img_VecN_grey;
    img_VecN_grey = img_VecN;

    Mat2F64 m=  Analysis::histogram(img_VecN_grey);
    std::cout<<"Realization porosity"<<m(0,1)<<std::endl;

    img_VecN_grey = pop::Processing::greylevelRemoveEmptyValue(img_VecN_grey);
    Mat3F64 phasefield = PDE::allenCahn(img_VecN_grey,15);
    phasefield = PDE::getField(img_VecN_grey,phasefield,1,3);
    Scene3d scene;
    pop::Visualization::marchingCubeLevelSet(scene,phasefield);
    pop::Visualization::lineCube(scene,img_VecN);
    scene.display();
    * \endcode
    * \image html cylinder.png
    */
    static void  cylinder( pop::ModelGermGrain3  & grain,Distribution  distradius,Distribution  distheight,const DistributionMultiVariate & distorientation )throw(pexception);
    template<int DIM>
    static void matrixBinary( ModelGermGrain<DIM> &  grain,const  MatN<DIM,UI8> &img,const DistributionMultiVariate& distangle )throw(pexception);

    /*!
    \fn static void   addition(const ModelGermGrain<DIM> &  grain1, ModelGermGrain<DIM>& grain2);
    * \brief addition of the grains
    * \param grain1 input/output grain
    * \param grain2  radius probability distribution
    *
    * addition of the grains
    *
    * \code
    double radius=10;
    Distribution ddirac_radius(radius,"DIRAC");

    double heightmix=40;
    double heightmax=70;
    Distribution duniform_height(heightmix,heightmax,"UNIFORMREAL");

    Vec3F64 domain(256);//2d field domain

    ModelGermGrain3 grain1 = RandomGeometry::poissonPointProcess(domain,0.01);//generate the 2d Poisson point process

    Vec<Distribution> v_orientation;
    v_orientation.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    v_orientation.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    v_orientation.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    RandomGeometry::cylinder(grain1,ddirac_radius,duniform_height,v_orientation);


    ModelGermGrain3 grain2 = RandomGeometry::poissonPointProcess(domain,0.01);//generate the 2d Poisson point process
    Vec<Distribution> v_radius_ellipsoid;
    v_radius_ellipsoid.push_back(Distribution (5,25,"UNIFORMREAL"));
    v_radius_ellipsoid.push_back(Distribution (5,25,"UNIFORMREAL"));
    v_radius_ellipsoid.push_back(Distribution (5,25,"UNIFORMREAL"));
    RandomGeometry::ellipsoid(grain2,v_radius_ellipsoid,v_orientation);

    RandomGeometry::addition(grain1,grain2);

    Mat3RGBUI8 img_VecN = RandomGeometry::continuousToDiscrete(grain2);
    Mat3UI8 img_VecN_grey;
    img_VecN_grey = img_VecN;
    img_VecN_grey = pop::Processing::greylevelRemoveEmptyValue(img_VecN_grey);
    Mat3F64 phasefield = PDE::allenCahn(img_VecN_grey,15);
    phasefield = PDE::getField(img_VecN_grey,phasefield,1,3);
    Scene3d scene;
    pop::Visualization::marchingCubeLevelSet(scene,phasefield);
    pop::Visualization::lineCube(scene,img_VecN);
    scene.display();
    * \endcode
    * \image html additiongrain.png
    */
    template<int DIM>
    static void   addition(const ModelGermGrain<DIM> &  grain1, ModelGermGrain<DIM>& grain2);
    //@}
    //-------------------------------------
    //
    //! \name RGB
    //@{
    //-------------------------------------
    /*!
    \fn static void RGBRandomBlackOrWhite( ModelGermGrain<DIM> &  grain);
    * \brief affect a black and white RGB randomly
    * \param grain input/output grain
    *
    * affect a black and white RGB randomly to the grains
    *
    * \code
    Vec2F64 domain(512,512);
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.001);//generate the 2d Poisson point process

    Distribution d("1/x^3");
    Distribution dproba = pop::Statistics::toProbabilityDistribution(d,5,128);
    Vec<Distribution> v_radius;
    v_radius.push_back(dproba);
    v_radius.push_back(dproba);
    Vec<Distribution> v_angle;
    v_angle.push_back(Distribution(0,3.14159265*2,"UNIFORMREAL"));

    RandomGeometry::box(grain,v_radius,v_angle);
    RandomGeometry::RGBRandomBlackOrWhite(grain);
    grain.setModel( DeadLeave);
    Mat2RGBUI8 deadleave_blackwhite = RandomGeometry::continuousToDiscrete(grain,false);
    deadleave_blackwhite.display();
    * \endcode
    * \image html deadleave_blackwhite.png
    */
    template<int DIM>
    static void  RGBRandomBlackOrWhite( ModelGermGrain<DIM> &  grain);

    /*!
    \fn     static void  RGBRandom( ModelGermGrain<DIM> &  grain,DistributionMultiVariate distRGB_RGB)throw(pexception);
    * \brief affect a RGB randomly
    * \param grain input/output grain
    * \param distRGB_RGB multi variate probability distribution
    *
    * affect a RGB randomly following the input distributions
    *
    * \code
    Vec2F64 domain(512,512);
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.001);//generate the 2d Poisson point process

    Distribution d("1/x^3");
    Distribution dproba = pop::Statistics::toProbabilityDistribution(d,5,128);
    Vec<Distribution> v_radius;
    v_radius.push_back(dproba);
    v_radius.push_back(dproba);
    Vec<Distribution> v_angle;
    v_angle.push_back(Distribution(0,3.14159265*2,"UNIFORMREAL"));

    RandomGeometry::box(grain,v_radius,v_angle);
    DistributionMultiVariate dmulti(DistributionUniformInt(0,255));
    DistributionMultiVariate dcoupled(dmulti,3);
    RandomGeometry::RGBRandom(grain,dcoupled);
    grain.setModel( DeadLeave);
    Mat2RGBUI8 deadleave_blackwhite = RandomGeometry::continuousToDiscrete(grain,false);
    deadleave_blackwhite.display();
    * \endcode
    * \image html deadleave_random.png
    */
    template<int DIM>
    static void  RGBRandom( ModelGermGrain<DIM> &  grain,DistributionMultiVariate distRGB_RGB)throw(pexception);

    /*!
    \fn  static void  RGBFromMatrix( ModelGermGrain<DIM> &  grain,const MatN<DIM,RGBUI8 > & in);
    * \brief dress the grains with a RGB coming from the input image
    * \param grain input/output grain
    * \param in input matrix
    *
    * Dress the grains with a RGB coming from the input matrix such that the grain RGB at the position x as the RGB img(x)
    *
    * \code
    Mat2RGBUI8 img;
    img.load("../image/Lena.bmp");
    Vec2F64 domain;
    domain= img.getDomain();
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.01);//generate the 2d Poisson point process
    Distribution d("1/x^(3.1)");
    Distribution dproba = pop::Statistics::toProbabilityDistribution(d,5,128);
    Vec<Distribution> v_radius;
    v_radius.push_back(dproba);
    v_radius.push_back(dproba);
    Vec<Distribution> v_angle;
    v_angle.push_back(Distribution(0,3.14159265*2,"UNIFORMREAL"));

    RandomGeometry::box(grain,v_radius,v_angle);
    RandomGeometry::RGBFromMatrix(grain,img);
    grain.setModel( DeadLeave);
    Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain,false);

    aborigenart.display();
    * \endcode
    * \image html AborigenLenaBox.png
    */
    template<int DIM>
    static void  RGBFromMatrix( ModelGermGrain<DIM> &  grain,const MatN<DIM,RGBUI8 > & in);

    /*!
    \fn static void  randomWalk( ModelGermGrain<DIM> &  grain, F64 radius)throw(pexception);
    * \brief move the grains with a random walk
    * \param grain input/output grain
    * \param radius standard deviation
    *
    * move each grain with\n
      - a random radius following a centered normal distribution with a standard deviation=radius
      - a random direction
    *
    * \code
    Mat2RGBUI8 img;
    img.load("../image/Lena.bmp");
    Vec2F64 domain;
    domain= img.getDomain();
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.1);//generate the 2d Poisson point process
    Distribution d("1/x^3");
    Distribution dproba = pop::Statistics::toProbabilityDistribution(d,5,128);

    RandomGeometry::sphere(grain,dproba);
    RandomGeometry::RGBFromMatrix(grain,img);
    grain.setModel( DeadLeave);

    int i=0;
    MatNDisplay windows;
    do{
        Mat2RGBUI8 art = RandomGeometry::continuousToDiscrete(grain,false);
        windows.display(art);
        RandomGeometry::randomWalk(grain,4);
        art.save(std::string("art"+BasicUtility::IntFixedDigit2String(i,4)+".bmp").c_str());
        i++;
    }while(!windows.is_closed());
    * \endcode
    * \image html art.gif
    */
    template<int DIM>
    static void  randomWalk( ModelGermGrain<DIM> &  grain, F64 radius)throw(pexception);

    //@}
    //-------------------------------------
    //
    //! \name Model
    //@{
    //-------------------------------------

    //@}
    //-------------------------------------
    //
    //! \name Continuous to Lattice
    //@{
    //-------------------------------------
    /*!
    * \brief map all germs in a labeled matrix
    * \param germ input germ
    * \return labeled matrix
    *
    *
    */
    template<int DIM>
    static MatN<DIM,UI32> germToMatrix(const ModelGermGrain<DIM> &germ)
    {
        MatN<DIM,UI32> map(germ.getDomain());
        for(int i =0; i<(int)germ.grains().size();i++)
        {
            VecN<DIM,int> x = germ.grains()[i]->x;
            map(x)=i+1;
        }
        return map;
    }


    /*!
    \fn static MatN<DIM,pop::RGBUI8> continuousToDiscrete(const ModelGermGrain<DIM> &grain,bool btore=true);
    * \brief transform the continuous model to the lattice model
    * \param grain input grain
    * \return lattice model
    *
    * transform the continuous model to the lattice model such that the domain is equal to the domain of the grain
    *
    */
    template<int DIM>
    static pop::MatN<DIM,pop::RGBUI8>  continuousToDiscrete(const ModelGermGrain<DIM> &grain);
    //@}
    //-------------------------------------
    //
    //! \name Others
    //@{
    //------------- ------------------------


    /*!
    \fn static MatN<DIM,UI8> gaussianThesholdedRandomField(const Mat2F64 &mcorre,const VecN<DIM,int> &domain);
    * \brief Diffusion limited aggregation
    * \param mcorre correlation function
    * \param domain domain of the model
    * \param gaussianfield output gaussian field
    * \return model
    *
    *   Diffusion-limited aggregation process
    *
    */
    template<int DIM>
    static MatN<DIM,UI8> gaussianThesholdedRandomField(const Mat2F64 &mcorre,const VecN<DIM,int> &domain,MatN<DIM,F64> & gaussianfield);



    /*!
    \fn static MatN<DIM,UI8> randomStructure(const VecN<DIM,int> &domainmodel,const Mat2F64 & volume_fraction);
    * \brief generate a random structure at the given volume fraction
    * \param domainmodel domain of the model
    * \param volume_fraction volume fraction
    * \return model
    *
    *
    *
    */
    template<int DIM>
    static MatN<DIM,UI8> randomStructure(const VecN<DIM,I32> &domainmodel,const Mat2F64 & volume_fraction);


    /*!


    * \brief annealing simulated method for 3d reconstruction based on correlation function
    * \param model model matrix (input t=0, output=end)
    * \param img_reference reference matrix
    * \param lengthcorrelation number of walkers
    * \param nbr_permutation_by_pixel number of permutations by pixel/voxel
    * \param temperature_inverse initial inverse temperature
    * \return model
    *
    *  see chapter 5 of my Pdd thesis http://tel.archives-ouvertes.fr/tel-00516939/
    * \code
    Mat2UI8 ref;
    ref.load("/home/vincent/Desktop/CI_4_0205_18_00_seg.pgm");
        ref= Processing::greylevelRemoveEmptyValue(ref);
        Mat2F64 m = Analysis::histogram(ref);
        VecN<3,int> p(256,256,256);
        Mat3UI8 model =RandomGeometry::randomStructure(p,m);
        RandomGeometry::annealingSimutated(model, ref,128,8 );//long algorithm at least 15 hours
        Visualization::labelToRandomRGB(model).display();
    * \endcode
    *
    *
    */

    template<int DIM1,int DIM2>
    static void annealingSimutated(MatN<DIM1,UI8> & model,const MatN<DIM2,UI8> & img_reference, F64 nbr_permutation_by_pixel=8,int lengthcorrelation=-1,double temperature_inverse=500);


    /*!
    \fn MatN<2,UI8 > diffusionLimitedAggregation2D(int size,int nbrwalkers);
    * \brief Diffusion limited aggregation
    * \param size size of the lattice
    * \param nbrwalkers number of walkers
    * \return DLA
    *
    *   Diffusion-limited aggregation process
    *
    */
    static MatN<2,UI8 > diffusionLimitedAggregation2D(int size=256,int nbrwalkers=10000);

    /*!
    \fn static MatN<3,UI8 > diffusionLimitedAggregation3D(int size,int nbrwalkers);
    * \brief Diffusion limited aggregation
    * \param size size of the lattice
    * \param nbrwalkers number of walkers
    * \return DLA
    *
    *   Diffusion-limited aggregation process in 3d
    *
    */
    static MatN<3,UI8 > diffusionLimitedAggregation3D(int size,int nbrwalkers);
    //@}
private:
    template <int DIM>
            static void divideTilling(const Vec<int> & v,Private::IntersectionRectangle<DIM> &rec,const  ModelGermGrain<DIM> & grainlist, MatN<DIM,RGBUI8> &img);
    static Distribution generateProbabilitySpectralDensity(const Mat2F64& correlation,double beta);

    inline static bool annealingSimutatedLaw(double energynext,double energycurrent,double temperature_inverse){
        if(energynext<energycurrent)
            return true;
        else
        {
            double v1 = rand()*1.0/RAND_MAX;
            double v2 = std::exp((energycurrent-energynext)*temperature_inverse);
            if(v1<v2)
                return true;
            else
                return false;
        }
    }

    template<int DIM>
    static Vec< Vec<Vec<int> > >  autoCorrelation( const MatN<DIM,UI8> & img , int taille, int nbr_phase,bool boundary =true)
    {
        Vec< Vec<Vec<int> > > autocorrhit( nbr_phase,Vec< Vec<int> >(nbr_phase,  Vec<int>(taille+1)));
        typename MatN<DIM,UI8>::E x,y;
        int etat1,etat2,lenght;
        typename MatN<DIM,UI8>::IteratorEDomain b(img.getIteratorEDomain());
        while(b.next())
        {
            x = b.x();
            etat1 = img(x);
            for(int i=0;i<DIM;i++)
            {
                y=x;
                for(int k=x[i];k<=x[i]+taille;k++)
                {
                    y[i]=k;
                    lenght = absolute(y[i]-x[i]);
                    if(boundary==true){
                        MatNBoundaryConditionPeriodic::apply(img.getDomain(),y);
                        etat2 = img(y);
                        autocorrhit[etat1][etat2][lenght]++;
                    }else{
                        if(MatNBoundaryConditionBounded::isValid(img.getDomain(),y)){
                            MatNBoundaryConditionBounded::apply(img.getDomain(),y);
                            etat2 = img(y);
                            autocorrhit[etat1][etat2][lenght]++;
                        }
                    }
                }
            }

        }
        return autocorrhit;
    }
    template<int DIM>
    static Vec< Vec<Vec<int> > >   autoCorrelationCross( const MatN<DIM,UI8> & img , int taille, int nbr_phase)
    {
        Vec< Vec<Vec<int> > >  autocorrhit ( nbr_phase,Vec< Vec<int> >(nbr_phase,  Vec<int>(taille+1)));
        typename MatN<DIM,UI8>::E x,y;
        int etat1,etat2,lenght;
        typename MatN<DIM,UI8>::IteratorEDomain b(img.getIteratorEDomain());
        while(b.next())
        {
            x = b.x();
            etat1 = img(x);
            for(int i=0;i<DIM;i++)
            {
                for(int j=i+1;j<DIM;j++){
                    for( int m=0;m<=1;m++){
                        y=x;
                        for(int k=0;k<=taille;k++)
                        {

                            y[i]=x[i]+k;
                            if(m==0)
                                y[j]=x[j]+k;
                            else
                                y[j]=x[j]-k;
                            lenght = k;
                            MatNBoundaryConditionPeriodic::apply(img.getDomain(),y);
                            etat2 = img(y);
                            autocorrhit[etat1][etat2][lenght]++;
                        }
                    }

                }
            }
        }
        return autocorrhit;

    }
    template<int DIM>
    static void   switch_state_pixel_cross(Vec<Vec< Vec<int> > >& v_cor, const MatN<DIM,UI8>& img , int taille,const typename MatN<DIM,UI8>::E & x  , int new_state)
    {

        int lenght=0;
        typename MatN<DIM,UI8>::E y;
        int old_state = (int)img(x);
        for(int i=0;i<DIM;i++)
        {
            for(int j=i+1;j<DIM;j++){
                for( int m=0;m<=1;m++){
                    y=x;
                    v_cor[old_state][old_state][0]--;
                    v_cor[new_state][new_state][0]++;

                    for(int k = 1;k<=taille;k++)
                    {
                        y[i]=x[i]+k;
                        if(m==0)
                            y[j]=x[j]+k;
                        else
                            y[j]=x[j]-k;
                        lenght = absolute(k);
                        MatNBoundaryConditionPeriodic::apply(img.getDomain(),y);
                        int etat2 = img(y);
                        v_cor[old_state][etat2][lenght]--;
                        v_cor[new_state][etat2][lenght]++;
                    }
                    for(int k = -taille;k<0;k++)
                    {
                        y[i]=x[i]+k;
                        if(m==0)
                            y[j]=x[j]+k;
                        else
                            y[j]=x[j]-k;
                        lenght = absolute(k);
                        MatNBoundaryConditionPeriodic::apply(img.getDomain(),y);
                        int etat2 = img(y);
                        v_cor[etat2][old_state][lenght]--;
                        v_cor[etat2][new_state][lenght]++;
                    }
                }
            }
        }

    }
    template<int DIM>
    static void   switch_state_pixel(Vec<Vec< Vec<int> > >& v_cor, const MatN<DIM,UI8>& img , int taille,const typename MatN<DIM,UI8>::E & x  , int new_state)
    {

        int lenght=0;
        typename MatN<DIM,UI8>::E y;
        int old_state = (int)img(x);
        for(int i=0;i<DIM;i++)
        {
            y=x;
            v_cor[old_state][old_state][0]--;
            v_cor[new_state][new_state][0]++;

            for(int k = x[i]+1;k<=x[i]+taille;k++)
            {
                y[i]=k;
                lenght = absolute(y[i]-x[i]);
                MatNBoundaryConditionPeriodic::apply(img.getDomain(),y);
                int etat2 = img(y);
                v_cor[old_state][etat2][lenght]--;
                v_cor[new_state][etat2][lenght]++;
            }
            for(int k = x[i]-taille;k<x[i];k++)
            {
                y[i]=k;
                lenght = absolute(y[i]-x[i]);
                MatNBoundaryConditionPeriodic::apply(img.getDomain(),y);
                int etat2 = img(y);
                v_cor[etat2][old_state][lenght]--;
                v_cor[etat2][new_state][lenght]++;
            }
        }


    }

    static double energy(Vec< Vec<int> >& v1,Vec< Vec<int> >& v2,Vec<int> &nbr_pixel_by_phase1,Vec<int> &nbr_pixel_by_phase2)
    {
        double sum=0;
        for(int i =0;i<(int)v1.size();i++){
            for(int j =0;j<(int)v1[i].size();j++){
                double diff = 1.0*v1[i][j]/nbr_pixel_by_phase1[i]-1.0*v2[i][j]/nbr_pixel_by_phase2[i];
                sum+=(diff*diff);
            }
        }
        return std::sqrt(sum);
    }
    static double energy(Vec< Vec< Vec<int> > >& v1,Vec< Vec< Vec<int> > >& v2)
    {
        double sum=0;
        for(int i =0;i<(int)v1.size();i++){
            for(int j =0;j<(int)v1[i].size();j++){
                for(int k =0;k<(int)v1[i][j].size();k++){

                    //                    sum+=absolute(1.0*v1[i][i][k]/v1[i][i][0]-1.0*v2[i][i][k]/v2[i][i][0]);
                    double diff = 1.0*v1[i][j][k]/v1[i][i][0]-1.0*v2[i][j][k]/v2[i][i][0];
                    //                    if(i==j)
                    sum+=(diff*diff);
                    //                    else
                    //                        sum+=(diff*diff/10);
                }
            }
        }
        return std::sqrt(sum);
    }




};


template<int DIM>
void  RandomGeometry::RGBRandomBlackOrWhite( ModelGermGrain<DIM> &  grain)
{
    DistributionUniformInt rint(0,1);
    for(typename Vec<Germ<DIM> * >::iterator it= grain.grains().begin();it!=grain.grains().end();it++ ){
        int value=rint.randomVariable();
        (*it)->color= RGBUI8(255*value);
    }
}

template<int DIM>
void  RandomGeometry::RGBRandom( ModelGermGrain<DIM> &  grain,DistributionMultiVariate distRGB_RGB)throw(pexception)
{

    if(distRGB_RGB.getNbrVariable()!=3){
        throw(pexception("In RandomGeometry::RGBRandom, the distRGB_RGB distribution must generate a 3d multuvate random variable"));
    }
    for(typename Vec<Germ<DIM> * >::iterator it= grain.grains().begin();it!=grain.grains().end();it++ ){
        VecF64 v = distRGB_RGB.randomVariable();
        (*it)->color=RGBUI8(v(0),v(1),v(2));
    }
}
template<int DIM>
void RandomGeometry::RGBFromMatrix( ModelGermGrain<DIM> &  grain,const MatN<DIM,RGBUI8 > & in){

    for(typename Vec<Germ<DIM> * >::iterator it= grain.grains().begin();it!=grain.grains().end();it++ ){
        (*it)->color= in( (*it)->x);
    }
}

template<int DIM>
void  RandomGeometry::hardCoreFilter( ModelGermGrain<DIM> &  grain, F64 radius,bool btore)throw(pexception)
{
    KDTree<DIM,F64> kptree;
    Vec<bool> v_in;

    for(int i =0; i<(int)grain.grains().size();i++){
        VecN<DIM,F64> x = grain.grains()[i]->x;
        VecN<DIM,F64> xfind;
        double dist;
        kptree.search(x,xfind,dist);
        if(dist>radius){
            v_in.push_back(true);
            kptree.addItem(x);
            if(btore==true){
                for(int i=0;i<DIM;i++){
                    VecN<DIM,F64> xtore(x);
                    xtore(i)-=grain.getDomain()(i);
                    kptree.addItem(xtore);
                    xtore(i)+=(2*grain.getDomain()(i));
                    kptree.addItem(xtore);
                }
            }
        }
        else{
            v_in.push_back(false);
        }
    }

    Vec<Germ<DIM> * > vlist_temp;
    for(unsigned int i =0; i<v_in.size();i++){
        if(v_in[i]==true){
            vlist_temp.push_back(grain.grains()[i]);
        }else{
            delete grain.grains()[i];
        }
    }
    grain.grains() = vlist_temp;
}
template<int DIM>
void RandomGeometry::randomWalk( ModelGermGrain<DIM> &  grain, F64 radius)throw(pexception)
{
    Vec<F64> v;
    v= grain.getDomain();
    DistributionNormal d(0,radius);
    for(int i =0; i<(int)grain.grains().size();i++)
    {
        Germ<DIM> * g =grain.grains()[i];
        for(int i=0;i<DIM;i++){
            g->x(i)+=d.randomVariable();
            if(g->x(i)<0){
                g->x(i)=-g->x(i);
            }
            if(g->x(i)<0)
                g->x(i)=-g->x(i);
            if(g->x(i)>v(i)-1    )
                g->x(i)= v(i)-1 - (g->x(i) - (v(i)-1));
        }
    }
}

template<int DIM>
void  RandomGeometry::minOverlapFilter(ModelGermGrain<DIM> &  grain, F64 radius,bool btore)throw(pexception)
{

    Vec<VecN<DIM,F64> > v_x;

    for(int i =0; i<(int)grain.grains().size();i++){
        VecN<DIM,F64> x = grain.grains()[i]->x;
        v_x.push_back(x);
        if(btore==true){
            for(int i=0;i<DIM;i++){
                VecN<DIM,F64> xtore(x);
                xtore(i)-=grain.getDomain()(i);
                v_x.push_back(xtore);
                xtore(i)+=(2*grain.getDomain()(i));
                v_x.push_back(xtore);
            }
        }
    }
    VpTree<VecN<DIM,F64> > vptree;
    vptree.create(v_x);
    v_x.clear();
    Vec<bool> v_in;
    for(unsigned int i =0; i<grain.grains().size();i++){
        VecN<DIM,F64> x = grain.grains()[i]->x;
        Vec<VecN<DIM,F64> > v_neigh;
        Vec<double > v_dist;
        vptree.search(x,2,v_neigh,v_dist);
        if(v_dist[0]<=radius&&v_dist[1]<=radius)
            v_in.push_back(true);
        else
            v_in.push_back(false);
    }

    Vec<Germ<DIM> * > vlist_temp;
    for(unsigned int i =0; i<v_in.size();i++){
        if(v_in[i]==true){
            vlist_temp.push_back(grain.grains()[i]);
        }else{
            delete grain.grains()[i];
        }
    }
    grain.grains() = vlist_temp;
}

template<int DIM>
void  RandomGeometry::intersectionGrainToMask( ModelGermGrain<DIM> &  grain, const MatN<DIM,UI8> & img)throw(pexception)
{
    for(int i =0; i<(int)grain.grains().size();i++)
    {
        VecN<DIM,int> x = grain.grains()[i]->x;
        if(img(x)==0){
            delete grain.grains()[i];
            grain.grains()[i] =grain.grains()[grain.grains().size()-1];
            grain.grains().pop_back();
            i--;
        }
    }
}

template<int DIM>
void  RandomGeometry::addition(const ModelGermGrain<DIM> &  grain1,ModelGermGrain<DIM> &grain2)
{
    for(int i =0; i<(int)grain1.grains().size();i++)
    {
        grain2.grains().push_back(grain1.grains()[i]->clone());
    }
}

template<int DIM>
ModelGermGrain<DIM>  RandomGeometry::poissonPointProcess(VecN<DIM,F64> domain,double lambda)
{
    std::cout<<"Poisson Point process"<<std::endl;
    DistributionPoisson d(lambda*domain.multCoordinate());
    int nbr_VecNs = d.randomVariable();
    ModelGermGrain<DIM> grain;
    grain.setDomain(domain);
    DistributionUniformReal rx(0,1);
    for(int i =0;i<nbr_VecNs;i++)
    {
        Germ<DIM> * g = new Germ<DIM>();
        VecN<DIM,F64> x;
        for(int j = 0;j<DIM;j++)
        {
            F64 rand = rx.randomVariable();
            x[j]= rand*1.0*(domain)[j];
        }
        g->x=x;
        grain.grains().push_back(g);
    }
    return grain;
}

template<int DIM,typename TYPE>
ModelGermGrain<DIM>  RandomGeometry::poissonPointProcessNonUniform(const MatN<DIM,TYPE> & lambdafield)
{

    double lambdamax = Analysis::maxValue(lambdafield);
    DistributionPoisson d(lambdamax*lambdafield.getDomain().multCoordinate());
    int nbr_VecNs = d.randomVariable();
    ModelGermGrain<DIM> grain;
    grain.setDomain(lambdafield.getDomain());
    DistributionUniformReal rx(0,1);
    for(int i =0;i<nbr_VecNs;i++)
    {
        VecN<DIM,F64> x;
        for(int j = 0;j<DIM;j++)
        {
            F64 rand = rx.randomVariable();
            x[j]= rand*1.0*lambdafield.getDomain()(j);
        }
        double U = rx.randomVariable();
        if(U<lambdafield(x)/lambdamax){
            Germ<DIM> * germ = new Germ<DIM>();
            germ->x=x;
            grain.grains().push_back(germ);
        }

    }
    return grain;
}

template<int DIM>
void RandomGeometry::sphere( ModelGermGrain<DIM> &  grain, Distribution & dist)
{

    for(typename Vec<Germ<DIM> * >::iterator it= grain.grains().begin();it!=grain.grains().end();it++ )
    {
        Germ<DIM> * g = (*it);
        GrainSphere<DIM> * sphere = new GrainSphere<DIM>();
        sphere->setGerm(*g);
        F64 value =dist.randomVariable();
        sphere->radius = value;
        (*it) = sphere;
        delete g;
    }


}

template<int DIM>
void RandomGeometry::box( ModelGermGrain<DIM> &  grain,const DistributionMultiVariate& distradius,const DistributionMultiVariate& distangle )throw(pexception)
{

    if((int)distradius.getNbrVariable()!=DIM )
    {
        throw(pexception("In RandomGeometry::box, the radius DistributionMultiVariate must have d variables with d the space dimension"));//for radii (with a random angle) \n\t 3 random variables for the three radii plus 3 for the angle");
    }
    if(DIM==2 && distangle.getNbrVariable()!=1)
    {
        throw(pexception("In RandomGeometry::box, for d = 2, the angle distribution Vec must have 1 variable with d the space dimension"));
    }
    if(DIM==3 && distangle.getNbrVariable()!=3)
    {
        throw(pexception("In RandomGeometry::box, for d = 3, the angle distribution Vec must have 3 variables with d the space dimension"));
    }

    for(typename Vec<Germ<DIM> * >::iterator it = grain.grains().begin();it!=grain.grains().end();it++ )
    {
        Germ<DIM> * g = (*it);
        GrainBox<DIM> * box = new GrainBox<DIM>();
        box->setGerm(*g);
        VecN<DIM,F64> x;
        x = distradius.randomVariable();
        box->radius=x;
        if(DIM==3)
        {
            VecF64 v = distangle.randomVariable();
            for(int i=0;i<DIM;i++)
                box->orientation.setAngle_ei(v(i),i);
        }
        else
        {
            VecF64 v = distangle.randomVariable();
            box->orientation.setAngle_ei(v(0),0);
        }
        (*it) = box;
        delete g;

    }
}

template<int DIM>
void RandomGeometry::polyhedra( ModelGermGrain<DIM> &  grain,Vec<Distribution >& distradius,Vec<Distribution >& distnormal,Vec<Distribution >& distangle )throw(pexception)
{

    if((int)distnormal.size()*1.0/distradius.size()!=DIM )
    {
        throw(pexception("In RandomGeometry::polyhedra, we associate at each radius a normal Vec with d components (distradius.size()/distnormal.size()=d) "));
    }
    if(DIM==2 && distangle.size()!=1)
    {
        throw(pexception("In RandomGeometry::polyhedra, for d = 2, the angle distribution Vec must have 1 variable with d the space dimension"));
    }
    if(DIM==3 && distangle.size()!=3)
    {
        throw(pexception("In RandomGeometry::polyhedra, for d = 3, the angle distribution Vec must have 3 variables with d the space dimension"));
    }


    for(typename Vec<Germ<DIM> * >::iterator it = grain.grains().begin();it!=grain.grains().end();it++ )
    {
        Germ<DIM> * g = (*it);
        GrainPolyhedra<DIM> * polyedra = new GrainPolyhedra<DIM>();

        polyedra->setGerm(*g);
        for(int i =0;i<(int)distradius.size();i++){
            F64 distance =    distradius[i].randomVariable();
            VecN<DIM,F64> normal;
            for(int j=0;j<DIM;j++)
                normal[j]=distnormal[DIM*i+j].randomVariable();
            polyedra->addPlane(distance,normal);
        }
        if(DIM==3)
        {
            for(int i=0;i<DIM;i++)
                polyedra->orientation.setAngle_ei(distangle[i].randomVariable(),i);
        }
        else
        {
            polyedra->orientation.setAngle_ei(distangle[0].randomVariable(),0);
        }

        (*it) = polyedra;

        delete g;
    }
}

template<int DIM>
void RandomGeometry::ellipsoid( ModelGermGrain<DIM> &  grain,const DistributionMultiVariate & distradius,const DistributionMultiVariate & distangle )throw(pexception)
{

    if((int)distradius.getNbrVariable()!=DIM )
    {
        throw(pexception("In RandomGeometry::box, the radius DistributionMultiVariate must have d variables with d the space dimension"));//for radii (with a random angle) \n\t 3 random variables for the three radii plus 3 for the angle");
    }
    if(DIM==2 && distangle.getNbrVariable()!=1)
    {
        throw(pexception("In RandomGeometry::box, for d = 2, the angle distribution Vec must have 1 variable with d the space dimension"));
    }
    if(DIM==3 && distangle.getNbrVariable()!=3)
    {
        throw(pexception("In RandomGeometry::box, for d = 3, the angle distribution Vec must have 3 variables with d the space dimension"));
    }

    for(typename Vec<Germ<DIM> * >::iterator it = grain.grains().begin();it!=grain.grains().end();it++ )
    {
        Germ<DIM> * g = (*it);
        GrainEllipsoid<DIM> * box = new GrainEllipsoid<DIM>();
        box->setGerm(*g);

        VecN<DIM,F64> x;
        x = distradius.randomVariable();
        box->setRadius(x);
        if(DIM==3)
        {
            VecF64 v = distangle.randomVariable();
            for(int i=0;i<DIM;i++)
                box->orientation.setAngle_ei(v(i),i);
        }
        else
        {
            VecF64 v = distangle.randomVariable();
            box->orientation.setAngle_ei(v(0),0);
        }

        (*it) = box;
        delete g;
    }
}


template <int DIM>
void RandomGeometry::divideTilling(const Vec<int> & v,Private::IntersectionRectangle<DIM> &rec,const  ModelGermGrain<DIM> & grainlist, MatN<DIM,RGBUI8> &img)
{


    // divide and conquer
    if(v.size()>10 && rec.size.allSuperior(2))
    {
        VecN<DIM,F64> upleft = rec.x - rec.size*0.5;
        for(int i =0; i<std::pow(2,DIM*1.0);i++)
        {
            VecN<DIM,F64> rectempsize,rectempcenter;
            for(int j =0; j<DIM;j++)
            {
                F64 value =  rec.size[j]*0.5;
                if((int)(i/std::pow(2,j*1.0))%2==0)
                {
                    rectempsize[j]= std::floor(value);
                    rectempcenter[j]=upleft [j]+0.5*rectempsize[j];

                }
                else
                {
                    rectempsize[j]= std::ceil(value);
                    rectempcenter[j]=upleft [j]+ std::floor(value)+0.5*rectempsize[j];

                }
            }
            Private::IntersectionRectangle<DIM> rectemp;
            rectemp.x = rectempcenter;
            rectemp.size = rectempsize;

            Vec<int> vtemp;
            for(int i=0;i<(int)v.size();i++)
            {
                if(rectemp.perhapsIntersection(grainlist.grains()[v[i]],grainlist.getDomain(),grainlist.getBoundaryCondition())==true)
                {

                    vtemp.push_back(v[i]);
                }
            }
            divideTilling(vtemp,rectemp,grainlist, img);
        }
    }
    else
    {


        VecN<DIM,F64> upleft = rec.x - rec.size*0.5;

        VecN<DIM,int> sizeboule =rec.size;
        typename MatN<DIM,int>::IteratorEDomain b(sizeboule);
        while(b.next())
        {

            VecN<DIM,int> ximg =  b.x() + ((upleft));
            VecN<DIM,F64> y = b.x();
            y = y;
            VecN<DIM,F64> x = upleft + y;

            Vec<int> v_hit;
            for(int i=0;i<(int)v.size();i++)
            {
                VecN<DIM,F64> xx;
                if(grainlist.getBoundaryCondition()==MATN_BOUNDARY_CONDITION_PERIODIC)
                    xx =  Private::smallestDistanceTore(x,(grainlist.grains()[v[i]])->x,grainlist.getDomain());
                else
                    xx = x;
                if((grainlist.grains()[v[i]])->intersectionPoint(xx)==true)
                {
                    v_hit.push_back(v[i]);
                }
            }
            if(v_hit.size()!=0)
            {
                //Max
                if(grainlist.getModel()==Boolean)
                {
                    Vec<int>::iterator it;
                    for(it=v_hit.begin();it!=v_hit.end();it++)
                        img(ximg) = std::max (img(ximg),   (grainlist.grains()[*it])->color);
                }else if(grainlist.getModel()==DeadLeave){
                    int v=0;
                    Vec<int>::iterator it;
                    if(v_hit.size()>6)
                        v=0;
                    for(it=v_hit.begin();it!=v_hit.end();it++){
                        if(v<*it){
                            img(ximg) =   grainlist.grains()[*it]->color;
                            v = *it;
                        }
                    }
                }
                else if(grainlist.getModel()==Transparent){
                    std::sort(v_hit.begin(),v_hit.end());
                    img(ximg) =   (grainlist.grains()[*(v_hit.begin())])->color;
                    Vec<int>::iterator it;
                    for(it=v_hit.begin()+1;it!=v_hit.end();it++){
                        //                        std::cout<<img(ximg)<<"t"<<transparancy<<" "<<transparancy*RGBF64(  (grainlist.v_list[*it])->color)<<" "<<img(ximg)<<std::endl;
                        img(ximg) = grainlist.getTransparency()*RGBF64(  (grainlist.grains()[*it])->color)+(1-grainlist.getTransparency())*RGBF64(img(ximg));
                        //                        std::cout<<"t"<<transparancy<<" "<<transparancy*RGBF64(  (grainlist.v_list[*it])->color)<<" "<<img(ximg)<<std::endl;
                    }
                }else{
                    Vec<int>::iterator it;
                    for(it=v_hit.begin();it!=v_hit.end();it++){
                        img(ximg) +=   grainlist.grains()[*it]->color;
                    }
                }
            }

        }
    }

}

template<int DIM>
pop::MatN<DIM,pop::RGBUI8> RandomGeometry::continuousToDiscrete(const ModelGermGrain<DIM> &grain)
{
    Vec<int> v(grain.grains().size(),0);
    for(int i =0; i<(int)v.size();i++)
        v[i]=i;
    MatN<DIM,RGBUI8>  img (grain.getDomain());
    Private::IntersectionRectangle<DIM> rec;
    rec.x = grain.getDomain()*0.5;
    rec.size = grain.getDomain();
    RandomGeometry::divideTilling(v,rec,  grain, img);
    return img;
}
template<int DIM>
MatN<DIM,UI8> RandomGeometry::gaussianThesholdedRandomField(const Mat2F64 &mcorre,const VecN<DIM,int> &domain,MatN<DIM,F64> & gaussianfield )
{
    CollectorExecutionInformationSingleton::getInstance()->startExecution("gaussianThesholdedRandomField");
    double porosity = mcorre(0,1);
    Distribution f("1/(2*pi)^(1./2)*exp(-(x^2)/2)");
    double beta= Statistics::maxRangeIntegral(f,porosity,-3,3,0.001);
    Distribution Pmagnitude = RandomGeometry::generateProbabilitySpectralDensity(mcorre,beta);

    DistributionUniformReal d2pi(0,2*3.14159265);
    DistributionMultiVariateUnitSphere dpshere(DIM);
    int number_cosinus=10000;
    Vec< VecN<DIM, double> > direction(number_cosinus);
    Vec<  double > module(number_cosinus);
    Vec<  double > phase(number_cosinus);
    for(int i=0;i<number_cosinus;i++){
        double p = d2pi.randomVariable();
        phase[i] = p;
        module[i] = Pmagnitude.randomVariable();
        direction[i]= dpshere.randomVariable();
        //        std::cout<<phase[i]<<std::endl;
        //        std::cout<<module[i]<<std::endl;
        //        std::cout<<direction[i]<<std::endl;
        //        sleep(1);


    }
    gaussianfield.resize(domain);
    typename MatN<DIM,double>::IteratorEDomain b(gaussianfield.getIteratorEDomain());
    while(b.next())
    {
        if(DIM==3&&b.x()(0)==0&&b.x()(1)==0)
            CollectorExecutionInformationSingleton::getInstance()->progression(1.0*b.x()(2)/domain(2));
        if(DIM==2&&b.x()(0)==0)
            CollectorExecutionInformationSingleton::getInstance()->progression(1.0*b.x()(1)/domain(1));
        double sum=0;
        VecN<DIM, double> x = b.x();
        for(int i=0;i<number_cosinus;i++)
        {
            sum +=std::cos(productInner(x,direction[i])*module[i]+phase[i]);
        }
        sum= sum*std::pow(2./number_cosinus,1./2.);
        gaussianfield(b.x())=sum;
    }
    //    Mat2UI8 mm =Processing::greylevelRange(gaussianfield,0,255);
    //    mm.display();
    MatN<DIM,unsigned char> gaussianfieldthresolded(domain);

    double betamin= beta -1;
    double betamax= beta +1;
    bool test =true;
    do{
        test =false;
        gaussianfieldthresolded = Processing::threshold(gaussianfield,beta,100);
        Mat2F64 mhisto = Analysis::histogram(gaussianfieldthresolded);
        double porositymodel =mhisto(0,1);
        double porosityref   = mcorre(0,1);
        std::cout<<"porositymodel "<<porositymodel<<std::endl;
        std::cout<<"porosityref   "<<porosityref<<std::endl;
        if(absolute(porosityref - porositymodel)>0.0001){
            if(porositymodel>porosityref )
                betamax = beta;
            else
                betamin = beta;
            test=true;
            beta = betamin  + (betamax-betamin)/2;
        }
    }while(test==true);
    CollectorExecutionInformationSingleton::getInstance()->endExecution("gaussianThesholdedRandomField");
    return gaussianfieldthresolded;
}
template<int DIM>
MatN<DIM,UI8> RandomGeometry::randomStructure(const VecN<DIM,I32>& domain, const Mat2F64& volume_fraction){
    MatN<DIM,UI8> out (domain);
    Vec<DistributionUniformInt> dist;
    for(int i=0;i<DIM;i++)
        dist.push_back(DistributionUniformInt(0,out.getDomain()(i)-1));
    for(unsigned int i=1;i<volume_fraction.sizeI() ;i++)
    {
        int nbr= volume_fraction(i,1)*out.getDomain().multCoordinate();
        int j=0;
        while(j<nbr)
        {
            VecN<DIM,int> x;
            for(int k=0;k<DIM;k++)
                x(k)=dist[k].randomVariable();
            if(out(x)==0)
            {
                j++;
                out(x)=i;
            }
        }

    }
    return out;
}


template<int DIM1,int DIM2>
void RandomGeometry::annealingSimutated(MatN<DIM1,UI8> & model,const MatN<DIM2,UI8> & img_reference,F64 nbr_permutation_by_pixel, int lengthcorrelation,double temperature_inverse){

    CollectorExecutionInformationSingleton::getInstance()->startExecution("annealingSimutated");
    if(temperature_inverse==-1)
        temperature_inverse=model.getDomain().multCoordinate();
    if(lengthcorrelation==-1){
        lengthcorrelation = 10000;
        for(int i=0;i<DIM1;i++){
            lengthcorrelation = minimum(lengthcorrelation,model.getDomain()(i)-1);
        }
        for(int i=0;i<DIM2;i++){
            lengthcorrelation = minimum(lengthcorrelation,img_reference.getDomain()(i)/2);
        }
    }
    std::cout<<"correlation lenght "<<lengthcorrelation<<std::endl;
    MatN<DIM2,UI8> ref= Processing::greylevelRemoveEmptyValue(img_reference);
    model= Processing::greylevelRemoveEmptyValue(model);
    Mat2F64 m = Analysis::histogram(ref);
    int nbrphase=m.sizeI();

    Vec<Vec< Vec<int> > > vref = RandomGeometry::autoCorrelation(ref,lengthcorrelation,nbrphase);
    Vec<Vec< Vec<int> > > vmodel = RandomGeometry::autoCorrelation(model,lengthcorrelation,nbrphase);


    Vec<Vec< Vec<int> > > vrefcross = RandomGeometry::autoCorrelationCross(ref,lengthcorrelation,nbrphase);
    Vec<Vec< Vec<int> > > vmodelcross = RandomGeometry::autoCorrelationCross(model,lengthcorrelation,nbrphase);

    int count=0;
    double energy_current =10000;

    MatNDisplay d;


    typename MatN<DIM1,UI8>::IteratorENeighborhood it(model.getIteratorENeighborhood(1,0));
    while(nbr_permutation_by_pixel>count*1.0/model.getDomain().multCoordinate()){
        typename MatN<DIM1,UI8>::E p1,p2;
        int state1,state2=0;
        bool boundary=false;
        do{
            for(int i=0;i<DIM1;i++)
                p1(i)=rand()%model.getDomain()(i);
            state1 = model(p1);

            it.init(p1);
            int temp;
            while(it.next()){
                temp=model(it.x());
                if(temp!=state1){
                    state2=temp;
                    boundary =true;
                }
            }
        }while(boundary==false);
        boundary=false;
        do{
            for(int i=0;i<DIM1;i++)
                p2(i)=rand()%model.getDomain()(i);
            if(model(p2)==state2){
                it.init(p2);
                while(it.next()){
                    if(model(it.x())==state1){
                        boundary =true;
                    }
                }
            }
        }while(boundary==false);
        RandomGeometry::switch_state_pixel(vmodel,model,lengthcorrelation,p1,state2);
        RandomGeometry::switch_state_pixel_cross(vmodelcross,model,lengthcorrelation,p1,state2);
        model(p1)=state2;
        RandomGeometry::switch_state_pixel(vmodel,model,lengthcorrelation,p2,state1);
        RandomGeometry::switch_state_pixel_cross(vmodelcross,model,lengthcorrelation,p2,state1);
        model(p2)=state1;
        double energytemp = RandomGeometry::energy(vref,vmodel);
        energytemp += RandomGeometry::energy(vrefcross,vmodelcross);
        temperature_inverse++;
        if(annealingSimutatedLaw(energytemp, energy_current,temperature_inverse)==false){
            //            if(energytemp> energy_current){

            RandomGeometry::switch_state_pixel(vmodel,model,lengthcorrelation,p1,state1);
            RandomGeometry::switch_state_pixel_cross(vmodelcross,model,lengthcorrelation,p1,state1);
            model(p1)=state1;
            RandomGeometry::switch_state_pixel(vmodel,model,lengthcorrelation,p2,state2);
            RandomGeometry::switch_state_pixel_cross(vmodelcross,model,lengthcorrelation,p2,state2);
            model(p2)=state2;
        }else{
            energy_current = energytemp;
        }

        if(count%(model.getDomain().multCoordinate()/10)==0){
            CollectorExecutionInformationSingleton::getInstance()->progression(1.0*count/model.getDomain().multCoordinate(),"Number of permutation per pixels/voxels");
            CollectorExecutionInformationSingleton::getInstance()->progression(energy_current,"Energy");
            d.display(Visualization::labelToRandomRGB(model));
        }
        count++;

    }
    CollectorExecutionInformationSingleton::getInstance()->endExecution("annealingSimutated");
}

template<int DIM>
void RandomGeometry::matrixBinary( ModelGermGrain<DIM> &  grain,const  MatN<DIM,UI8> &img,const DistributionMultiVariate& distangle )throw(pexception){
    if(DIM==2 && distangle.getNbrVariable()!=1)
    {
        throw(pexception("In RandomGeometry::box, for d = 2, the angle distribution Vec must have 1 variable with d the space dimension"));
    }
    if(DIM==3 && distangle.getNbrVariable()!=3)
    {
        throw(pexception("In RandomGeometry::box, for d = 3, the angle distribution Vec must have 3 variables with d the space dimension"));
    }
    for(typename Vec<Germ<DIM> * >::iterator it = grain.grains().begin();it!=grain.grains().end();it++ )
    {
        Germ<DIM> * g = (*it);
        GrainFromBinaryMatrix<DIM> * box = new GrainFromBinaryMatrix<DIM>();
        box->setGerm(*g);
        box->img = &img;
        if(DIM==3)
        {
            VecF64 v = distangle.randomVariable();
            for(int i=0;i<DIM;i++)
                box->orientation.setAngle_ei(v(i),i);
        }
        else
        {
            VecF64 v = distangle.randomVariable();
            box->orientation.setAngle_ei(v(0),0);
        }
        (*it) = box;
        delete g;
    }
}


}
#endif // RANDOMGEOMETRY_H
