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

#ifndef GRAINGERM_H_
#define GRAINGERM_H_

#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"
#include"data/germgrain/Germ.h"

/*! \ingroup Data
* \defgroup Other Other
* \brief general utility classes
*/
/// @cond DEV
namespace pop
{


template<int DIM>
class POP_EXPORTS GrainSphere: public Germ<DIM>
{
public:
    F64 radius;
    F64 getRadiusBallNorm0IncludingGrain(){
        return radius;
    }
    bool intersectionPoint(const VecN<DIM,F64> &  x)
    {
        VecN<DIM,F64>  temp=x-this->x;
        if(temp.normPower()<=radius*radius)return true;
        else return false;
    }
    virtual Germ<DIM> * clone()const
    {
        return new GrainSphere<DIM>(*this);
    }
};


class POP_EXPORTS GrainEquilateralRhombohedron:public Germ<3>
{
private:
    F64 cosangle;
    F64 angleequi;
    VecN<3,F64> normalplanx;
    VecN<3,F64> normalplany;
    VecN<3,F64> normalplanz;

public:
    F64 radius;
    OrientationEulerAngle<3> orientation;
    GrainEquilateralRhombohedron();
    void setAnglePlane(F64 angleradian);
    virtual F64 getRadiusBallNorm0IncludingGrain();
    bool intersectionPoint(const VecN<3,F64> &  x);
    virtual Germ<3> * clone()const;
};




template<int DIM>
class POP_EXPORTS GrainPolyhedra:public Germ<DIM>
{
public:
    Vec<F64> radius;
    F64 maxRadius;
    Vec<VecN<DIM,F64> > normalplan;
    OrientationEulerAngle<DIM> orientation;
    GrainPolyhedra()
        :maxRadius(0)
    {
    }

    virtual F64 getRadiusBallNorm0IncludingGrain(){
        //TODO Find the minimum radius depending on the normal and the radius
        return 4*maxRadius;
    }


    void addPlane(F64 distance, VecN<DIM,F64> normal)
    {
        normalplan.push_back(normal);
        radius.push_back(distance);
        maxRadius= maximum(maxRadius,distance);
    }
    bool intersectionPoint(const VecN<DIM,F64> &  x)
    {
        VecN<DIM,F64> p = this->x -x;
        p = this->orientation.inverseRotation(p);
        for(int i =0;i<(int)radius.size();i++){
            if(productInner(normalplan[i],p)>this->radius[i])
                return false;
        }
        return true;
    }
    virtual Germ<DIM> * clone()const
    {
        return new GrainPolyhedra<DIM>(*this);
    }
};

template<int DIM>
class POP_EXPORTS GrainEllipsoid:public Germ<DIM>
{
private:
    VecN<DIM,F64> radius;
public:

    VecN<DIM,F64> radiusinverse;
    OrientationEulerAngle<DIM> orientation;
    virtual void setRadius(const VecN<DIM,F64> & _radius){
        radius=_radius;
        for(int i =0;i<DIM;i++)
            radiusinverse[i]=1./radius[i];
    }
    virtual F64 getRadiusBallNorm0IncludingGrain(){
        return 1.1*radius.norm(0);
    }
    bool intersectionPoint(const VecN<DIM,F64> &  p)
    {
        VecN<DIM,F64> pp = this->x -p;
        pp = this->orientation.inverseRotation(pp);
        F64 sum=0;
        for(int i =0;i<DIM;i++)
        {
            sum+=pp[i]*pp[i]*radiusinverse[i]*radiusinverse[i];
        }
        if(sum<=1)
            return true;
        else
            return false;
    }
    virtual Germ<DIM> * clone()const
    {
        return new GrainEllipsoid<DIM>(*this);
    }
};

class POP_EXPORTS GrainCylinder:public Germ<3>
{
public:
    F64 radius;
    F64 height;
    F64 maxradius;
    OrientationEulerAngle<3> orientation;
    GrainCylinder();
    virtual F64 getRadiusBallNorm0IncludingGrain();
    bool intersectionPoint(const VecN<3,F64> &  x);
    virtual Germ<3> * clone()const;
};
template<int DIM>
class POP_EXPORTS GrainBox:public Germ<DIM>
{
public:
    VecN<DIM,F64> radius;
    OrientationEulerAngle<DIM> orientation;
    virtual F64 getRadiusBallNorm0IncludingGrain(){
        return 2*radius.norm(0);
    }
    bool intersectionPoint(const VecN<DIM,F64> &  x)
    {
        VecN<DIM,F64> p = this->x -x;
        p = this->orientation.inverseRotation(p);
        for(I32 i = 0;i<DIM;i++)
        {
            p(i)=absolute(p(i));
        }
        if(p.allInferior(this->radius))return true;
        else return false;
    }
    virtual Germ<DIM> * clone()const
    {
        return new GrainBox<DIM>(*this);
    }
};

template<int DIM>
class POP_EXPORTS GrainFromBinaryMatrix:public Germ<DIM>
{
public:
    const MatN<DIM,UI8> * img;
    OrientationEulerAngle<DIM> orientation;
    virtual F64 getRadiusBallNorm0IncludingGrain(){
        return this->img->getDomain().norm(0);
    }
    bool intersectionPoint(const VecN<DIM,F64> &  x){
            VecN<DIM,F64> p = this->x -x;
            p = this->orientation.inverseRotation(p);
            p += this->img->getDomain()/2;
            if(p.allSuperiorEqual(0)&&p.allInferior(this->img->getDomain())&&(*img)(p)!=0){
                return true;
            }
            else return false;
    }
    virtual Germ<DIM> * clone()const{
        return new GrainFromBinaryMatrix(*this);
    }
};


enum ModelGermGrainEnum
{
    Boolean=0,
    DeadLeave=1,
    Transparent=2,
    SpotNoise=3
};

/*! \ingroup Other
* \defgroup ModelGermGrain ModelGermGrain{2,3}
* \brief collection of geometrical figures for random geometry
*/
template<int DIM>
class POP_EXPORTS ModelGermGrain
{
    /*!
        \class pop::ModelGermGrain
        \ingroup ModelGermGrain
        \brief contain the list of elementary grains for the Germ/Grain model
        \author Tariel Vincent
        \tparam Dim Space dimension

         This class contains a vector of elementary grains


        \code
        DistributionUniformReal duniform_radius(3,25);
        DistributionMultiVariate dd (DistributionMultiVariate(duniform_radius,duniform_radius),duniform_radius);
        Vec3F64 domain(100,100,100);
        ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,0.00005);//generate the 2d Poisson VecN process
        DistributionMultiVariate angle(DistributionMultiVariate(Distribution(0,3.14159265,"UNIFORMREAL"),Distribution(0,3.14159265,"UNIFORMREAL")),Distribution(0,3.14159265,"UNIFORMREAL") );

        RandomGeometry::ellipsoid(grain,dd,angle);
        Mat3RGBUI8 img_VecN = RandomGeometry::continuousToDiscrete(grain);
        img_VecN*=0.5;
        Scene3d scene;
        pop::Visualization::marchingCube(scene,img_VecN);
        pop::Visualization::lineCube(scene,img_VecN);
        scene.display();
        \endcode
      */



    /*! \var _model
     * model of the germ grain that can be   Boolean,  DeadLeave or Transparent
     */
    ModelGermGrainEnum _model;


    /*! \var _transparency
     * coefficient of transparency for the Transparent model
     */
    F64 _transparency;
    /*! \var _grains
     * Vec of elementary grains
     */
    std::vector<Germ<DIM> * > _grains;
    /*! \var _domain
     * domain size of the model
     */
    VecN<DIM,F64> _domain;

    MatNBoundaryConditionType _boundary;
public:
    /*!
    \fn ModelGermGrain();
    *
    * default constructor
    */
    ModelGermGrain()
        :_model(Boolean),_transparency(1),_boundary(MATN_BOUNDARY_CONDITION_PERIODIC)
    {
    }
    /*!
    \fn ~ModelGermGrain()
    *
    * default destructor
    */
    virtual ~ModelGermGrain()
    {
        for(int i=0; i <(int)_grains.size();i++)
        {
            delete this->_grains[i];
        }
        this->_grains.clear();
    }
    /*!
    *
    * copy constructor
    */
    ModelGermGrain(const ModelGermGrain& g){
        this->_domain = g._domain;
        this->_transparency = g._transparency;
        this->_model = g._model;
        for(int i=0; i <(int)g._grains.size();i++){
            this->_grains.push_back(g._grains[i]->clone());
        }
    }
    ModelGermGrain  operator =(const ModelGermGrain& g){
        this->_domain = g._domain;
        this->_transparency = g._transparency;
        this->_model = g._model;
        for(int i=0; i <(int)_grains.size();i++){
            delete this->_grains[i];
        }
        this->_grains.clear();
        for(int i=0; i <(int)g._grains.size();i++){
            this->_grains.push_back(g._grains[i]->clone());
        }
        return *this;
    }
    void setModel(ModelGermGrainEnum model){
        this->_model = model;
    }
    void setTransparency(F64 transparency){
        this->_transparency = transparency;
    }
    void setDomain(const VecN<DIM,F64> & domain){
        _domain=domain;
    }
    void setBoundaryCondition(MatNBoundaryConditionType boundary){
        _boundary = boundary;
    }

    ModelGermGrainEnum getModel( )const{
        return this->_model;
    }
    F64 getTransparency()const{
        return this->_transparency;
    }

    VecN<DIM,F64> getDomain()const{
        return _domain;
    }
    MatNBoundaryConditionType getBoundaryCondition()const{
        return _boundary;
    }
    std::vector<Germ<DIM> * > &grains(){
        return _grains;
    }
    const std::vector<Germ<DIM> * > &grains()const{
        return _grains;
    }
};

typedef ModelGermGrain<2> ModelGermGrain2;
typedef ModelGermGrain<3> ModelGermGrain3;
}
/// @endcond
#endif /* GRAINGERM_H_ */

