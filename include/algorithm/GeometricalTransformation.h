#ifndef GEOMETRICALTRANSFORMATION_H
#define GEOMETRICALTRANSFORMATION_H

#include"data/functor/FunctorF.h"
#include"data/vec/VecN.h"
#include"data/mat/Mat2x.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"
#include"algorithm/ProcessingAdvanced.h"

namespace pop
{
/*!
* \defgroup GeometricalTransformation GeometricalTransformation
* \ingroup Algorithm
* \brief Matrix In -> Matrix Out (rotation, scale, projective, affine)
*
* This module provides some functions to apply geometrical transformations
* - directly for the basic transformation (scale, rotate, mirror)
* - by creating the the geometrical transformation matrix following by its application (projective, affine, composition of transformations).
\code
    Mat2UI8 m;
    m.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    MatNDisplay disp;
    Mat2x33F32 maffine2 = GeometricalTransformation::translation2DHomogeneousCoordinate(m.getDomain()/2);//go back to the buttom left corner (origin)
    maffine2 *= GeometricalTransformation::rotation2DHomogeneousCoordinate(0.1);//rotate
    maffine2 *= GeometricalTransformation::scale2DHomogeneousCoordinate(Vec2F32(1.3,1.1));//scale the image
    maffine2 *= GeometricalTransformation::translation2DHomogeneousCoordinate(-m.getDomain()/2);//from to the buttom left corner (origin), go to the center of the image

    int i=0;
    //iterate transformation
    do{
        m = GeometricalTransformation::transformHomogeneous2D(maffine2,m);//apply this transformation
        disp.display(m);//display it
        m.save("rotate"+BasicUtility::IntFixedDigit2String(i++,4)+".png");
    }while(disp.is_closed()==false);
\endcode
* \image html rotate2.gif
*
*
*/
struct POP_EXPORTS GeometricalTransformation
{

    /*!
        \class pop::GeometricalTransformation
        \ingroup GeometricalTransformation
        \brief Geometrical Transformation
        \author Tariel Vincent

    */



    //-------------------------------------
    //
    //! \name Direct transformation
    //@{
    //-------------------------------------
    /*!
     * \brief get a 2d matrix from a 3d matrix (e.g. a slice in a core sample)
     * \param m input 3d matrix
     * \param index_plane plane index
     * \param fixed_coordinate coordinate fixed
     *
     * \code
     * \endcode
     * \image html Lena.bmp
     * \image html Lenascale.png
    */
    template<typename PixelType>
    static MatN<2,PixelType> subResolution(const MatN<2,PixelType> &m , int sub_resolution_factor)
    {
        MatN<2,PixelType> m_sub (m.getDomain()/sub_resolution_factor);
        typename MatN<2,PixelType>::iterator it_sub=m_sub.begin();
        typename MatN<2,PixelType>::const_iterator it=m.begin();
        typename MatN<2,PixelType>::const_iterator it_index_start=m.begin();
        int step_m_y=sub_resolution_factor;
        int step_m_x=(sub_resolution_factor)*m.sizeJ();
        for(unsigned int i=0;i<m_sub.sizeI();i++){
            for(unsigned int j=0;j<m_sub.sizeJ();j++){
                *it_sub= *it;
                it_sub++;
                it+=step_m_y;
            }
            it_index_start+=step_m_x;
            it=it_index_start;
        }
        return m_sub;
    }
    /*!
     * \brief get a 2d matrix from a 3d matrix (e.g. a slice in a core sample)
     * \param m input 3d matrix
     * \param index_plane plane index
     * \param fixed_coordinate coordinate fixed
     *
     * \code
     * \endcode
     * \image html Lena.bmp
     * \image html Lenascale.png
    */
    template<typename VoxelType>
    static MatN<2,VoxelType> plane(const MatN<3,VoxelType> &m , int index_plane, int fixed_coordinate=2)
    {
        MatN<2,VoxelType> plane (m.getDomain().removeCoordinate(2));
        ForEachDomain2D(x2d,plane){
            Vec3I32 x3d = x2d.addCoordinate(fixed_coordinate,index_plane);
            plane(x2d)= m(x3d);
        }
        return plane;
    }


    template<int DIM>
    struct _FunctorScale
    {
        VecN<DIM,F32> _alpha;
        MatNInterpolation _interpolation;
        _FunctorScale(VecN<DIM,F32> alpha,MatNInterpolation interpolation)
            :_alpha(alpha),_interpolation(interpolation){}
        template<typename PixelType>
        PixelType operator()(const MatN<DIM,PixelType> & f , const typename MatN<DIM,PixelType>::E itx){
            VecN<DIM,F32> x;
            x=(VecN<DIM,F32>(itx)-0.5)*_alpha;
            if(_interpolation.isValid(f.getDomain(),x)){
                return _interpolation.apply(f,x);
            }else{
                return PixelType();
            }
        }
    };
    /*!
     * \brief scale the image
     * \param scale vector of scale factor
     * \param f input matrix
     * \param interpolation: 0=no, 1=BI-LINEAR
     *
     * scale the matrix with its domain of definition. For instance, in this code:
     * \code
     * Mat2RGBUI8 img;
     * img.load("D:/Users/vtariel/Desktop/ANV/Population/doc/image2/Lena.bmp");
     * std::cout<<img.getDomain()<<std::endl;
     * img = GeometricalTransformation::scale(img,Vec2F32(0.5,2));
     * img.save("D:/Users/vtariel/Desktop/ANV/Population/doc/image2/Lenascale.png");
     * \endcode
     *  the initial domain is (256,,256) and then (128,512)
     * \image html Lena.bmp
     * \image html Lenascale.png
    */
    template<int DIM,  typename PixelType>
    static MatN<DIM,PixelType> scale(const MatN<DIM,PixelType> & f,const VecN<DIM,F32> & scale,MatNInterpolation interpolation=MATN_INTERPOLATION_BILINEAR)
    {

        typename MatN<DIM,PixelType>::Domain domain (scale*VecN<DIM,F32>(f.getDomain()));
        MatN<DIM,PixelType> temp(domain);
        VecN<DIM,F32> alpha = VecN<DIM,F32>(1.f)/scale;
        _FunctorScale<DIM> func(alpha,interpolation); 
        typename MatN<DIM,PixelType>::IteratorEDomain it = temp.getIteratorEDomain();
        forEachFunctorBinaryFunctionE(f,temp,func,it);
		
        return temp;
    }
    /*!
    \brief rotate the input matrix with an angle plus or minus PI/2
    \param f  input function
    \param plus_minus center (1=PI/2, else - PI/2)
    \return rotate function
    *
    *
    * \code
    Mat2UI8 m;
    m.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    m=GeometricalTransformation::rotateMultPi_2(m,1);
    m.display();
    m.save("../doc/image2/lenarotpi2.png");
    * \endcode
    * \image html lena.png
    * \image html lenarotpi2.png
    */
    template<int DIM,typename PixelType>
    static MatN<DIM,PixelType> rotateMultPi_2(const MatN<DIM,PixelType> & f,int plus_minus=1)
    {
        MatN<DIM,PixelType> g(f.getDomain()(1),f.getDomain()(0));
        typename MatN<DIM,PixelType>::IteratorEDomain it = f.getIteratorEDomain();
        if(plus_minus==1){
            while(it.next()){
                g(g.getDomain()(0)-1-it.x()(1),it.x()(0))=f(it.x());
            }
        }else{
            while(it.next()){
                g(it.x()(1),g.getDomain()(1)-1-it.x()(0))=f(it.x());
            }
        }
        return g;
    }
    /*!
    \brief rotate the input matrix
    \param f  input function
    \param angle rotation angle
    \return rotate function
    *
    * rotate the matrix
    *
    * \code
        Mat2UI8 lena;
        lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
        GeometricalTransformation::rotate( lena, PI/6).display();
    * \endcode
    * \image html lena.png
    * \image html lenarotpi6.png
    */
    template<int DIM,typename PixelType>
    static MatN<DIM,PixelType> rotate(const MatN<DIM,PixelType> & f,F32 angle,MatNBoundaryConditionType boundarycondtion=MATN_BOUNDARY_CONDITION_BOUNDED)
    {
        if(DIM==2){
            Mat2x22F32 rot22 = GeometricalTransformation::rotation2D(angle);
            typename MatN<DIM,PixelType>::IteratorEDomain it = f.getIteratorEDomain();

            MatN<DIM,PixelType> g(f.getDomain());
            Vec2F32 c(f.getDomain()/2);
            it.init();
            MatNBoundaryCondition condition(boundarycondtion);
            while(it.next()){
                Vec2F32 y =Vec2F32(it.x()(0),it.x()(1))-c;
                y= (rot22*y)+c;

                if(condition.isValid(f.getDomain(),y)){
                    condition.apply(f.getDomain(),y);
                    g(it.x())= f.interpolationBilinear(y);
                }
            }
            return g;
        }else{
            return f;
        }
    }
    /*!
    \brief Mirror (flip) an matrix along the specified axis given by the coordinate (0=x-axis, 1=y-axis)
    \param f  input function
    \param coordinate mirror axis
    \return flip function
    *
    * Mirror an matrix along the specified axis given by the coordinate (0=x-axis, 1=y-axis)
    *
    * \code
    Mat2UI8 lena;
    lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    GeometricalTransformation::mirror( lena, 1).display();
    * \endcode
    * \image html lena.png
    * \image html lenamirror.png
    */
    template<int DIM,typename PixelType>
    static MatN<DIM,PixelType> mirror(const MatN<DIM,PixelType> & f,int coordinate=0){
        MatN<DIM,PixelType> h(f.getDomain());
        typename MatN<DIM,PixelType>::IteratorEDomain it = f.getIteratorEDomain();
        while(it.next()){
            typename MatN<DIM,PixelType>::E x=it.x();
            x(coordinate)=h.getDomain()(coordinate)-1-x(coordinate);
            h(x)=f(it.x());
        }
        return h;
    }
    /*!
     * \brief matrix translation
     * \param trans translation vector
     * \param f input matrix
     * \param boundarycondtion
     *
     * translate the matrix with the input vector h(x)= f(x+trans)
     *
     * \sa pop::Representation::FFTDisplay
     */
    template<int DIM,typename PixelType>
    static MatN<DIM,PixelType> translate(const MatN<DIM,PixelType> & f,const typename MatN<DIM,PixelType>::E &trans,MatNBoundaryConditionType boundarycondtion=MATN_BOUNDARY_CONDITION_PERIODIC)
    {
        MatN<DIM,PixelType> temp(f.getDomain());
        typename MatN<DIM,PixelType>::IteratorEDomain it = f.getIteratorEDomain();
        MatNBoundaryCondition condition(boundarycondtion);
        while(it.next())
        {
            typename MatN<DIM,PixelType>::E z = it.x()+trans;
            if(condition.isValid(f.getDomain(),z)){
                condition.apply(f.getDomain(),z);
                temp(z)=f(it.x());
            }
        }
        return temp;
    }
    /*!
    \brief elastic deformation of the input matrix
    \param Img input matrix
    \param sigma standard deviation (correlation length)
    \param alpha intensity of the deformation (deformation length)
    \return elastic deformation function
    *
    *  elastic deformation of the input matrix
    *
    * \code
    Mat2UI8 lena;
    lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    GeometricalTransformation::elasticDeformation( lena, 20, 8).display();
    * \endcode
    * \image html lenaED.png
    */
    template<int DIM, typename Type>
    static MatN<DIM,Type> elasticDeformation(const MatN<DIM,Type> & Img,F32 sigma=10,F32 alpha=8)
    {

        VecN<DIM,Mat2F32> dist(Mat2F32(Img.getDomain()));
        DistributionSign d;
        typename MatN<DIM,Type>::IteratorEDomain it = Img.getIteratorEDomain();
        while(it.next()){
            for(unsigned int i=0;i<DIM;i++)
                dist(i)(it.x())=d.randomVariable();
        }
        for(unsigned int i=0;i<DIM;i++)
            dist(i) = ProcessingAdvanced::smoothGaussian(dist(i),sigma,(int)(sigma*2.5f),dist(i).getIteratorEDomain());
        it.init();
        F32 dist_sum=0;
        int number=0;
        while(it.next()){
            F32 distance=0;
            for(unsigned int i=0;i<DIM;i++)
                distance+=dist(i)(it.x())*dist(i)(it.x());
            dist_sum+=distance;
            number++;
        }
        F32 standard_deviation = std::sqrt(dist_sum/number);

        it.init();
        while(it.next()){
            for(unsigned int i=0;i<DIM;i++)
                dist(i)(it.x())*=(alpha/standard_deviation);
        }
        MatN<DIM,Type> mdist(Img.getDomain());
        it.init();
        while(it.next()){
            VecN<DIM,F32> xx(it.x());
            for(unsigned int i=0;i<DIM;i++)
                xx(i)+=dist(i)(it.x());
            if(Img.isValid(xx)){
                mdist(it.x())=Img.interpolationBilinear(xx);
            }
        }
        return mdist;
    }

    //@}
    //-------------------------------------
    //
    //! \name Get transformation Matrices
    //@{
    //-------------------------------------
    /*!
     * \brief 2D rotation matrix
     * \param theta_radian angle in radian
     * \return  Rotation matrix
     *
     *  Generate the 2D rotation matrix from the angle \a theta_radian (not in homogeneous coordinates)\n
     * \f[ R(\theta) = \left( \begin{array}{cc} \cos \theta & -\sin \theta \\\sin \theta & \cos \theta \end{array} \right)\f].
     *
    */
    static pop::Mat2x22F32 rotation2D(F32 theta_radian);

    /*!
     * \brief 3D rotation matrix
     * \param theta_radian angle in radian
     * \param coordinate axis
     * \return  rotation matrix
     *
     *  Generate the 3D rotation matrix from the rotation of angle \a theta_radian about the axis at the given coordinate (not in homogeneous coordinates).
     *  For coordinatte=0 we have,\f$R_x(\theta) = \left(\begin{array}{ccc}
     *  1 & 0 & 0 \\
     *  0 & \cos \theta & -\sin \theta \\
     *  0 & \sin \theta  & \cos \theta \\
     *  \end{array}\right)\f$, for coordinatte=1 we have,
     *  \f$R_y(\theta) = \left(\begin{array}{ccc}
     *  \cos \theta & 0 & \sin \theta \\
     *  0 & 1 & 0 \\[3pt]
     *  -\sin \theta & 0 & \cos \theta \end{array}\right) \f$ and for coordinatte=2 we have,
     *  \f$R_y(\theta) = \left(\begin{array}{ccc}
     *  \cos \theta & -\sin \theta & 0 \\
     *  \sin \theta & \cos \theta & 0\\
     *  0 & 0 & 1\\ \end{array}\right) \f$
     */
    static pop::Mat2x33F32 rotation3D(F32 theta_radian,int coordinate);
    /*!
     * \brief rotation around an axis
     * \param u  unit vector indicating the direction of an axis of rotation
     * \param  angle_radian the magnitude of the rotation about the axis in radian
     * \return  rotation matrix
     *
     * \code
     * Vec3F32 axis(1,1,1);
     * axis/=axis.norm();
     * Vec3F32 v(1,1,0);
     * v/=v.norm();

     * Mat2x33F32 mrot = GeometricalTransformation::rotationFromAxis(axis,0.1);

     * Scene3d scene;
     * scene.addGeometricalFigure(FigureArrow::referencielEuclidean());

     * FigureArrow * arrowaxis= new FigureArrow;
     * arrowaxis->setRGB(RGBUI8(255,255,255));
     * arrowaxis->setArrow(Vec3F32(0,0,0),axis,0.1);
     * scene._v_figure.push_back(arrowaxis);

     * FigureArrow * arrow = new FigureArrow;
     * arrow->setRGB(RGBUI8(0,0,255));
     * arrow->setArrow(Vec3F32(0,0,0),v,0.1);
     * scene._v_figure.push_back(arrow);

     * int i=0;
     * scene.display(false);
     * while(1==1){
     *     scene.lock();
     *     v= mrot*v;
     *     arrow->setArrow(Vec3F32(0,0,0),v,0.1);
     *     scene.unlock();
     *     scene.snapshot(std::string("rotateaxis"+BasicUtility::IntFixedDigit2String(i++,4)+".png").c_str());
     * }
     * \endcode
     * \image html rotateaxis.gif
    */
    static Mat2x33F32 rotationFromAxis(Vec3F32 u,F32 angle_radian);

    /*!
     * \brief rotation from vector to another vector
     * \param s  unit vector indicating the direction of the first vector
     * \param t  unit vector indicating the direction of the second vector
     * \return  rotation matrix
     *
    */
    static Mat2x33F32 rotationFromVectorToVector(const Vec3F32 & s, const Vec3F32 &  t);
    /*!
     * \brief 3D rotation matrix in homogeneous coordinate
     * \param theta_radian angle in radian
     * \param coordinate axis
     * \return  Rotation Mat2F32
     *
     *  Generate the 3D rotation matrix from the rotation of angle \a theta_radian about the axis at the given coordinate  in homogeneous coordinates:\n
     *  For coordinatte=0 we have,\f$R_x(\theta) = \left(\begin{array}{cccc}
     *  1 & 0 & 0 & 0\\
     *  0 & \cos \theta & -\sin \theta &0 \\
     *  0 & \sin \theta  & \cos \theta &0\\
     *  0 & 0 & 0 & 1
     *  \end{array}\right)\f$, and so one
    */
    static Mat2F32 rotation3DHomogeneousCoordinate(F32 theta_radian,int coordinate);

    /*!
     * \brief 2D rotation matrix in homogeneous coordinate
     * \param theta_radian angle in radian
     * \return  Rotation Mat2F32
     *
     *  Generate the 2D rotation matrix from the rotation of angle \a theta_radian  in homogeneous coordinates:\n
     *  For coordinatte=0 we have,\f$R_x(\theta) = \left(\begin{array}{ccc}
     *   \cos \theta & -\sin \theta &0 \\
     *   \sin \theta  & \cos \theta &0\\
     *   0 & 0 & 1
     *  \end{array}\right)\f$, and so one
    */
    static pop::Mat2x33F32 rotation2DHomogeneousCoordinate(F32 theta_radian);

    /*!
     * \brief 2D translation matrix in homogeneous coordinate
     * \param t translation following this vector t
     * \return  Translation matrix
     *
     *  Generate the 3D translation matrix in homogeneous coordinates:\n
     *  We have\f$T(t=(tx,ty,tz)) = \left(\begin{array}{cccc}
     *  1 & 0 & 0 &tx\\
     *  0 & 1 & 0 &ty\\
     *  0 & 0 & 1 &tz\\
     *  0 & 0 & 0 & 1
     *  \end{array}\right)\f$
    */
    static Mat2F32 translation3DHomogeneousCoordinate(const Vec3F32 &t );

    /*!
     * \brief 2D translation matrix in homogeneous coordinates
     * \param t translation following this vector t
     * \return  Translation matrix
     *
     *
     *  We have\f$T(t=(tx,ty)) = \left(\begin{array}{ccc}
     *  1 & 0  &tx\\
     *  0 & 1  &ty\\
     *  0 & 0 & 1
     *  \end{array}\right)\f$
    */
    static pop::Mat2x33F32 translation2DHomogeneousCoordinate(const Vec2F32 &t );
    /*!
     * \brief 3d scale matrix in homogeneous coordinate
     * \param s  scale vector
     * \return  Scale matrix
     *
     *  Generate the 3D scale matrix in homogeneous coordinates:\n
     *  We have\f$S( s=(sx,sy,sz)) = \left(\begin{array}{cccc}
     *  sx & 0 & 0 &0\\
     *  0 & sy & 0 &0\\
     *  0 & 0 & sz &0\\
     *  0 & 0 & 0 & 1
     *  \end{array}\right)\f$
    */
    static Mat2F32         scale3DHomogeneousCoordinate(const Vec3F32 &s);

    /*!
     * \brief 2d scale matrix in homogeneous coordinate
     * \param s  scale vector
     * \return  Scale matrix
     *
     *  Generate the 2D scale matrix in homogeneous coordinates:\n
     *  We have\f$S( s=(sx,sy)) = \left(\begin{array}{ccc}
     *  sx & 0 & 0\\
     *  0 & sy & 0\\
     *  0 & 0 &  1
     *  \end{array}\right)\f$
    */
    static pop::Mat2x33F32 scale2DHomogeneousCoordinate(const Vec2F32 &s);
    /*!
     * \brief 3d scale matrix
     * \param s  scale vector
     * \return  Scale Mat2F32
     *
     *  Generate the 3D scale matrix :\n
     *  We have\f$S( s=(sx,sy,sz)) = \left(\begin{array}{ccc}
     *  sx & 0 & 0 \\
     *  0 & sy & 0 \\
     *  0 & 0 & sz
     *  \end{array}\right)\f$
 */
    static pop::Mat2x33F32 scale3D(const Vec3F32 &s);
    /*!
     * \brief 2d scale matrix
     * \param s  scale vector
     * \return  Scale Mat2F32
     *
     *  Generate the 2D scale matrix :\n
     *  We have\f$S( s=(sx,sy)) = \left(\begin{array}{cc}
     *  sx & 0 \\
     *  0 & sy
     *  \end{array}\right)\f$
    */
    static pop::Mat2x22F32 scale2D(const Vec2F32 &s);



    /*!
     * \brief affine transformation from 3 pairs of the corresponding points
     * \param src 3 source points
     * \param dst 3 destination points
     * \param isfastinversion true=gaussian elimintation, false=school way
     *
     *
     \code
        Mat2UI8 m;
        m.load("/usr/share/doc/opencv-doc/examples/c/lena.jpg");

        Vec2F32 src[3];
        Vec2F32 dst[3];

        src[0]=Vec2F32(0,0);
        src[1]=Vec2F32(m.getDomain()(0),0);
        src[2]=Vec2F32(0,m.getDomain()(1));

        dst[0]=Vec2F32(m.getDomain()(0)*0.1,m.getDomain()(1)*0.1);
        dst[1]=Vec2F32(m.getDomain()(0)*0.8,m.getDomain()(1)*0.1);
        dst[2]=Vec2F32(m.getDomain()(0)*0.4,m.getDomain()(1)*0.6);
        Mat2x33F32 maffine = affine2D(src,dst);

        transformAffine2D(m,maffine).display();
     \endcode
    */
    static pop::Mat2x33F32 affine2D(const Vec2F32 src[3], const Vec2F32 dst[3],bool isfastinversion=true);


    /*!
     * \brief projective transformation from 4 pairs of the corresponding points
     * \param src 4 source points
     * \param dst 4 destination points
     * \param isfastinversion true=gaussian elimintation, false=school way
     *
     *
     * \code
    Mat2RGBUI8 m;
    m.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Vec2F32 src[4];
    Vec2F32 dst[4];
    src[0]=Vec2F32(0,0);
    src[1]=Vec2F32(m.getDomain()(0),0);
    src[2]=Vec2F32(0,m.getDomain()(1));
    src[3]=Vec2F32(m.getDomain()(0),m.getDomain()(1));
    dst[0]=Vec2F32(m.getDomain()(0)*0.1,m.getDomain()(1)*0.1);
    dst[1]=Vec2F32(m.getDomain()(0)*0.9,m.getDomain()(1)*0.1);
    dst[2]=Vec2F32(m.getDomain()(0)*0.1,m.getDomain()(1)*0.9);
    dst[3]=Vec2F32(m.getDomain()(0)*0.8,m.getDomain()(1)*0.7);
    Mat2x33F32 mproj = GeometricalTransformation::projective2D(src,dst);
    m =GeometricalTransformation::transformHomogeneous2D(mproj,m);
    Draw::circle(m,dst[0],10,RGBUI8::randomRGB(),2);
    Draw::circle(m,dst[1],10,RGBUI8::randomRGB(),2);
    Draw::circle(m,dst[2],10,RGBUI8::randomRGB(),2);
    Draw::circle(m,dst[3],10,RGBUI8::randomRGB(),2);
    m.display("projective",true,false);
    m.save("../doc/image2/lenaprojective.png");
     * \endcode
     * \image html lenaprojective.png
    */
    static pop::Mat2x33F32 projective2D(const Vec2F32 src[4], const Vec2F32 dst[4],bool isfastinversion=true);


    /*!
     * \brief 2D shear matrix
     * \param theta_radian angle in radian
     * \param coordinate shear-axis
     * \return  Shear Mat2x22F32
     *
     *  Generate the 2D shear matrix from the shear of angle \a theta_radian \n
     *  For coordinate=0 we have,\f$S_x(\theta) = \left(\begin{array}{cc}
     *   1 & \sin \theta \\
     *   0  & 1 \\
     *  \end{array}\right)\f$, and so one
    */
    static pop::Mat2x22F32 shear2D(F32 theta_radian, int coordinate);

    /*!
     * \brief 2D shear matrix in homogeneous coordinates
     * \param theta_radian angle in radian
     * \param coordinate shear-axis
     * \return  Shear Mat2x33F32
     *
     *  Generate the 2D shear matrix from the rotation of angle \a theta_radian  in homogeneous coordinates:\n
     *  For coordinate=0 we have,\f$S_x(\theta) = \left(\begin{array}{ccc}
     *   1 & \sin \theta &0 \\
     *   0  & 1 &0\\
     *   0 & 0 & 1
     *  \end{array}\right)\f$, and so one
    */
    static pop::Mat2x33F32 shear2DHomogeneousCoordinate(F32 theta_radian, int coordinate);


    //@}
    //-------------------------------------
    //
    //! \name Apply transformation Matrices
    //@{
    //-------------------------------------
    /*!
     *  \brief affine transformation on matrix
     * \param f input matrix
     * \param maffine affine transformation matrix
     * \param automaticsize resize the image to include the whole initial image in the destination domain
     * \return output matrix
     *
     *
     * \code
    Mat2RGBUI8 m;
    m.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
    Vec2F32 src[3];
    Vec2F32 dst[3];
    src[0]=Vec2F32(0,0);
    src[1]=Vec2F32(m.getDomain()(0),0);
    src[2]=Vec2F32(0,m.getDomain()(1));
    dst[0]=Vec2F32(m.getDomain()(0)*0.1,m.getDomain()(1)*0.1);
    dst[1]=Vec2F32(m.getDomain()(0)*0.8,m.getDomain()(1)*0.3);
    dst[2]=Vec2F32(m.getDomain()(0)*0.4,m.getDomain()(1)*0.8);
    Mat2x33F32 mproj = GeometricalTransformation::affine2D(src,dst);
     std::cout<<mproj<<std::endl;
    std::cout<<mproj.inverse()<<std::endl;
    m =GeometricalTransformation::transformAffine2D(mproj,m);
    Draw::circle(m,dst[0],10,RGBUI8::randomRGB(),2);
    Draw::circle(m,dst[1],10,RGBUI8::randomRGB(),2);
    Draw::circle(m,dst[2],10,RGBUI8::randomRGB(),2);
    m.display("affine",true,false);
    m.save("../doc/image2/lenaffine.png");
    * \endcode
    * \image html lenaffine.png
    */
    template< typename Type>
    static MatN<2,Type> transformAffine2D(const pop::Mat2x33F32 & maffine,const MatN<2,Type> & f,bool automaticsize=false)
    {
        Vec2F32 domain(f.getDomain());
        Vec2F32 xmin(NumericLimits<F32>::maximumRange());
        Vec2F32 xmax(-NumericLimits<F32>::maximumRange());
        if(automaticsize==true){
            Vec2F32 x(0,0); x = GeometricalTransformation::transformAffine2D(maffine,x);xmin=minimum(xmin,x); xmax=maximum(xmax,x);
            x = Vec2F32(f.getDomain()(0),0); x = GeometricalTransformation::transformAffine2D(maffine,x);xmin=minimum(xmin,x); xmax=maximum(xmax,x);
            x = Vec2F32(0,f.getDomain()(1));x = GeometricalTransformation::transformAffine2D(maffine,x);xmin=minimum(xmin,x); xmax=maximum(xmax,x);
            x = Vec2F32(f.getDomain()(0),f.getDomain()(1)); x = GeometricalTransformation::transformAffine2D(maffine,x); xmin=minimum(xmin,x); xmax=maximum(xmax,x);
            domain  = xmax - xmin;
        }else{
            xmin = 0;
            xmax = f.getDomain();
        }
        MatN<2,Type> g(domain);
        typename MatN<2,Type>::IteratorEDomain it(g.getIteratorEDomain());
        Mat2x33F32 maffine_inverse;
        maffine_inverse = maffine.inverse();//inverse of affine matrix is still affine matrix !
        while(it.next()){
            Vec2F32 x(it.x()+xmin);
            x =  GeometricalTransformation::transformAffine2D(maffine_inverse,x);
            if(f.isValid(x))
                g(it.x())=f.interpolationBilinear(x);
        }
        return g;
    }
    /*!
     * \brief Projective transformation on matrix
     * \param f input matrix
     * \param mproj projection transformation matrix
     * \return output matrix with same domain as input matrix
     *
     *
     * \code
     * Mat2RGBUI8 m;
     * m.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
     * Vec2F32 src[4];
     * Vec2F32 dst[4];
     * src[0]=Vec2F32(0,0);
     * src[1]=Vec2F32(m.getDomain()(0),0);
     * src[2]=Vec2F32(0,m.getDomain()(1));
     * src[3]=Vec2F32(m.getDomain()(0),m.getDomain()(1));
     * dst[0]=Vec2F32(m.getDomain()(0)*0.1,m.getDomain()(1)*0.1);
     * dst[1]=Vec2F32(m.getDomain()(0)*0.9,m.getDomain()(1)*0.1);
     * dst[2]=Vec2F32(m.getDomain()(0)*0.1,m.getDomain()(1)*0.9);
     * dst[3]=Vec2F32(m.getDomain()(0)*0.8,m.getDomain()(1)*0.7);
     * Mat2x33F32 mproj = GeometricalTransformation::projective2D(src,dst);
     * m =GeometricalTransformation::transformHomogeneous2D(mproj,m);
     * Draw::circle(m,dst[0],10,RGBUI8::randomRGB(),2);
     * Draw::circle(m,dst[1],10,RGBUI8::randomRGB(),2);
     * Draw::circle(m,dst[2],10,RGBUI8::randomRGB(),2);
     * Draw::circle(m,dst[3],10,RGBUI8::randomRGB(),2);
     * m.display("projective",true,false);
     * m.save("../doc/image2/lenaprojective.png");
     * \endcode
     * \image html lenaprojective.png
    */
    template< typename Type>
    static MatN<2,Type> transformHomogeneous2D(const pop::Mat2x33F32 & mproj,const MatN<2,Type> & f)
    {
        MatN<2,Type> gtransform(f.getDomain());
        typename MatN<2,Type>::IteratorEDomain it(gtransform.getIteratorEDomain());
        Mat2x33F32 mproj_inverse;
        mproj_inverse = mproj.inverse();
        while(it.next()){
            Vec2F32 x(Vec2F32(it.x()));
            x =  GeometricalTransformation::transformHomogeneous2D(mproj_inverse,x);
            if(f.isValid(x))
                gtransform(it.x())=f.interpolationBilinear(x);
        }
        return gtransform;
    }
    /*!
    * \brief Homogenous transformation on a 2d vector
    * \param x 2d input vector
    * \param mhom  transformation matrix
    * \return output vector after the homogeneous transformation

   * \f$\begin{bmatrix} x'_0 \\ x'_1 \\ x'_2 \end{bmatrix} = \begin{bmatrix} m_{0,0} & m_{0,1} & m_{0,2} \\ m_{1,0} & m_{1,1} & m_{1,2} \\ m_{2,0} & m_{2,1} & m_{2,2}\end{bmatrix} \begin{bmatrix} x_0 \\ y_1 \\ 1 \end{bmatrix}.\f$
    * with output=\f$(x'_0/x'_2 ,x'_1/x'_2 )\f$ and x=\f$(x_0,x_1)\f$
    *
    */
    static inline Vec2F32 transformHomogeneous2D(const pop::Mat2x33F32 & mhom,const Vec2F32 & x)
    {
        //fast implementation
        const F32 norm = mhom._dat[6]*x(0) + mhom._dat[7]*x(1)+mhom._dat[8];
        return Vec2F32(    (mhom._dat[0]*x(0) + mhom._dat[1]*x(1)+mhom._dat[2])/norm,(mhom._dat[3]*x(0) + mhom._dat[4]*x(1)+mhom._dat[5])/norm);
    }
    /*!
    * \brief Affine transformation on a 2d vector
    * \param x 2d input vector
    * \param maffine affine transformation matrix
    * \return output vector after the affine transformation
    *
    * \f$\begin{bmatrix} x'_0 \\ x'_1 \\ 1 \end{bmatrix} = \begin{bmatrix} a_{0,0} & a_{0,1} & a_{0,2} \\ a_{1,0} & a_{1,1} & a_{1,2} \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_0 \\ y_1 \\ 1 \end{bmatrix}.\f$
    * with output=\f$(x'_0,x'_1)\f$ and x=\f$(x_0,x_1)\f$
    */
    static inline Vec2F32 transformAffine2D(const pop::Mat2x33F32 & maffine,const Vec2F32 & x)
    {
        return Vec2F32(    (maffine._dat[0]*x(0) + maffine._dat[1]*x(1)+maffine._dat[2]),(maffine._dat[3]*x(0) + maffine._dat[4]*x(1)+maffine._dat[5]));
    }

    /*!
    * \brief merge two matrices following the homogeneous matrix
    * \param f first image
    * \param g second image that will be transform by the   homogeneous matrix
    * \param mhom homogeneous matrix
    * \param trans global domain translation
    * \return merge of two images
    *
    *
    * \code
     std::string path= "/home/tariel/Dropbox/";
     Mat2RGBUI8 img3;
     img3.load(path+"photo1.jpg");
     Mat2RGBUI8 img4;
     img4.load(path+"photo2.jpg");
     Mat2RGBUI8 img5;
     img5.load(path+"photo3.jpg");
     Mat2RGBUI8 img6;
     img6.load(path+"photo4.jpg");
     std::vector<Mat2RGBUI8> vv;
     vv.push_back(img3);
     vv.push_back(img4);
     vv.push_back(img5);
     vv.push_back(img6);

     Mat2RGBUI8 panoimg =Feature::panoramic(vv);
     panoimg.save("../doc/image2/panoramic.png");
     panoimg.display();
    * \endcode
    * \image html panoramic.png
    */
    template< typename Type>
    static MatN<2,Type> mergeTransformHomogeneous2D(const pop::Mat2x33F32 & mhom,const MatN<2,Type> & f,const MatN<2,Type> & g,Vec2F32& trans)
    {
        Vec2F32 xmax(f.getDomain());
        pop::Mat2x33F32  mhominverse;
        mhominverse = mhom.inverse();
        Vec2F32 x(0,0);
        x = GeometricalTransformation::transformHomogeneous2D(mhominverse,x);
        trans=minimum(trans,x); xmax=maximum(xmax,x);

        x = Vec2F32(g.getDomain()(0),0);
        x = GeometricalTransformation::transformHomogeneous2D(mhominverse,x);
        trans=minimum(trans,x); xmax=maximum(xmax,x);

        x = Vec2F32(0,g.getDomain()(1));
        x = GeometricalTransformation::transformHomogeneous2D(mhominverse,x);
        trans=minimum(trans,x); xmax=maximum(xmax,x);

        x = Vec2F32(g.getDomain()(0),g.getDomain()(1));
        x = GeometricalTransformation::transformHomogeneous2D(mhominverse,x);
        trans=minimum(trans,x); xmax=maximum(xmax,x);

        MatN<2,Type> panoramic(xmax-trans);
        ForEachDomain2D(xit,panoramic){
            Vec2F32 x_trans = Vec2F32(xit)+trans;
            Vec2F32 xx  =  GeometricalTransformation::transformHomogeneous2D(mhom,x_trans);
            if(f.isValid(x_trans)==true)
                panoramic(xit) = f(x_trans);
            if(g.isValid(xx)==true)
                panoramic(xit)=g(xx);

        }
        return panoramic;
    }

    //@}
};
//@}
}

#endif // GEOMETRICALTRANSFORMATION_H
