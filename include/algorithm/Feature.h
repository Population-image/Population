#ifndef VISION_H
#define VISION_H

#include<vector>
#include<cmath>
#include"data/vec/VecN.h"
#include"data/functor/FunctorPDE.h"
#include"data/notstable/Descriptor.h"
#include"data/notstable/Ransac.h"
#include"algorithm/LinearAlgebra.h"
#include"algorithm/Statistics.h"
#include"algorithm/Processing.h"
#include"algorithm/Draw.h"

namespace pop
{
/*!
\defgroup Feature Feature
\ingroup Algorithm
\brief Matrix In -> Features (harris, Hough, SIFT,...)


*/
class POP_EXPORTS Feature
{
private:

    /*!
        \class pop::Feature
        \ingroup Feature
        \brief Feature detection for 2D/3D matrix
        \author Tariel Vincent
     *
     *  For feature extraction, OpenCV is the best library with platform optimization. But due to some constraints in industrial applications, I create some algorithms in population. For instance,
     * \code
        std::string path= std::string(POP_PROJECT_SOURCE_DIR)+"/image/";

        Mat2UI8 img1;
        img1.load(path+"Image1.jpg");
        //sift descriptor
        Pyramid<2,F32> pyramid1 = Feature::pyramidGaussian(img1);
        Vec<KeyPointPyramid<2> > keypoint1 = Feature::keyPointSIFT(pyramid1);
        Vec<Descriptor<KeyPointPyramid<2> > > descriptors1 = Feature::descriptorPyramidPieChart(pyramid1,keypoint1);

        Mat2UI8 img2;
        img2.load(path+"Image2.jpg");
        //sift descriptor
        Pyramid<2,F32> pyramid2 = Feature::pyramidGaussian(img2);
        Vec<KeyPointPyramid<2> > keypoint2 = Feature::keyPointSIFT(pyramid2);
        Vec<Descriptor<KeyPointPyramid<2> > > descriptors2 = Feature::descriptorPyramidPieChart(pyramid2,keypoint2);

        //match VP tree
        Vec<DescriptorMatch<Descriptor<KeyPointPyramid<2> > >   > match = Feature::descriptorMatchVPTree(descriptors1,descriptors2);

        //keep the best
        int nbr_math_draw = std::min((int)match.size(),30);
        match.erase(match.begin()+nbr_math_draw,match.end());
        Feature::drawDescriptorMatch(img1,img2,match,1).save("SIFT.png");
     * \endcode
     * \image html SIFT.png
     */
public:
    //-------------------------------------
    //
    //! \name Lines
    //@{
    //-------------------------------------

    /*!
    * \brief hough transformation  http://en.wikipedia.org/wiki/Hough_transform
    * \param binary input binary matrix
    * \return hough transformation normalized in the range [0-1]

    * \code
        Mat2UI8 m;
        m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/barriere.png"));
        Mat2UI8 edge = Processing::edgeDetectorCanny(m,2,0.5,5);
        edge.display("edge",false);
        Mat2F32 hough = Feature::transformHough(edge);
        hough.display("hough",false);
        std::vector< std::pair<Vec2F32, Vec2F32 > > v_lines = Feature::HoughToLines(hough,edge ,0.5);
        Mat2RGBUI8 m_hough(m);
        for(unsigned int i=0;i<v_lines.size();i++){
            Draw::line(m_hough,v_lines[i].first,v_lines[i].second,  RGBUI8(255,0,0),2);
        }
        m_hough.display();
    * \endcode
    * \image html hough.png
    */
    static inline Mat2F32 transformHough(Mat2UI8 binary)
    {
        F32 DEG2RAD=0.017453293f;
        //Create the accu
        F32 hough_h = ((sqrt(2.f) * (F32)(binary.sizeI()>binary.sizeJ()?binary.sizeI():binary.sizeJ())) / 2.f);
        int heigh = static_cast<int>(hough_h * 2.f); // -r -> +r
        int width = 180;
        Mat2F32 accu (heigh,width);
        F32 center_x = binary.sizeJ()/2.f;
        F32 center_y = binary.sizeI()/2.f;
        for(unsigned int i=0;i<binary.sizeI();i++){
            for(unsigned int j=0;j<binary.sizeJ();j++){
                if( binary(i,j) > 125){
                    for(int t=0;t<180;t++){
                        F32 r = ( (j- center_x) * std::cos((F32)t * DEG2RAD)) + ((i - center_y) * std::sin((F32)t * DEG2RAD));
                        r = r + hough_h;
                        unsigned int r_min= static_cast<unsigned int>(std::floor(r));
                        F32 weigh_min= 1-(r-r_min);
                        unsigned int r_max= r_min+1;
                        F32 weigh_max= 1-(r_max-r);
                        if(accu.isValid(r_min,t))
                            accu(r_min,t)+=weigh_min;
                        if(accu.isValid(r_max,t))
                            accu(r_max,t)+=weigh_max;
                    }
                }
            }
        }
        accu = Processing::greylevelRange(accu,0,1);
        return accu;
    }



    /*!
    * \brief get lines from the hough transformation  http://en.wikipedia.org/wiki/Hough_transform
    * \param hough transformation normalized in the range [0-1]
    * \param binary input binary matrix input binary matrix
    * \param threshold the minimum value to be considered as a line in the Hough tranformations
    * \param radius_maximum_neightborhood radius to find the local minima
    * \code
        Mat2UI8 m;
        m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/barriere.png"));
        Mat2UI8 edge = Processing::edgeDetectorCanny(m,2,0.5,5);
        edge.display("edge",false);
        Mat2F32 hough = Feature::transformHough(edge);
        hough.display("hough",false);
        std::vector< std::pair<Vec2F32, Vec2F32 > > v_lines = Feature::HoughToLines(hough,edge ,0.5);
        Mat2RGBUI8 m_hough(m);
        for(unsigned int i=0;i<v_lines.size();i++){
            Draw::line(m_hough,v_lines[i].first,v_lines[i].second,  RGBUI8(255,0,0),2);
        }
        m_hough.display();
    * \endcode
    * \image html hough.png
    */
    static inline std::vector< std::pair<Vec2F32, Vec2F32 > > HoughToLines(Mat2F32 hough,Mat2UI8 binary,  F32 threshold=0.7, F32 radius_maximum_neightborhood=4)
    {
        std::vector< std::pair<Vec2F32, Vec2F32 > > lines;
        F32 DEG2RAD=0.017453293f;
        Mat2F32::IteratorENeighborhood it=hough.getIteratorENeighborhood(radius_maximum_neightborhood,2);
        ForEachDomain2D(x,hough){
            if(hough(x) >= threshold){
                it.init(x);
                F32 value=hough(x);
                bool max_local=true;
                while(it.next()){
                    if(hough(it.x())>value){
                        max_local=false;
                        break;
                    }
                }
                if(max_local==true){
                    Vec2F32 x1,x2;
                    F32 radius  = static_cast<F32>(x(0));
                    F32 angle   = static_cast<F32>(x(1));

                    if(angle>45&&angle<135){
                        F32 value1=binary.sizeJ()/2.f;
                        x1(0) = (-cos(angle* DEG2RAD)*value1+ radius-hough.sizeI()/2)/sin(angle* DEG2RAD)+binary.sizeI()/2.f;
                        x1(1) = value1 + binary.sizeJ()/2.f;
                        x2(0) = (-cos(angle* DEG2RAD)*(-value1)+ radius-hough.sizeI()/2)/sin(angle* DEG2RAD)+binary.sizeI()/2.f;
                        x2(1) = (-value1) + binary.sizeJ()/2.f;
                    }else{
                        F32 value1=binary.sizeI()/2.f;
                        x1(1) = (-sin(angle* DEG2RAD)*value1+ radius-hough.sizeI()/2)/cos(angle* DEG2RAD)+binary.sizeJ()/2.f;
                        x1(0) = value1 + binary.sizeI()/2.f;
                        x2(1) = (-sin(angle* DEG2RAD)*(-value1)+ radius-hough.sizeI()/2)/cos(angle* DEG2RAD)+binary.sizeJ()/2.f;
                        x2(0) = -(value1) + binary.sizeI()/2.f;
                    }
                    lines.push_back(std::make_pair(x1,x2));
                }
            }
        }
        return lines;
    }
    //@}
    //-------------------------------------
    //
    //! \name Key points
    //@{
    //-------------------------------------

    /*!
    * \brief detection of Harris & Stephens's corners  http://en.wikipedia.org/wiki/Corner_detection
    * \param img input matrix
    * \param sigma gaussian filter factor of the hessian matrix
    * \param kappa sensitivity parameter

    * This algorithm can be use for 2D or 3D matrix.
    * \code
    * Mat2UI8 img;
    * img.load("lena.jpg");
    * Vec<KeyPoint<2> > v_harris = Feature::keyPointHarris(img);
    * Feature::drawKeyPointsCircle(img,v_harris,3).display();
    * \endcode
    * \image html lenaharris.jpg
    */
    template<int DIM,typename PixelType>
    static Vec<KeyPoint<DIM> > keyPointHarris(const MatN<DIM,PixelType> & img,F32 sigma = 2,F32 kappa=0.20)
    {

        MatN<DIM,F32> imgf(img);
        imgf = Processing::greylevelRange(imgf,0,1);
        FunctorPDE::HessianMatrix<> func_hessian;


        MatN<DIM,Mat2x<F32,DIM,DIM> >  img_hessian(imgf.getDomain());
        forEachFunctorBinaryFunctionE(imgf,img_hessian, func_hessian);
        typename MatN<DIM,F32>::IteratorEDomain itdomain(imgf.getIteratorEDomain());
        img_hessian = FunctorMatN::convolutionGaussian(img_hessian,itdomain,sigma,2*sigma);

        itdomain.init();
        while(itdomain.next()){
            Mat2x22F32 & m=img_hessian(itdomain.x());
            imgf(itdomain.x())= m.determinant()-kappa*m.trace()*m.trace();
        }
        typename MatN<DIM,F32>::IteratorENeighborhood itneigh(imgf.getIteratorENeighborhood(1,0));
        itdomain.init();
        Vec<KeyPoint<DIM> > v_maxima;
        while(itdomain.next()){
            if(imgf(itdomain.x())>0.00001){
                F32 value = imgf(itdomain.x());
                bool maxima=true;
                itneigh.init(itdomain.x());
                while(itneigh.next()&&maxima==true){
                    if(imgf(itneigh.x())>value){
                        maxima = false;
                    }
                }
                if(maxima==true){
                    KeyPoint<DIM> keypoint(itdomain.x());
                    v_maxima.push_back(keypoint);
                }
            }
        }
        return v_maxima;
    }

    /*!
    * \brief  SIFT algorithm with some personnal modification (mean orientation and descriptor independant of the dimension)
    * \param pyramid_gaussian gaussina pyramid
    * \param threshold_low_contrast contrast threshold value
    * \param ratio_edge_response radius of the sample
    * \return descriptor
    *
    *
    *
    */
    template<int DIM,typename PixelType>
    static Vec<KeyPointPyramid<DIM> > keyPointSIFT(const Pyramid<DIM,PixelType> & pyramid_gaussian,F32 threshold_low_contrast=0.04,F32 ratio_edge_response=10)
    {
        Pyramid<DIM,F32>         pyramid_difference=  Feature::pyramidDifference(pyramid_gaussian);
        Vec<KeyPointPyramid<DIM> > extrema   =   Feature::pyramidExtrema(pyramid_difference);
        Vec<KeyPointPyramid<DIM> > extrema2  =   Feature::pyramidExtremaAdjust(pyramid_difference,extrema);
        Vec<KeyPointPyramid<DIM> > extrema3  =   Feature::pyramidExtremaThresholdLowContrast(pyramid_difference,extrema2,threshold_low_contrast);
        Vec<KeyPointPyramid<DIM> > extrema4  =   Feature::pyramidExtremaThresholdEdgeResponse(pyramid_difference,extrema3,ratio_edge_response);
        return extrema4;
    }


    //@}
    //-------------------------------------
    //
    //! \name Descriptors
    //@{
    //-------------------------------------



    /*!
        * \brief  paper in preparation
        * \param pyramid_G input pyramid as  gaussian
        * \param keypoints key points in the pyramid
        * \param sigma gaussian factor
        * \param nbr_orientation number of orientations in the unit sphere
        * \param scale_factor_gaussien scale the gaussian factor with this parameter
        * \return descriptor
        *
        *
        * \image html SphereOfSphere.png
        */

    template<int DIM,typename PixelType>
    static Vec<Descriptor<KeyPointPyramid<DIM> > > descriptorPyramidPieChart(const Pyramid<DIM,PixelType>& pyramid_G,const Vec<KeyPointPyramid<DIM> >   & keypoints, F32 sigma=1.6,F32 scale_factor_gaussien=4,int nbr_orientation=6)
    {
        Vec<Descriptor<KeyPointPyramid<DIM> > > descriptors;


        const MatN<DIM+1,PixelType> & octaveinit = pyramid_G.octave(0);
        typename MatN<DIM,PixelType>::IteratorENeighborhood itn (pyramid_G.getLayer(0,0).getIteratorENeighborhood(sigma*2*3*scale_factor_gaussien,2));
        F32 k = std::pow( 2., 1. / (octaveinit.getDomain()(DIM)-pyramid_G.getNbrExtraLayerPerOctave()) );



        for(int index_extrema=0;index_extrema<(int)keypoints.size();index_extrema++){
            const KeyPointPyramid<DIM>  & keypoint =keypoints[index_extrema];
            int octave =keypoint.octave();
            F32 layer=round(keypoint.layer());
            MatN<DIM,PixelType> plane= pyramid_G.getLayer(octave,layer);
            itn.setDomainMatN(plane.getDomain());
            VecN<DIM,F32> x =keypoint.xInPyramid();
            itn.init(round(x));
            F32 sigma_layer = std::pow(k,keypoint.layer())*sigma*scale_factor_gaussien;
            VecN<DIM,F32> orientation = orientationMeanDoubleWeighByGradienMagnitudeAndGaussianCircularWindow(plane,itn,sigma_layer);
            itn.init(round(x));
            Mat2F32 data = pieChartInPieChartDoubleWeighByGradienMagnitudeAndGaussianCircularWindow(plane,itn,orientation,nbr_orientation,sigma_layer);
            Descriptor<KeyPointPyramid<DIM> >  descriptor;
            descriptor.keyPoint() = keypoint;
            descriptor.orientation() = orientation;
            descriptor.data() = data;
            descriptors.push_back(descriptor);
        }
        return descriptors;
    }
    template<int DIM,typename PixelType,typename TKeyPoint>
    static Vec<Descriptor<TKeyPoint> > descriptorPieChart(const MatN<DIM,PixelType>& f,const Vec<TKeyPoint >   & keypoints, F32 radius=10,int nbr_orientation=6)
    {
        MatN<DIM,F32> img(f);
        Vec<Descriptor<TKeyPoint> > descriptors;
        typename MatN<DIM,PixelType>::IteratorENeighborhood itn (img.getIteratorENeighborhood(radius*2,2));

        for(int index_extrema=0;index_extrema<(int)keypoints.size();index_extrema++){
            const TKeyPoint  & keypoint =keypoints[index_extrema];

            itn.init(round(keypoint.x()));
            VecN<DIM,F32> orientation = orientationMeanDoubleWeighByGradienMagnitudeAndGaussianCircularWindow(img,itn,radius);
            itn.init(round(keypoint.x()));
            Mat2F32 data = pieChartInPieChartDoubleWeighByGradienMagnitudeAndGaussianCircularWindow(img,itn,orientation,nbr_orientation,radius);
            Descriptor<TKeyPoint>  descriptor;
            descriptor.keyPoint() = keypoint;
            descriptor.orientation() = orientation;
            descriptor.data() = data;
            descriptors.push_back(descriptor);
        }
        return descriptors;
    }
    //@}
    //-------------------------------------
    //
    //! \name KeyPoints/Descriptors/MatchDescriptor filter
    //@{
    //-------------------------------------

    template<typename Descriptor>
    static Vec<Descriptor   > descriptorFilterNoOverlap(const Vec<Descriptor > & descriptors, F32 min_distance){

        KDTree<Descriptor::DIM,F32> kdtree;
        Vec<Descriptor   >  descriptorfilter;
        for(unsigned int i=0;i<descriptors.size();i++){
            VecN<Descriptor::DIM,F32> target = descriptors(i).x();
            F32 distance;
            VecN<Descriptor::DIM,F32> result;
            kdtree.search(target,result,distance);
            if(distance>min_distance){
                kdtree.addItem(target);
                descriptorfilter.push_back(descriptors(i));
            }
        }
        return descriptorfilter;
    }

    //@}
    //-------------------------------------
    //
    //! \name Matching descripors
    //@{
    //-------------------------------------
    template<typename Descriptor>
    static Vec<DescriptorMatch<Descriptor >   > descriptorMatchVPTree(const Vec<Descriptor > & descriptor1, const Vec<Descriptor > & descriptor2){
        VpTree<Descriptor > c;
        c.create(descriptor1);

        Vec<DescriptorMatch<Descriptor > > v_match;

        for(unsigned int i=0;i<descriptor2.size();i++){
            Vec<Descriptor > v_target;
            Vec<F32 > v_distance;
            c.search(descriptor2[i],2,v_target,v_distance);

            if(v_distance.size()>0){
                DescriptorMatch<Descriptor > match;
                match._d1 = v_target[0];
                match._d2 = descriptor2[i];
                match._error= v_distance[0];
                v_match.push_back(match);
            }
        }
        std::sort(v_match.begin(),v_match.end());
        return v_match;
    }
    template<typename Descriptor>
    static Vec<DescriptorMatch<Descriptor >   > descriptorMatchBruteForce(const Vec<Descriptor > & descriptor1, const Vec<Descriptor > & descriptor2){
        Vec<DescriptorMatch<Descriptor > > v_match;
        for(unsigned int j=0;j<descriptor1.size();j++){
            F32 distance =1000;
            int index=-1;
            for(unsigned int i=0;i<descriptor2.size();i++){
                const Descriptor& d1=descriptor1[j];
                const Descriptor& d2=descriptor2[i];
                F32 dist_temp = pop::distance(d1.data(),d2.data(),2);
                if(dist_temp<distance){
                    index = i;
                    distance =dist_temp;
                }
            }
            DescriptorMatch<Descriptor > match;
            match._d1 = descriptor1[j];
            match._d2 = descriptor2[index];
            match._error= distance;
            v_match.push_back(match);
        }
        std::sort(v_match.begin(),v_match.end());
        return v_match;
    }
    //@}
    //-------------------------------------
    //
    //! \name Pyramid facilities
    //@{
    //-------------------------------------

    /*!
    * \brief create the gaussian pyramid \f$L(x,y,\sigma)=G(x,y)*I(x,y)\f$
    * \param img input matrix
    * \param number_octave number of octaves (-1 means log(I.getDomain()) / log(2.) - 2) )
    * \param number_layers_per_octave number of layers per octave
    * \param sigma_init initial sigma factor for the gaussian convolution
    * \param sigma  sigma factor
    * \param number_extra_layer_per_octave number of extra layers by octave
    * \brief gaussian pyramid
    *
    * This algorithm can be use for 2D or 3D matrix.
    * The number of octave is often defined as round(log(static_cast<F32>(I.getDomain().minCoordinate())) / log(2.) - 2);
    * \code
    * Mat2UI8 imgc;
    * imgc.load("/home/vincent/Population/doc/matrix/Lena.pgm");
    * Pyramid<2,F32 >        pyramid_gaussian =   Feature::pyramidGaussian(imgc);
    * Pyramid<2,F32 >        pyramid_difference=  Feature::pyramidDifference(pyramid_gaussian);
    * Vec<KeyPointPyramid<2> > extrema   =   Feature::pyramidExtrema(pyramid_difference);
    * Feature::drawKeyPointsCircle(imgc,extrema).display("extrema");
    * \endcode
    *
    */
    template<int DIM,typename PixelType>
    static Pyramid<DIM,F32 > pyramidGaussian(const MatN<DIM,PixelType> & img,  F32 sigma=1.6,F32 sigma_init=0.5,int number_octave=-1,int number_layers_per_octave=3,int number_extra_layer_per_octave=3)
    {
        MatN<DIM,F32> I(img);

        I = Processing::greylevelRange(I,0,1);
        if(number_octave<1)
            number_octave = round(std::log(  (F32)img.getDomain().minCoordinate()) / std::log(2.) - 2);

        F32 sig_diff = std::sqrt( std::max(sigma * sigma - sigma_init * sigma_init, 0.0001f) );
        if(sig_diff>0.01)
            I = FunctorMatN::convolutionGaussian(I,sig_diff,std::ceil(2.5*sig_diff));


        Vec<F32> sig(number_layers_per_octave + number_extra_layer_per_octave);
        F32 k = std::pow( 2., 1. / number_layers_per_octave );
        for( int i = 1; i < number_layers_per_octave + number_extra_layer_per_octave; i++ )
        {
            F32 sig_i_minus_one= std::pow(k, static_cast<F32>(i-1))*sigma;
            F32 sig_i          = std::pow(k, static_cast<F32>(i)  )*sigma;
            sig[i] =std::sqrt(sig_i*sig_i - sig_i_minus_one*sig_i_minus_one);
        }

        Pyramid<DIM,F32 > pyramid;
        for(int i=0;i<number_octave;i++){
            if(i==0){
                pyramid.pushOctave(I.getDomain(),number_layers_per_octave + number_extra_layer_per_octave);
                pyramid.setLayer(0,0,I);
            }
            else{
                VecN<DIM,int> size(pyramid.octave(pyramid.nbrOctave()-1).getDomain()/2);//half size of the previous octave
                pyramid.pushOctave(size,number_layers_per_octave + number_extra_layer_per_octave);
                MatN<DIM,F32>  temp = pyramid.getLayer(pyramid.nbrOctave()-2,number_layers_per_octave);//last octave and last layer
                temp = GeometricalTransformation::subResolution(temp,2);
                pyramid.setLayer(pyramid.nbrOctave()-1,0,temp);
                temp = pyramid.getLayer(pyramid.nbrOctave()-1,0);


            }
            for(int j=1;j< number_layers_per_octave + number_extra_layer_per_octave;j++){
                MatN<DIM,F32>  temp = pyramid.getLayer(pyramid.nbrOctave()-1,j-1);
                typename MatN<DIM,F32>::IteratorEDomain it= temp.getIteratorEDomain();
                temp = FunctorMatN::convolutionGaussian(temp,it,sig[j],maximum(2.,std::ceil(sig[j]*2.5)));
                pyramid.setLayer(pyramid.nbrOctave()-1,j,temp);
            }
        }
        pyramid.setNbrExtraLayerPerOctave(number_extra_layer_per_octave);
        return pyramid;
    }

    /*!
    * \brief create the sigma-derivate of the gaussian pyramid by a simple difference \f$D(x,y,k_i \sigma) =L(x,y,k_{i+1}\sigma)-L(x,y,k_{i}\sigma)\f$
    * \param pyramid input gaussian pyramid
    * \return sigma-derivate of the gaussian pyramid


    */
    template<int DIM,typename PixelType>
    static Pyramid<DIM,PixelType> pyramidDifference(const Pyramid<DIM,PixelType> & pyramid){
        Pyramid<DIM,PixelType> pyramiddiff;
        for(unsigned int index_octave=0;index_octave<pyramid.nbrOctave();index_octave++){
            pyramiddiff.pushOctave(pyramid.octave(index_octave).getDomain(),pyramid.nbrLayers(index_octave) -1);
            for(unsigned int scale=0;scale<pyramiddiff.nbrLayers(index_octave);scale++){
                pyramiddiff.setLayer(pyramiddiff.nbrOctave()-1, scale, pyramid.getLayer(index_octave,scale+1)-pyramid.getLayer(index_octave,scale));
            }
            pyramiddiff.setNbrExtraLayerPerOctave(pyramid.getNbrExtraLayerPerOctave()-1);
        }
        return pyramiddiff;
    }
    /*!
    * \brief extract the extrema of the pyramid \f$\{(x,y,k_i):  D(x,y,k_i \sigma)\leq \min_{x'\in\{x-1,x,x+1\},y'\in\{y-1,y,y+1\},k'\in\{k_{i-1},k_{i},k_{i+1}\}  }D(x',y',k'\sigma) \}\f$
    * \param pyramid_DofG input sigma-derivate gaussian pyramid
    * \param contrast_threshold the contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
    * \param border do not extract the extrema in the border zone of the domain
    * \param nbr_layer_per_octave number of layers per octave
    * \return extrema of the pyramid
    */
    template<int DIM,typename PixelType>
    static  Vec<KeyPointPyramid<DIM> > pyramidExtrema(const Pyramid<DIM,PixelType>& pyramid_DofG,F32 contrast_threshold=0.04, int border=5,unsigned int nbr_layer_per_octave=3){
        F32 threshold = 0.5*contrast_threshold/(pyramid_DofG.nbrLayers(0)-2);
        Vec<KeyPointPyramid<DIM> >  extrema;
        for(unsigned int index_octave=0;index_octave<pyramid_DofG.nbrOctave();index_octave++){
            const MatN<DIM+1,PixelType> & foctave = pyramid_DofG.octave(index_octave);
            VecN<DIM+1,PixelType> xmin(border);
            xmin(DIM)=1;
            VecN<DIM+1,PixelType> xmax(foctave.getDomain()-1-border);
            xmax(DIM)= foctave.getDomain()(DIM)-2;

            typename MatN<DIM+1,PixelType>::IteratorERectangle itg (foctave.getIteratorERectangle(xmin,xmax));
            typename MatN<DIM+1,PixelType>::IteratorENeighborhood itn (foctave.getIteratorENeighborhood(1,0));
            while(itg.next()){
                F32 value = foctave(itg.x());
                if(std::abs(value)>threshold)
                {
                    if(value>0){
                        itn.init(itg.x());
                        bool bmax=true;
                        while(itn.next()){
                            if(foctave(itn.x())>value){
                                bmax=false;
                                break;
                            }
                        }
                        if(bmax==true)
                            extrema.push_back(KeyPointPyramid<DIM>(itg.x(),index_octave,nbr_layer_per_octave));
                    }
                    if(value<0){
                        itn.init(itg.x());
                        bool bmax=true;
                        while(itn.next()){
                            if(foctave(itn.x())<value){
                                bmax=false;
                                break;
                            }
                        }
                        if(bmax==true)
                            extrema.push_back(KeyPointPyramid<DIM>(itg.x(),index_octave,nbr_layer_per_octave));
                    }
                }
            }
        }
        return extrema;
    }



    /*!
    * \brief adjust extrema by Taylor expansion by an iterative process following this equation \f$\mathbf{x}=\frac{\partial^2D^{-1}}{\partial \mathbf{x}^2}\frac{\partial D}{\partial \mathbf{x}}\f$
    * \param pyramid_DofG input sigma-derivate gaussian pyramid
    * \param extrema extrema of the sigma-derivate gaussian pyramid
    * \param border do not extract the extrema in the border zone of the domain
    * \param max_iteration max number of iteration
    * \return extrema after the adjustements (the VecNs are located with float type for more precisions)
    */

    template<int DIM,typename PixelType>
    static   Vec<KeyPointPyramid<DIM> >  pyramidExtremaAdjust(const Pyramid<DIM,PixelType> & pyramid_DofG,const Vec<KeyPointPyramid<DIM> >  & extrema, int border=5, int max_iteration=5){

        Vec<KeyPointPyramid<DIM> > v_extrema_adjust;
        FunctorPDE::Gradient<> grad;
        FunctorPDE::HessianMatrix<> hessian;
        VecN<DIM+1,F32> ZeroCinq;
        ZeroCinq = 0.5;
        for(int index_extrema=0;index_extrema<(int)extrema.size();index_extrema++){
            if(index_extrema==2157){
                index_extrema++;
                index_extrema--;

            }
            int octave = extrema[index_extrema].octave();

            VecN<DIM+1,F32> x=extrema[index_extrema].xInPyramid();

            for(int local_iteration=0 ; local_iteration < max_iteration; local_iteration++ )
            {
                VecN<DIM+1,F32> v;
                VecN<DIM+1,int> xint = round(x);
                v= grad(pyramid_DofG.octave(octave),xint);
                Mat2x<F32,DIM+1,DIM+1> H =  hessian(pyramid_DofG.octave(octave),xint);
                VecN<DIM+1,F32> X;
                H = -H.inverse();
                X=H*v;
                x+=X;
                for(int i=0;i<DIM;i++)
                    if(round(x(i))<border||round(x(i))>pyramid_DofG.octave(octave).getDomain()(i)-1-border)
                        local_iteration= max_iteration;
                    if(round(x(DIM))<0||round(x(DIM))>=pyramid_DofG.octave(octave).getDomain()(DIM))
                    local_iteration= max_iteration;
                if(absolute(X).allInferior(ZeroCinq)&&local_iteration!= max_iteration){
                    local_iteration= max_iteration;

                    v_extrema_adjust.push_back(KeyPointPyramid<DIM>(x,octave));
                    if(v_extrema_adjust.size()==1602){
                        local_iteration++;

                    }
                    if(v_extrema_adjust.size()==1601){
                        local_iteration++;
                    }
                }
            }
        }
        return v_extrema_adjust;
    }

    /*!
    * \brief filter unstable extrema if  \f$D(\mathbf{x})+\frac{1}{2}\frac{\partial D^T}{\partial \mathbf{x}} \mathbf{x} <v\f$
    * \param pyramid_DofG input sigma-derivate gaussian pyramid
    * \param extrema extrema of the sigma-derivate gaussian pyramid
    * \param contrast_threshold contrast threshold value v
    * \return extrema without unstable extrema
    */

    template<int DIM,typename PixelType>
    static Vec<KeyPointPyramid<DIM> >   pyramidExtremaThresholdLowContrast(const Pyramid<DIM,PixelType> & pyramid_DofG,const Vec<KeyPointPyramid<DIM> >  & extrema, F32 contrast_threshold=0.04)
    {

        Vec<KeyPointPyramid<DIM> > v_extrema_adjust2;
        for(int index_extrema=0;index_extrema<(int)extrema.size();index_extrema++){
            VecN<DIM+1,F32> x=extrema[index_extrema].xInPyramid();
            int octave = extrema[index_extrema].octave();
            FunctorPDE::Gradient<FunctorPDE::PartialDerivateCentered> grad;
            VecN<DIM+1,int> xint(round(x));
            VecN<DIM+1,F32> D=    grad(pyramid_DofG.octave(octave),xint);
            if(absolute(0.5*productInner(D,x)+pyramid_DofG.octave(octave)(x))* (pyramid_DofG.nbrLayers(0)-2)>contrast_threshold*2)
                v_extrema_adjust2.push_back(KeyPointPyramid<DIM>(x,octave));
        }
        return v_extrema_adjust2;
    }
    /*!
    * \brief filter along edges using hessian matrix  \f$ \frac{\max_i \lambda_i}{\min_i\lambda_i}<v\f$ with \f$\lambda_i\f$ the eigen value of the Hessian matrix \f$\frac{\partial^2D}{\partial \mathbf{x}^2} \f$
    * \param pyramid_DofG input sigma-derivate gaussian pyramid
    * \param extrema extrema of the sigma-derivate gaussian pyramid
    * \param ratio_threshold ratio threshold
    * \return extrema without unstable extrema

    */
    template<int DIM,typename PixelType>
    static Vec<KeyPointPyramid<DIM> >  pyramidExtremaThresholdEdgeResponse(const Pyramid<DIM,PixelType>& pyramid_DofG,const Vec<KeyPointPyramid<DIM> >   & extrema, F32 ratio_threshold=10)
    {
        Vec<KeyPointPyramid<DIM> > v_extrema_adjust3;
        for(int index_extrema=0;index_extrema<(int)extrema.size();index_extrema++){
            VecN<DIM+1,F32> x=extrema[index_extrema].xInPyramid();
            int octave = extrema[index_extrema].octave();
            FunctorPDE::PartialDerivateSecondCentered  derivate;
            VecN<DIM+1,int> xint(round(x));
            F32 dxx = derivate.operator ()(pyramid_DofG.octave(octave),xint,0,0);
            F32 dyy = derivate.operator ()(pyramid_DofG.octave(octave),xint,1,1);
            F32 dxy = derivate.operator ()(pyramid_DofG.octave(octave),xint,1,0);
            F32 tr = dxx + dyy;
            F32 det = dxx * dyy - dxy * dxy;
            if( det > 0 && tr*tr*ratio_threshold< (ratio_threshold + 1)*(ratio_threshold + 1)*det )
                v_extrema_adjust3.push_back(KeyPointPyramid<DIM>(x,octave));
        }
        return v_extrema_adjust3;
    }
    //@}


    //-------------------------------------
    //
    //! \name Some applications
    //@{
    //-------------------------------------

    /*!
    * \brief  panoramic algorithm with SIFT algorithm
    * \param V_img_fromleft_toright vector of input matrices
    * \param mode_transformation transformation type to match image (0=affine, 1=projective)
    * \param distmax distance threshold value for determining when a data fits a model
    * \param number_match_point number of points to estimate the model (ransac)
    * \param min_overlap remove points to close each other following this distance min_overlap
    * \return panoramic

    * Note: The geometrical transformation to make the correspondance between images is a projective transformation
    \code
        std::string path= "D:/Users/vtariel/Downloads/";
        Mat2RGBUI8 img3;
        img3.load(path+"Image1.jpg");
        Mat2RGBUI8 img4;
        img4.load(path+"Image2.jpg");
        Vec<Mat2RGBUI8> vv;
        vv.push_back(img3);
        vv.push_back(img4);
        Mat2RGBUI8 panoimg = Feature::panoramic(vv);
        panoimg.display();
    \endcode
    \image html Image1.jpg
    \image html Image2.jpg
    \image html Panorama.jpg
    */
    template<typename PixelType>
    static MatN<2,PixelType> panoramic(const Vec<MatN<2,PixelType> >  & V_img_fromleft_toright,int mode_transformation=1,F32 distmax=1,unsigned int number_match_point=100,unsigned int min_overlap=20){
        typedef KeyPointPyramid<2> KeyPointAlgo;
        Vec<Vec<DescriptorMatch<Descriptor<KeyPointAlgo > > > >  matchs;
        for(unsigned int i=0;i<V_img_fromleft_toright.size()-1;i++){
            Pyramid<2,F32> pyramid1 = Feature::pyramidGaussian(V_img_fromleft_toright[i]);
            Vec<KeyPointAlgo > keypoint1 = Feature::keyPointSIFT(pyramid1);
            Vec<Descriptor<KeyPointAlgo > >descriptor1 = Feature::descriptorPieChart(V_img_fromleft_toright[i],keypoint1);
            Pyramid<2,F32> pyramid2 = Feature::pyramidGaussian(V_img_fromleft_toright[i+1]);
            Vec<KeyPointAlgo > keypoint2 = Feature::keyPointSIFT(pyramid2);
            Vec<Descriptor<KeyPointAlgo > >descriptor2 = Feature::descriptorPieChart(V_img_fromleft_toright[i+1],keypoint2);
            Vec<DescriptorMatch<Descriptor<KeyPointAlgo > > > match = Feature::descriptorMatchVPTree(descriptor1,descriptor2);
            if(number_match_point<match.size())
                match.erase(match.begin()+number_match_point,match.end());
            match = Feature::descriptorFilterNoOverlap(match,min_overlap);
            matchs.push_back(match);
        }
        Vec<Mat2x33F32 > v_hom;
        for(unsigned int i=0;i<matchs.size();i++){
            Vec<pop::GeometricalTransformationRANSACModel::Data> v_data;
            for(unsigned int j=0;j<matchs(i).size();j++){
                DescriptorMatch<Descriptor<KeyPointAlgo > > match_descriptor = matchs(i)(j);
                v_data.push_back( pop::GeometricalTransformationRANSACModel::Data(match_descriptor._d1.keyPoint().x(), match_descriptor._d2.keyPoint().x()  )  );
            }
            Vec<pop::AffineTransformationRANSACModel::Data> v_data_best;
            if(mode_transformation==0){
                pop::AffineTransformationRANSACModel model;
                ransacMaxDataFitModel(v_data,10000,distmax,model,v_data_best);
                v_hom.push_back(model.getTransformation());
            }else{
                pop::ProjectionTransformationRANSACModel model;
                ransacMaxDataFitModel(v_data,10000,distmax,model,v_data_best);
                v_hom.push_back(model.getTransformation());
            }
        }
        Mat2x33F32 Mmult = Mat2x33F32::identity();
        MatN<2,PixelType> panoramic(V_img_fromleft_toright[0]);
        for(unsigned int i=0;i<v_hom.size();i++){

            Mmult = v_hom[i]*Mmult;
            Vec2F32 trans;
            panoramic = GeometricalTransformation::mergeTransformHomogeneous2D(Mmult,panoramic,V_img_fromleft_toright[i+1],trans);
            Mmult*=GeometricalTransformation::translation2DHomogeneousCoordinate(trans);
        }
        return panoramic;
    }
    //@}
    //-------------------------------------
    //
    //! \name Visualization
    //@{
    //-------------------------------------

    /*!
    * \brief  draw circle where an extrema is located
    * \param f input matrix
    * \param key_points key points or descriptors
    * \param radius circle radius
    * \return output matrix
   */
    template<int DIM,typename PixelType,typename TKeyPoint>
    static MatN<DIM,RGBUI8>  drawKeyPointsCircle(const MatN<DIM,PixelType> & f,const Vec<TKeyPoint >    & key_points,int radius=2)
    {
        MatN<DIM,RGBUI8> h(f);
        for(unsigned int index_extrema=0;index_extrema<key_points.size();index_extrema++){
            VecN<DIM,F32> x= key_points[index_extrema].x();
            Draw::circle(h,round(x),radius,RGBUI8::randomRGB(),1);
        }
        return h;
    }

    /*!
    * \brief  draw arrow where an extrema is located with its orientation
    * \param f input matrix
    * \param descriptors descriptors
    * \param length arrow length
    * \param sigma sigma filter of the gaussian pyramid
    * \return output matrix
   */
    template<int DIM,typename PixelType,typename TDescriptor>
    static MatN<DIM,RGBUI8>  drawDescriptorArrow(const MatN<DIM,PixelType> & f,const Vec<TDescriptor >    & descriptors,int length=20,F32 sigma=1.6)
    {
        MatN<DIM,RGBUI8> h(f);
        for(unsigned int index_extrema=0;index_extrema<descriptors.size();index_extrema++){
            const TDescriptor & descriptor = descriptors[index_extrema];
            VecN<DIM,F32> x= descriptor.keyPoint().x();
            RGBUI8 color = RGBUI8::randomRGB();
            Draw::circle(h,round(x),sigma*descriptor.keyPoint().scale(),color,1);
            Draw::arrow(h,round(x),round(x)+descriptor.orientation()*length,color,1);

        }
        return h;
    }



    /*!
    * \brief  draw a line between matched descriptors
    * \param f input first matrix
    * \param g input second matrix
    * \param match_descriptor  matched descriptors
    * \param line_width  line width
    * \return matrix representing the matched descriptors
    */

    template<int DIM,typename PixelType,typename Descriptor>
    static MatN<DIM,RGBUI8>  drawDescriptorMatch(const MatN<DIM,PixelType> & f,const MatN<DIM,PixelType> & g,const Vec<DescriptorMatch<Descriptor >   >   & match_descriptor,unsigned int line_width=1)
    {
        MatN<DIM,RGBUI8> imgrgb =Draw::mergeTwoMatrixHorizontal(f,g);
        for(unsigned int i=0;i<match_descriptor.size();i++){
            const DescriptorMatch<Descriptor > & d = match_descriptor[i];
            VecN<DIM,int> x1(d._d1.keyPoint().x());
            VecN<DIM,int> x2(d._d2.keyPoint().x());
            x2(DIM-1)+=f.getDomain()(DIM-1);
            Draw::line(imgrgb,x1,x2,RGBUI8::randomRGB(),line_width);
        }
        return imgrgb;
    }

    //@}
    //-------------------------------------
    //
    //! \name Some usefull facilities
    //@{
    //-------------------------------------

    /*!
     * \brief principal orientation as mean of the orientations
     * \param v_orientions vector of orientations
     * \param weight vector of weights (by default the same weight for each orientaion)
     * \return principal orientation
     *
     *  The orientations must be normalized at 1
     *
     * \code
     * Vec<Vec2F32> v_orientation;
     * Vec2F32 p1(1,0);
     * v_orientation.push_back(p1/p1.norm());
     * Vec2F32 p2(0,1);
     * v_orientation.push_back(p2/p2.norm());
     * cout<<Statistics::orientationPrincipalMean(v_orientation)<<endl;
     * \endcode
    */
    template<int DIM,typename PixelType>
    static VecN<DIM,PixelType> orientationPrincipalMean(const Vec<VecN<DIM,PixelType> > & v_orientions, const Vec<PixelType> & weight=std::vector<PixelType >(0))
    {

        if(weight.size()==0){
            VecN<DIM,PixelType> sum_of_elems= std::accumulate(v_orientions.begin(),v_orientions.end(),VecN<DIM,PixelType>())*(1./v_orientions.size());
            return sum_of_elems*(1/sum_of_elems.norm());
        }
        else
        {
            VecN<DIM,PixelType>  sum_of_elems(0);
            typename std::vector<PixelType>::const_iterator jw=weight.begin();
            typename std::vector<VecN<DIM,PixelType> >::const_iterator jo=v_orientions.begin();
            for(;jo!=v_orientions.end();++jo,++jw)
                sum_of_elems += ((*jo)*(*jw));
            return sum_of_elems*(1./sum_of_elems.norm());
        }

    }
    /*!
     * \brief principal orientation as the most probable orientation
     * \param v_orientions vector of orientations
     * \param weight vector of weights (by default the same weight for each orientaion)
     * \param nbr_angle number of bins
     * \return principal orientation
     *
     *  The orientations must be normalized at 1
     *
     * \code
     * Vec<Vec2F32> v_orientation;
     * Vec2F32 p1(1,0);
     * v_orientation.push_back(p1/p1.norm());
     * Vec2F32 p2(0,1);
     * v_orientation.push_back(p2/p2.norm());
     * cout<<Statistics::orientationPrincipalHistogram(v_orientation)<<endl;
     * \endcode
    */
    template<typename PixelType>
    static VecN<2,PixelType> orientationPrincipalHistogram(const Vec<VecN<2,PixelType> > & v_orientions, const Vec<PixelType> & weight,int nbr_angle=36)
    {
        Vec<F32> repartition(nbr_angle);
        Vec<F32> angles;
        for(int i=0;i<nbr_angle;i++){
            angles.push_back(360.*i/nbr_angle-180);
        }
        Vec<int> v_affect;
        for(unsigned int i=0;i<v_orientions.size();i++){
            F32 angle =std::atan2(v_orientions(i)(0),v_orientions(i)(1))*180/pop::PI;
            std::vector<F32>::const_iterator  low=std::lower_bound (angles.begin(), angles.end(),angle );
            I32 indice = I32(low- angles.begin())-1 ;
            //std::cout<<indice<<std::endl;
            repartition[indice]+=weight(i);
            v_affect.push_back(indice);
        }
        I32 index_max = I32 (std::max_element(repartition.begin(),repartition.end())-repartition.begin());
        VecN<2,PixelType>  sum_of_elems(0);
        for(unsigned int i=0;i<v_orientions.size();i++){
            if(v_affect[i]==index_max)
                sum_of_elems += (v_orientions[i]*weight[i]);
        }
        return sum_of_elems*(1./sum_of_elems.norm());
    }
    template<int DIM,typename PixelType,typename IteratorNeigh>
    static VecN<DIM,F32>   orientationMeanDoubleWeighByGradienMagnitudeAndGaussianCircularWindow(const MatN<DIM,PixelType>& f, IteratorNeigh& itn,F32 sigma)
    {
        FunctorPDE::Gradient<> gradient;
        Vec<F32>  gradient_magnitude;
        Vec< VecN<DIM,F32> >  gradient_orientation;
        F32 gaussian_coefficient_exp = -1/(2 * sigma * sigma);
        while(itn.next()){
            VecN<DIM,F32> g = gradient(f,itn.x());
            F32 norm = g.norm();
            F32 radius = itn.xWithoutTranslation().normPower();
            radius*=gaussian_coefficient_exp;
            gradient_magnitude.push_back(std::exp(radius)*norm);
            if(norm!=0)
                gradient_orientation.push_back(g*(1/norm));
            else
                gradient_orientation.push_back(g);
        }
        return orientationPrincipalMean(gradient_orientation,gradient_magnitude);
    }
    template<int DIM,typename PixelType,typename IteratorNeigh>
    static Mat2F32   pieChartInPieChartDoubleWeighByGradienMagnitudeAndGaussianCircularWindow(const MatN<DIM,PixelType>& f, IteratorNeigh& itn,VecN<DIM,F32> &orientation,int nbr_orientation, F32 sigma)
    {
        FunctorPDE::Gradient<> gradient;
        PieChartInPieChart<DIM> disk(orientation,nbr_orientation);
        F32 gaussian_coefficient_exp = -1/(2 * sigma * sigma);
        while(itn.next()){
            VecN<DIM,F32> orientationglobal =itn.xWithoutTranslation();
            VecN<DIM,F32> orientationlocal = gradient(f,itn.x());
            F32 norm = normValue(orientationlocal);
            F32 radius = itn.xWithoutTranslation().normPower();
            radius*=gaussian_coefficient_exp;
            F32 weight =norm*std::exp(radius);
            disk.addValue(orientationglobal,orientationlocal,weight);
        }
        disk.normalize();
        return disk._data;
    }
    //@}

};




}
#endif // VISION_H
