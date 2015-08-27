#ifndef ARTICLEGERMGRAIN_H
#define ARTICLEGERMGRAIN_H
#include"Population.h"
using namespace pop;//Population namespace
#include<map>


void SectionI_GaussianField(){
    Mat2UI8 img;//2d grey-level image object
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/CTG.pgm"));//replace this path by those on your computer

    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    int value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//segmentation
    threshold.display("initial",false);
    threshold = Processing::greylevelRemoveEmptyValue(threshold);//pixel value of the pore space = 0 and  pixel value of the matrix space = 1
    Mat2F32 m_field_field_correlation_experimental =Analysis::correlation(threshold,200);//2-point correlation function
    m_field_field_correlation_experimental = m_field_field_correlation_experimental.deleteCol(1);//remove the pore-pore correlation function in the first column
    m_field_field_correlation_experimental.saveAscii("CTG_field_field_correlation_experimental.m");
    Mat3F32 m_gaussian_field;
    Mat3UI8 m_U_bin =RandomGeometry::gaussianThesholdedRandomField(m_field_field_correlation_experimental,128,m_gaussian_field);
    m_U_bin = Processing::greylevelRemoveEmptyValue(m_U_bin);
    Mat2F32 m_field_field_correlation_model = Analysis::correlation(m_U_bin,200);
    m_field_field_correlation_model = m_field_field_correlation_model.deleteCol(1);//remove the pore-pore correlation function in the first column
    m_field_field_correlation_model.saveAscii("CTG_field_field_correlation_model.m");
    //test the correlation match
    for(unsigned int i= 0; i<10;i++){
        std::cout<<i<<" "<<m_field_field_correlation_experimental(i,1)<<" "<<m_field_field_correlation_model(i,1) <<std::endl;
    }

    Scene3d scene;
    Visualization::marchingCubeSmooth(scene,m_U_bin);
    Visualization::lineCube(scene,m_U_bin);
    scene.display();
}

void SectionI_Annealing(){
    Mat2UI8 threshold;//2d grey-level image object
    threshold.load(POP_PROJECT_SOURCE_DIR+std::string("/image/meb_A_675_7j.png"));
    threshold = GeometricalTransformation::subResolution(threshold,2);
    threshold = Processing::greylevelRemoveEmptyValue(threshold);
    Visualization::labelToRandomRGB(threshold).display("initial",false,false);
    Mat2F32 volume_fraction = Analysis::histogram(threshold);
    //2D case
    {

        //Vec2I32 v(256,256);
        //Mat2UI8 random = RandomGeometry::randomStructure(v,volume_fraction);
        //RandomGeometry::annealingSimutated(random,threshold);
        //Visualization::labelToRandomRGB(random).display("2d reconstruction",false,false);
    }
    //3D case (expensive process)
    {
        std::cout<<"Long computation time around 2 hours"<<std::endl;
        Vec3I32 v(128,128,128);
        Mat3UI8 random = RandomGeometry::randomStructure(v,volume_fraction);
        RandomGeometry::annealingSimutated(random,threshold,8,60);
        Mat2F32 m_field_field_annealing_model = Analysis::correlation(random,60);
        m_field_field_annealing_model = m_field_field_annealing_model.deleteCol(1);//remove the pore-pore correlation function in the first column
        m_field_field_annealing_model.saveAscii("CTG_field_field_annealing_model.m");

        random.save("annealing.pgm");
        Scene3d scene;
        Visualization::cubeExtruded(scene,random);
        Visualization::lineCube(scene,random);
        scene.display();
    }
}

void SectionII_PoissonPointProcess()
{
    {
        Vec2F32 domain(512,512);//2d field domain
        double lambda= 0.01;// parameter of the Poisson Point process
        ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson Point process
        DistributionDirac d (2);//because the Poisson Point process has a surface equal to 0, we associate each point with mono-disperse sphere to display the result
        RandomGeometry::sphere(grain,d);
        Mat2UI8 img = RandomGeometry::continuousToDiscrete(grain);
        img = img.opposite();
        img.save("UPPP.png");
        img.display();
    }
    {
        Mat2UI8 img;
        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
        Mat2F32 imgf(img);
        imgf=imgf/255.;
        imgf=imgf*0.07;
        ModelGermGrain2 grain = RandomGeometry::poissonPointProcessNonUniform(imgf);
        DistributionDirac d(2);
        RandomGeometry::sphere(grain,d);
        img = RandomGeometry::continuousToDiscrete(grain);
        img = img.opposite();
        img.save("NUPPP.png");
        img.display();
    }
}
void SectionII_MaternFilter()
{
    {
        Vec2F32 domain(512,512);//2d field domain
        double lambda= 0.1;// parameter of the Poisson Point process
        ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson Point process
        RandomGeometry::hardCoreFilter(grain,20);
        DistributionDirac d(10);
        RandomGeometry::sphere(grain,d);
        Mat2UI8 img = RandomGeometry::continuousToDiscrete(grain);
        img = img.opposite();
        img.save("HardCore.png");
        img.display();
    }
    {
        Vec3F32 domain(256,256,256);//2d field domain
        double lambda= 0.01;// parameter of the Poisson Point process
        ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 3d Poisson Point process
        RandomGeometry::hardCoreFilter(grain,40);
        DistributionDirac d(20);
        RandomGeometry::sphere(grain,d);
        Mat3UI8 img = RandomGeometry::continuousToDiscrete(grain);
        Scene3d scene;
        pop::Visualization::marchingCube(scene,img);
        scene.setTransparentMode(true);
        scene.setTransparencyAllGeometricalFigure(40);
        pop::Visualization::lineCube(scene,img);
        scene.display();
    }
}
void SectionII_MinOverlap(){
    Vec2F32 domain(512,512);//2d field domain
    double lambda= 0.003;// parameter of the Poisson Point process
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson Point process
    RandomGeometry::minOverlapFilter(grain,20);
    DistributionDirac d(10.5);
    RandomGeometry::sphere(grain,d);
    Mat2UI8 img = RandomGeometry::continuousToDiscrete(grain);
    img = img.opposite();
    img.save("MinOverlap.png");
    img.display();
}
void SectionII_Voronoi(){
    {
        Vec2F32 domain(512,512);
        ModelGermGrain2 germ = RandomGeometry::poissonPointProcess(domain,0.01);
        Mat2UI32 label = RandomGeometry::germToMatrix(germ);
        label = Processing::voronoiTesselationEuclidean(label);
        Mat2RGBUI8 voronoicolor =  Visualization::labelToRandomRGB(label);
        voronoicolor.save("Voronoi3d.png");
        voronoicolor.display();
    }
    {
        Vec3F32 domain(200,200,200);
        ModelGermGrain3 germ = RandomGeometry::poissonPointProcess(domain,0.0001);
        Mat3UI32 label = RandomGeometry::germToMatrix(germ);
        label = Processing::voronoiTesselationEuclidean(label);
        Mat3RGBUI8 voronoicolor =  Visualization::labelToRandomRGB(label);
        Scene3d scene;
        Visualization::cube(scene,voronoicolor);
        pop::Visualization::lineCube(scene,voronoicolor);
        scene.display();
    }
}



void SectionII_EulerAngle(){

    GrainBox<3> * box = new GrainBox<3>();
    box->radius = Vec3F32(20,30,10);
    box->x      = Vec3F32(50,50,50);
    ModelGermGrain3 germ_grain;
    germ_grain.setDomain(Vec3F32(100,100,100));
    germ_grain.grains().push_back(box);

    Mat3UI8 img = RandomGeometry::continuousToDiscrete(germ_grain);
    Scene3d scene;
    Visualization::marchingCube(scene,img);
    pop::Visualization::lineCube(scene,img);
    scene.display(false);
    waitKey("Press the key to go to the next figure");



    box->orientation.setAngle_ei(PI/6,0);
    img = RandomGeometry::continuousToDiscrete(germ_grain);
    scene.lock();
    scene.clear();
    Visualization::marchingCube(scene,img);
    pop::Visualization::lineCube(scene,img);
    scene.unlock();
    scene.display(false);
    waitKey("Press the key to go to the next figure");

    box->orientation.setAngle_ei(PI/6,1);
    img = RandomGeometry::continuousToDiscrete(germ_grain);
    scene.lock();
    scene.clear();
    Visualization::marchingCube(scene,img);
    pop::Visualization::lineCube(scene,img);
    scene.unlock();
    scene.display(false);
    waitKey("Press the key to go to the next figure");


    box->orientation.setAngle_ei(PI/6,2);
    img = RandomGeometry::continuousToDiscrete(germ_grain);
    scene.lock();
    scene.clear();
    Visualization::marchingCube(scene,img);
    pop::Visualization::lineCube(scene,img);
    scene.unlock();
    scene.display(false);
    waitKey("Press the key to go to the next figure");

}

void SectionII_ProbabilityDistributionDirection_Fix(){


    Vec3F32 domain(128,128,128);
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,0.00005);//generate the 2d Poisson point process

    DistributionMultiVariateProduct d_radius(DistributionDirac(20),DistributionDirac(20),DistributionDirac(4));
    DistributionMultiVariateProduct d_angle(DistributionDirac(0),DistributionDirac(PI/4),DistributionDirac(0));
    RandomGeometry::box(grain,d_radius,d_angle);
    Mat3UI8 lattice = RandomGeometry::continuousToDiscrete(grain);

    Scene3d scene;
    pop::Visualization::marchingCube(scene,lattice);
    scene.setColorAllGeometricalFigure(RGBUI8(100,100,100));
    pop::Visualization::lineCube(scene,lattice);
    scene.display();
}
void SectionII_ProbabilityDistributionDirection_Random(){


    Vec3F32 domain(128,128,128);
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,0.00005);//generate the 2d Poisson point process

    DistributionMultiVariateProduct d_radius(DistributionDirac(20),DistributionDirac(20),DistributionDirac(4));

    DistributionMultiVariateProduct d_angle(DistributionUniformInt(0,PI*2),DistributionUniformInt(0,PI*2),DistributionUniformInt(0,PI*2));

    RandomGeometry::box(grain,d_radius,d_angle);
    Mat3UI8 lattice = RandomGeometry::continuousToDiscrete(grain);

    Scene3d scene;
    pop::Visualization::marchingCube(scene,lattice);
    scene.setColorAllGeometricalFigure(RGBUI8(100,100,100));
    pop::Visualization::lineCube(scene,lattice);
    scene.display();
}
void SectionII_ProbabilityDistributionRadius_Normal(){

    double porosity_volume_fraction=0.6;
    DistributionNormal d(20,10);
    double grain_volume_expectation = 4./3.*pop::PI*Statistics::moment(d,3,0,50);
    double lambda = -std::log(porosity_volume_fraction)/grain_volume_expectation;


    Vec3F32 domain(256,256,256);//3d field domain
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 3d Poisson point process
    RandomGeometry::sphere(grain,d);
    Mat3UI8 lattice = RandomGeometry::continuousToDiscrete(grain);

    Scene3d scene;
    pop::Visualization::marchingCubeSmooth(scene,lattice);
    scene.setColorAllGeometricalFigure(RGBUI8(100,100,100));
    pop::Visualization::lineCube(scene,lattice);
    scene.display();
}
void SectionII_ProbabilityDistributionRadius_Power(){
    double porosity_volume_fraction=0.6;
    DistributionExpression dexp("1/x^(4.1)");
    DistributionRegularStep d= Statistics::toProbabilityDistribution(dexp,5,512,0.1);
    double grain_volume_expectation = 4./3.*pop::PI*Statistics::moment(d,3,0,50);
    double lambda = -std::log(porosity_volume_fraction)/grain_volume_expectation;


    Vec3F32 domain(256,256,256);//3d field domain
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 3d Poisson point process
    RandomGeometry::sphere(grain,d);
    Mat3UI8 lattice = RandomGeometry::continuousToDiscrete(grain);
    Scene3d scene;
    pop::Visualization::marchingCubeSmooth(scene,lattice);
    scene.setColorAllGeometricalFigure(RGBUI8(100,100,100));
    pop::Visualization::lineCube(scene,lattice);
    scene.display();
}
void SectionII_Sphere(){
    GrainSphere<3> *sphere = new GrainSphere<3>;

    double size=100;
    sphere->x = Vec3F32(size/2,size/2,size/2);
    sphere->radius = size/2;

    ModelGermGrain3 germgrain;
    germgrain.setDomain(Vec3F32(size,size,size));
    germgrain.grains().push_back(sphere);

    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
    Scene3d scene;
    Visualization::marchingCubeSmooth(scene,m);
    Visualization::lineCube(scene,m);
    scene.display();
}
void SectionII_Ellipse(){
    GrainEllipsoid<3> *ellipse = new GrainEllipsoid<3>;
    double size=100;
    ellipse->x = Vec3F32(size/2,size/2,size/2);
    ellipse->setRadius(Vec3F32(size/2,size/4,size/8));
    ModelGermGrain3 germgrain;
    germgrain.setDomain(Vec3F32(size,size,size));
    germgrain.grains().push_back(ellipse);
    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
    Scene3d scene;
    Visualization::marchingCubeSmooth(scene,m);
    Visualization::lineCube(scene,m);
    scene.display();
}
void SectionII_Cylinder(){
    GrainCylinder *cylinder = new GrainCylinder;
    double size=100;
    cylinder->x = Vec3F32(size/2,size/2,size/2);
    cylinder->radius = size/8;
    cylinder->height = size*3./4.;
    ModelGermGrain3 germgrain;
    germgrain.setDomain(Vec3F32(size,size,size));
    germgrain.grains().push_back(cylinder);
    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
    Scene3d scene;
    Visualization::marchingCube(scene,m);
    Visualization::lineCube(scene,m);
    scene.display();
}
void SectionII_Box(){
    GrainBox<3> *box = new GrainBox<3>;
    double size=100;
    box->x = Vec3F32(size/2,size/2,size/2);
    box->radius = Vec3F32(size/2,size/4,size/8);
    ModelGermGrain3 germgrain;
    germgrain.setDomain(Vec3F32(size,size,size));
    germgrain.grains().push_back(box);
    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
    Scene3d scene;
    Visualization::marchingCube(scene,m);
    Visualization::lineCube(scene,m);
    scene.display();
}
void SectionII_Rhombohedra(){
    GrainEquilateralRhombohedron *rhombohedra = new GrainEquilateralRhombohedron;
    double size=100;
    rhombohedra->x = Vec3F32(size/2,size/2,size/2);
    rhombohedra->radius = size/6;
    rhombohedra->setAnglePlane(30.*pop::PI/180.);
    ModelGermGrain3 germgrain;
    germgrain.setDomain(Vec3F32(size,size,size));
    germgrain.grains().push_back(rhombohedra);
    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
    Scene3d scene;
    Visualization::marchingCubeSmooth(scene,m);
    Visualization::lineCube(scene,m);
    scene.display();
}
void SectionII_RegularTetrahedron(){
    GrainPolyhedra<3> *polyhedra = new GrainPolyhedra<3>;

    //The following Cartesian coordinates define the four vertices of a tetrahedron with edge-length 2, centered at the origin:
    Vec3F32 x1( 1, 0, -1./std::sqrt(2.));
    Vec3F32 x2(-1, 0, -1./std::sqrt(2.));
    Vec3F32 x3( 0, 1,  1./std::sqrt(2.));
    Vec3F32 x4( 0,-1,  1./std::sqrt(2.));

    double size=100;

    Vec3F32 normal_x1_x2_x3 = productVectoriel(x1-x2,x1-x3);
    normal_x1_x2_x3 *=sgn( productInner(normal_x1_x2_x3,x1));
    normal_x1_x2_x3/=normal_x1_x2_x3.norm();
    polyhedra->addPlane(size/6,normal_x1_x2_x3);

    Vec3F32 normal_x1_x3_x4 = productVectoriel(x1-x3,x1-x4);
    normal_x1_x3_x4 *=sgn( productInner(normal_x1_x3_x4,x1));
    normal_x1_x3_x4/=normal_x1_x3_x4.norm();
    polyhedra->addPlane(size/6,normal_x1_x3_x4);


    Vec3F32 normal_x1_x4_x2 = productVectoriel(x1-x4,x1-x2);
    normal_x1_x4_x2 *=sgn( productInner(normal_x1_x4_x2,x1));
    normal_x1_x4_x2/=normal_x1_x4_x2.norm();
    polyhedra->addPlane(size/6,normal_x1_x4_x2);

    Vec3F32 normal_x2_x3_x4 = productVectoriel(x2-x3,x2-x4);
    normal_x2_x3_x4 *=sgn( productInner(normal_x2_x3_x4,x2));
    normal_x2_x3_x4/=normal_x2_x3_x4.norm();
    polyhedra->addPlane(size/6,normal_x2_x3_x4);


    polyhedra->x = Vec3F32(size/2,size/2,size/2);

    ModelGermGrain3 germgrain;
    germgrain.setDomain(Vec3F32(size,size,size));
    germgrain.grains().push_back(polyhedra);
    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
    Scene3d scene;
    Visualization::marchingCube(scene,m);
    Visualization::lineCube(scene,m);
    scene.display();
}
void SectionII_Model_Boolean(){
    Mat2RGBUI8 img;
    img.load(std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp");
    img= GeometricalTransformation::scale(img,Vec2F32(1024./img.getDomain()(0)));
    Vec2F32 domain;
    domain= img.getDomain();
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.05);//generate the 2d Poisson point process
    grain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_BOUNDED);
    DistributionExpression d ("1/x^3");
    DistributionRegularStep d_prob = Statistics::toProbabilityDistribution(d,5,60);
    RandomGeometry::sphere(grain, d_prob);
    RandomGeometry::RGBFromMatrix(grain,img);
    grain.setModel( MODEL_BOOLEAN);
    Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain);
    aborigenart.display("art");
    aborigenart.save("art_boolean.jpg");
    // aborigenart.save("/home/vincent/Desktop/Population/doc/image/AborigenLena.bmp");
}

void SectionII_Model_DeadLeave(){
    Mat2RGBUI8 img;
    img.load(std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp");
    img= GeometricalTransformation::scale(img,Vec2F32(1024./img.getDomain()(0)));
    Vec2F32 domain;
    domain= img.getDomain();
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.5);//generate the 2d Poisson point process
    grain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_BOUNDED);
    DistributionExpression d ("1/x^3");
    DistributionRegularStep d_prob = Statistics::toProbabilityDistribution(d,5,256);
    RandomGeometry::sphere(grain, d_prob);
    RandomGeometry::RGBFromMatrix(grain,img);
    grain.setModel( MODEL_DEADLEAVE);
    Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain);
    aborigenart.display("art");
    aborigenart.save("art_deadleave.jpg");
    // aborigenart.save("/home/vincent/Desktop/Population/doc/image/AborigenLena.bmp");
}

void SectionII_Model_Transparency(){
    Mat2RGBUI8 img;
    img.load(std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp");
    img= GeometricalTransformation::scale(img,Vec2F32(1024./img.getDomain()(0)));
    Vec2F32 domain;
    domain= img.getDomain();
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.5);//generate the 2d Poisson point process
    grain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_BOUNDED);
    DistributionExpression d ("1/x^3");
    DistributionRegularStep d_prob = Statistics::toProbabilityDistribution(d,5,256);
    RandomGeometry::sphere(grain, d_prob);
    RandomGeometry::RGBFromMatrix(grain,img);
    grain.setModel( MODEL_TRANSPARENT);
    grain.setTransparency(0.5);
    Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain);
    aborigenart.display("art");
    aborigenart.save("art_transparency.jpg");
    // aborigenart.save("/home/vincent/Desktop/Population/doc/image/AborigenLena.bmp");
}
void SectionII_Model_ShotNoise(){
    Mat2RGBUI8 img;
    img.load(std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp");
    img= GeometricalTransformation::scale(img,Vec2F32(1024./img.getDomain()(0)));
    Vec2F32 domain;
    domain= img.getDomain();
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.015);//generate the 2d Poisson point process
    grain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_BOUNDED);
    DistributionExpression d ("1/x^3");
    DistributionRegularStep d_prob = Statistics::toProbabilityDistribution(d,5,256);
    RandomGeometry::sphere(grain, d_prob);
    img/=4;
    RandomGeometry::RGBFromMatrix(grain,img);
    grain.setModel( MODEL_SHOTNOISE);
    Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain);
    aborigenart.display("art");
    aborigenart.save("art_shotnoise.jpg");
    // aborigenart.save("/home/vincent/Desktop/Population/doc/image/AborigenLena.bmp");
}
void SectionII_Model_LatticeToContinious(){
    F32 porosity=0.6;
    F32 radiusmin=10;
    F32 radiusmax=20;
    DistributionUniformReal duniform_radius(radiusmin,radiusmax);

    F32 angle=15*PI/180;
    DistributionDirac ddirar_angle(angle);

    F32 moment_order3 = pop::Statistics::moment(duniform_radius,3,0,40);
    //8*E^3(R)/E^3(std::cos(theta))
    F32 volume_expectation = moment_order3*8./(std::pow(std::cos(angle),3.0));
    Vec3F32 domain(256);//3d field domain
    F32 lambda=-std::log(porosity)/std::log(2.718)/volume_expectation;
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process

    RandomGeometry::rhombohedron(grain,duniform_radius,ddirar_angle);
    int time1 =time(NULL);
    Mat3UI8 lattice = RandomGeometry::continuousToDiscrete(grain);
    std::cout<<"time execution "<<time(NULL)-time1 <<std::endl;
    lattice.save("big_size.pgm");

    Mat2F32 m=  Analysis::histogram(lattice);
    std::cout<<"Realization porosity"<<m(0,1)<<std::endl;

    Scene3d scene;
    Visualization::marchingCube(scene,lattice);
    scene.setColorAllGeometricalFigure(RGBUI8(100,100,100));
    Visualization::lineCube(scene,lattice);
    scene.display();
}


struct GenerateMicrostructure
{

    Mat2F32 correlationToNormalizedCorrelation(Mat2F32 correlation){
        Mat2F32 correlation_normalized(correlation);
        F32 phi = correlation(0,1);
        for(unsigned int i=0;i<correlation.sizeI();i++){
            correlation_normalized(i,1)=(correlation(i,1)-phi*phi)/(phi-phi*phi);
        }
        return correlation_normalized;

    }
    Mat3UI8 generateRealization(F32 porosity,Distribution & d){

        F32 moment_order_3= pop::Statistics::moment(d,3,0,50);
        F32 angle=15*PI/180.;//15 degre


        //8*E^3(R)/E^3(std::cos(theta))
        F32 volume_expectation = moment_order_3*8./(std::pow(std::cos(angle),3.0));
        Vec3F32 domain(128,128,128);//2d field domain
        F32 lambda=-std::log(porosity)/std::log(2.718)/volume_expectation;

        ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process
        DistributionDirac dirar_angle(angle);
        RandomGeometry::rhombohedron(grain,d,dirar_angle);
        return RandomGeometry::continuousToDiscrete(grain);
    }

    F32 distanceCorrelation(Mat2F32 correlation_realization_normalized, Mat2F32 correlation_expected_normalized ){
        F32 sum=0;
        for(unsigned int j=0;j<std::min(correlation_realization_normalized.sizeI(),correlation_expected_normalized.sizeI());j++){
            sum+=std::abs(correlation_realization_normalized(j,1)-correlation_expected_normalized(j,1));
        }
        return sum;
    }

    Mat3UI8 generateMicrosturcture(Mat2F32 correlation_experimental){

        Mat2F32 correlation_experimental_normalized = correlationToNormalizedCorrelation(correlation_experimental);
        int radius_min;
        F32 dist_max=std::numeric_limits<F32>::max();
        for(unsigned int radius=5;radius<50;radius++){
            DistributionDirac d(radius);
            Mat3UI8 realization = generateRealization(correlation_experimental(0,1),d);
            realization = Processing::greylevelRemoveEmptyValue(realization);
            Mat2F32 correlation_realization = Analysis::correlation(realization);
            Mat2F32 correlation_realization_normalized = correlationToNormalizedCorrelation(correlation_realization);
            F32 dist_temp = distanceCorrelation(correlation_experimental_normalized,correlation_realization_normalized);
            if(dist_temp<dist_max){
                radius_min=radius;
                dist_max = dist_temp;
            }
            std::cout<<"for radius="<<radius<<", the distance = "<<dist_temp<<std::endl;
        }
        std::cout<<"radius min"<<std::endl;
        DistributionDirac d(radius_min);
        return generateRealization(correlation_experimental(0,1),d);
    }
};

void SectionIII_Model_Correlation(){

    Mat3UI8 m;
    m.load("/media/tariel/5ee29f74-2a42-4cd5-8562-be01b51ee876/home/vincent/Desktop/WorkSegmentation/RockANU/SLB13/SLB_AU13.pgm");
    Mat2UI8 plane = GeometricalTransformation::plane(m,10);
    plane = PDE::nonLinearAnisotropicDiffusion(plane);
    int value;
    plane = Processing::thresholdOtsuMethod(plane,value);

    plane = Processing::greylevelRemoveEmptyValue(plane);
    Mat2F32 correlation_exprerimental= Analysis::correlation(plane);

    GenerateMicrostructure generate;
    //Mat3UI8 m_model_boolean = generate.generateMicrosturcture(correlation_exprerimental);



    Mat2F32 correlation_exprerimental_matrix = correlation_exprerimental.deleteCol(1);//remove the pore-pore correlation function in the first column

    //Mat3F32 m_gaussian_field;
    //Mat3UI8 m_model_gaussian_field =RandomGeometry::gaussianThesholdedRandomField(correlation_exprerimental_matrix,256,m_gaussian_field);

    Mat2F32 volume_fraction = Analysis::histogram(plane);
    Vec3I32 domain(256,256,256);
    Mat3UI8 m_model_annealing_field = RandomGeometry::randomStructure(domain,volume_fraction);
    RandomGeometry::annealingSimutated(m_model_annealing_field,plane,10);
    //Processing::greylevelRange(m_model_annealing_field,0,255).display();
    m_model_annealing_field.save("annealing_rock.pgm");
}

int main()
{
    SectionIII_Model_Correlation();
    //##UNCOMMENT EACH SECTION BY EACH SECTION BECAUSE THE PROGRAM STOP WHEN YOU CLOSE THE OPENGL WINDOW

    //SectionI_GaussianField();
    //SectionI_Annealing();
    //SectionII_PoissonPointProcess();
    //    SectionII_MaternFilter();
    //    SectionII_MinOverlap();
    //    SectionII_Voronoi();
    //    SectionII_EulerAngle();
    //    SectionII_ProbabilityDistributionDirection_Fix();
    //    SectionII_ProbabilityDistributionDirection_Random();
    //    SectionII_ProbabilityDistribution_Normal();
    //    SectionII_ProbabilityDistribution_Power();

    //    SectionII_Sphere();
    //   SectionII_Ellipse();
    //    SectionII_Cylinder();
    //    SectionII_Box();
    //    SectionII_Rhombohedra();
    //    SectionII_RegularTetrahedron();
    //        SectionII_Model_LatticeToContinious();
    //    SectionII_Model_DeadLeave();

    //   SectionII_Model_Transparency();
    //   SectionII_Model_ShotNoise();

    return 0;
}














//void testRandomColorDeadLeave(){
//    DistributionExpression dexp("1/x^(3.1)");
//    DistributionRegularStep d= Statistics::toProbabilityDistribution(dexp,5,512,0.1);
//    Vec2F32 domain(512,512);//2d field domain
//    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.1);//generate the 3d Poisson point process
//    RandomGeometry::sphere(grain,d);
//    grain.setModel(DeadLeave);
//    DistributionMultiVariateProduct drgb(DistributionUniformInt(0,255),DistributionUniformInt(0,255),DistributionUniformInt(0,255));
//    RandomGeometry::RGBRandom(grain,drgb);
//    Mat2RGBUI8 lattice = RandomGeometry::continuousToDiscrete(grain);
//    lattice.display();
//}



#endif // ARTICLEGERMGRAIN_H
