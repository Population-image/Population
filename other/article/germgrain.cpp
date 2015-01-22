#ifndef ARTICLEGERMGRAIN_H
#define ARTICLEGERMGRAIN_H
#include"Population.h"
using namespace pop;//Population namespace
void testAnnealing(){
    Mat2UI8 img;//2d grey-level image object
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    F32 value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);
    threshold.display("initial",false);
    threshold = Processing::greylevelRemoveEmptyValue(threshold);
    Vec2I32 v(512,512);
    Mat2F32 volume_fraction = Analysis::histogram(threshold);
    Mat2UI8 random = RandomGeometry::randomStructure(v,volume_fraction);
    RandomGeometry::annealingSimutated(random,threshold,8,256,0.01);
    Visualization::labelToRandomRGB(random).display();
}
void testUniformPoissonPointProcess2D(){
    Vec2F32 domain(512,512);//2d field domain
    double lambda= 0.001;// parameter of the Poisson VecN process

    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson VecN process
    Distribution d (1,"DIRAC");//because the Poisson VecN process has a surface equal to 0, we associate each point with mono-disperse sphere to display the result
    RandomGeometry::sphere(grain,d);
    Mat2RGBUI8 img = RandomGeometry::continuousToDiscrete(grain);
    img.display();
}


void testUniformPoissonPointProcess3D(){
    Vec3F32 domain(200,200,200);//2d field domain
    double lambda= 0.0001;// parameter of the Poisson VecN process

    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson VecN process
    Distribution d (1,"DIRAC");//because the Poisson VecN process has a surface equal to 0, we associate each point with mono-disperse sphere to display the result
    RandomGeometry::sphere(grain,d);
    Mat3RGBUI8 img = RandomGeometry::continuousToDiscrete(grain);
    Scene3d scene;
    pop::Visualization::marchingCube(scene,img);
    pop::Visualization::lineCube(scene,img);
    scene.display();
}
void testNonUniformPoissonPointProcess2D() {
    Mat2UI8 img("../Lena.bmp");
    Mat2F32 imgf(img);
    imgf=imgf/255.;
    imgf=imgf*0.05;
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcessNonUniform(imgf);
    Distribution d(4,"DIRAC");
    RandomGeometry::sphere(grain,d);
    img = RandomGeometry::continuousToDiscrete(grain);
    img.display();
}

void testMinOverlap(){

    Vec2F32 domain(512,512);
    ModelGermGrain<2> germgrain =  pop::RandomGeometry::poissonPointProcess(domain,0.002);

    ModelGermGrain<2> germgrain2 = germgrain;
    pop::RandomGeometry::minOverlapFilter(germgrain2,20);
    DistributionDirac d(10);
    pop::RandomGeometry::sphere(germgrain,d);
    pop::RandomGeometry::sphere(germgrain2,d);
    pop::RandomGeometry::continuousToDiscrete(germgrain).display("Boolean",false);
    pop::RandomGeometry::continuousToDiscrete(germgrain2).display("Min Overlap");
}
void testVoronoiTesselation3D(){
    Vec3F32 domain(100,100,100);
    ModelGermGrain3 germ = RandomGeometry::poissonPointProcess(domain,0.0001);

    Mat3UI32 label = RandomGeometry::germToMatrix(germ);

    label = Processing::voronoiTesselationEuclidean(label);
    Mat3RGBUI8 voronoicolor =  Visualization::labelToRandomRGB(label);
    Scene3d scene;
    Visualization::cube(scene,voronoicolor);
    scene.display();
}
void testSingleSphere(){
    GrainSphere<3> *sphere = new GrainSphere<3>;

    double size=100;
    sphere->x = Vec3F32(size/8,size/2,size/2);
    sphere->radius = size/2;

    ModelGermGrain3 germgrain;
    germgrain.setDomain(Vec3F32(size,size,size));
    germgrain.grains().push_back(sphere);
    germgrain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_BOUNDED);
    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
    Scene3d scene;
    Visualization::marchingCube(scene,m);
    Visualization::lineCube(scene,m);
    scene.display();
}
void testEllipse(){
    GrainEllipsoid<3> *ellipse = new GrainEllipsoid<3>;
    double size=100;
    ellipse->x = Vec3F32(size/2,size/2,size/2);
    ellipse->setRadius(Vec3F32(size/2,size/4,size/8));
    ModelGermGrain3 germgrain;
    germgrain.setDomain(Vec3F32(size,size,size));
    germgrain.grains().push_back(ellipse);
    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
    Scene3d scene;
    Visualization::marchingCube(scene,m);
    Visualization::lineCube(scene,m);
    scene.display();
}
void testCylinder(){
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
void testBox(){
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
void testRhombohedra(){
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
    Visualization::marchingCube(scene,m);
    Visualization::lineCube(scene,m);
    scene.display();
}

void testRegularTetrahedron(){
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


void testProbilityDistributionNormal(double porosity_volume_fraction=0.6){

        DistributionNormal d(20,10);
        double grain_volume_expectation = 4./3.*pop::PI*Statistics::moment(d,3,0,50);
        double lambda = -std::log(porosity_volume_fraction)/grain_volume_expectation;


        Vec3F32 domain(256,256,256);//3d field domain
        ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 3d Poisson point process
        RandomGeometry::sphere(grain,d);
        Mat3UI8 lattice = RandomGeometry::continuousToDiscrete(grain);
        std::cout<<"realization porosity: "<<Analysis::histogram(lattice)(0,1)<<std::endl;
        std::cout<<"expected porosity: "<<porosity_volume_fraction<<std::endl;
        Scene3d scene;
        pop::Visualization::marchingCube(scene,lattice);
        pop::Visualization::lineCube(scene,lattice);
        scene.display();
}

void testProbilityDistributionPowerLaw(double porosity_volume_fraction=0.6){
    Distribution dexp(DistributionExpression("1/x^(4.1)"));
    DistributionRegularStep d= Statistics::toProbabilityDistribution(dexp,5,512,0.1);
    double grain_volume_expectation = 4./3.*pop::PI*Statistics::moment(d,3,0,50);
    double lambda = -std::log(porosity_volume_fraction)/grain_volume_expectation;


    Vec3F32 domain(256,256,256);//3d field domain
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 3d Poisson point process
    RandomGeometry::sphere(grain,d);
    Mat3UI8 lattice = RandomGeometry::continuousToDiscrete(grain);
    std::cout<<"realization porosity: "<<Analysis::histogram(lattice)(0,1)<<std::endl;
    std::cout<<"expected porosity: "<<porosity_volume_fraction<<std::endl;
    Scene3d scene;
    pop::Visualization::marchingCube(scene,lattice);
    pop::Visualization::lineCube(scene,lattice);
    scene.display();
}

void testRandomColorDeadLeave(){
    Distribution dexp(DistributionExpression("1/x^(3.1)"));
    DistributionRegularStep d= Statistics::toProbabilityDistribution(dexp,5,512,0.1);
    Vec2F32 domain(512,512);//2d field domain
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.1);//generate the 3d Poisson point process
    RandomGeometry::sphere(grain,d);
    grain.setModel(DeadLeave);
    DistributionMultiVariate d0255(Distribution(0,255,"UNIFORMINT"));
    DistributionMultiVariate drgb(d0255 ,DistributionMultiVariate(d0255,d0255));
    RandomGeometry::RGBRandom(grain,drgb);
    Mat2RGBUI8 lattice = RandomGeometry::continuousToDiscrete(grain);
    lattice.display();
}

void artAborigene(){
    Mat2RGBUI8 img;
    img.load("art.jpg");
    img= GeometricalTransformation::scale(img,Vec2F32(1024./img.getDomain()(0)));
    double radius=10;
    Vec2F32 domain;
    domain= img.getDomain();
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,1);//generate the 2d Poisson point process
    grain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_BOUNDED);
    RandomGeometry::hardCoreFilter(grain,radius);
    Distribution d ("1/x^3");
    d = Statistics::toProbabilityDistribution(d,5,256);
    //Distribution d (radius,"DIRAC");//because the Poisson point process has a surface equal to 0, we associate each VecN with mono-disperse sphere to display the result
    RandomGeometry::sphere(grain,d);
    RandomGeometry::RGBFromMatrix(grain,img);
    grain.setModel( DeadLeave);
    grain.setTransparency(0.5);
    Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain);
    Mat2RGBUI8::IteratorEDomain it = img.getIteratorEDomain();
    RGBUI8 mean = AnalysisAdvanced::meanValue(img,it);
    std::cout<<mean<<std::endl;
    it.init();
    while(it.next()){
       if(aborigenart(it.x())==RGBUI8(0))
            aborigenart(it.x())=mean;
    }
    aborigenart.display();
    aborigenart.save("../../../art_aborigene.jpg");
   // aborigenart.save("/home/vincent/Desktop/Population/doc/image/AborigenLena.bmp");
}
int main()
{
    testAnnealing();
//    artAborigene();

    return 0;
}

#endif // ARTICLEGERMGRAIN_H
