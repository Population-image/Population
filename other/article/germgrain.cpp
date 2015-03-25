#ifndef ARTICLEGERMGRAIN_H
#define ARTICLEGERMGRAIN_H
#include"Population.h"
using namespace pop;//Population namespace

void figure1_PoissonPointProcess()
{
    {
        Vec2F32 domain(512,512);//2d field domain
        double lambda= 0.001;// parameter of the Poisson Point process
        ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson Point process
        DistributionDirac d (1);//because the Poisson Point process has a surface equal to 0, we associate each point with mono-disperse sphere to display the result
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
void figure2_MaternFilter()
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
void figure3_MinOverlap(){
    Vec2F32 domain(512,512);//2d field domain
    double lambda= 0.01;// parameter of the Poisson Point process
    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson Point process
    RandomGeometry::minOverlapFilter(grain,10);
    DistributionDirac d(5);
    RandomGeometry::sphere(grain,d);
    Mat2UI8 img = RandomGeometry::continuousToDiscrete(grain);
    img = img.opposite();
    img.save("MinOverlap.png");
    img.display();
}
void figure4_Voronoi(){
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
void figure5_EulerAngle(){

    GrainBox<3> * box = new GrainBox<3>();
    box->radius = Vec3F32(20,30,10);
    box->x      = Vec3F32(50,50,50);
    ModelGermGrain3 germ_grain;
    germ_grain.setDomain(Vec3F32(100,100,100));
    germ_grain.grains().push_back(box);



    box->orientation.setAngle_ei(PI/4,0);

    //    Scene3d scene;
    //    Visualization::marchingCube(scene,img);
    //    pop::Visualization::lineCube(scene,img);
    //    scene.display(false);
    //    waitKey();
    //    scene.clear();
    //    box->orientation.setAngle_ei(PI/4,0);
    //    Mat3UI8 img = RandomGeometry::continuousToDiscrete(germ_grain);
    //    Visualization::marchingCube(scene,img);
    //    pop::Visualization::lineCube(scene,img);
    //    scene.display(false);
    //    waitKey();

}

void testAnnealing(){
    Mat2UI8 threshold;//2d grey-level image object
    threshold.load("/home/vincent/Desktop/meb_A_675_7j_visu.bmp");//replace this path by those on your computer
    threshold = GeometricalTransformation::subResolution(threshold,4);
    threshold = Processing::greylevelRemoveEmptyValue(threshold);
    Visualization::labelToRandomRGB(threshold).display();
    Mat2F32 volume_fraction = Analysis::histogram(threshold);
    //2D case
    {

//        Vec2I32 v(256,256);
//        Mat2UI8 random = RandomGeometry::randomStructure(v,volume_fraction);
//        RandomGeometry::annealingSimutated(random,threshold);
//        Visualization::labelToRandomRGB(random).display();
    }
    //3D case (expensive process)
    {
        Vec3I32 v(256,256,256);
        Mat3UI8 random = RandomGeometry::randomStructure(v,volume_fraction);
        RandomGeometry::annealingSimutated(random,threshold,10);
        Mat2F32 m_field_field_annealing_model = Analysis::correlation(random,200);
        m_field_field_annealing_model = m_field_field_annealing_model.deleteCol(1);//remove the pore-pore correlation function in the first column
        m_field_field_annealing_model.saveAscii("CTG_field_field_annealing_model.m");

        random.save("annealing.pgm");
        Scene3d scene;
        Visualization::marchingCubeSmooth(scene,random);
        Visualization::lineCube(scene,random);
        scene.display();
    }
}

void testGaussianField(){
    Mat2UI8 img;//2d grey-level image object
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/CTG.pgm"));//replace this path by those on your computer
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    int value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);
    threshold.display("initial",false);
    threshold = Processing::greylevelRemoveEmptyValue(threshold);
    Mat2F32 m_field_field_correlation_experimental =Analysis::correlation(threshold,200);
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
int main()
{
    {
        Mat3UI8 random;
        random.load("annealing_save.pgm");
        Visualization::labelToRandomRGB(random).display();

    }
//    Mat2UI8 m(8,8);
//    m=0;
//    m(4,4)=1;m(5,5)=1;
//    Mat2F32 volume_fraction = Analysis::histogram(m);
//    Vec2I32 v(8,8);
//    Mat2UI8 random = RandomGeometry::randomStructure(v,volume_fraction);
//    RandomGeometry::annealingSimutated(random,m);

//    Mat3UI8 init;
//   Mat3UI8 random ;
//   init.load("/home/vincent/Desktop/WorkSegmentation/lavoux/lavoux.pgm");
//    random.load("/home/vincent/Desktop/WorkSegmentation/lavoux/segmentation.pgm");

//    Visualization::labelForegroundBoundary(random,init).display();
//    Mat2F32 m_field_field_annealing_model = Analysis::correlation(random,200);
//    m_field_field_annealing_model = m_field_field_annealing_model.deleteCol(1);//remove the pore-pore correlation function in the first column
//    m_field_field_annealing_model.saveAscii("CTG_field_field_annealing_model.m");

    testAnnealing();
//    testGaussianField();
    //    figure1_PoissonPointProcess();
//        figure2_MaternFilter();
    //figure3_MinOverlap();
//    figure4_Voronoi();
    figure5_EulerAngle();
    //    testVoronoiTesselation3D();
    //    testMinOverlap();
    //    //testUniformPoissonPointProcess2D();
    //    //testAnnealing();
    //    artAborigene();

    return 0;
}



//void testUniformPoissonPointProcess2D(){
//    Vec2F32 domain(512,512);//2d field domain
//    double lambda= 0.001;// parameter of the Poisson Point process

//    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson Point process
//    DistributionDirac d (1);//because the Poisson Point process has a surface equal to 0, we associate each point with mono-disperse sphere to display the result
//    RandomGeometry::sphere(grain,d);
//    Mat2RGBUI8 img = RandomGeometry::continuousToDiscrete(grain);
//    img.display();
//}


//void testUniformPoissonPointProcess3D(){
//    Vec3F32 domain(200,200,200);//2d field domain
//    double lambda= 0.0001;// parameter of the Poisson Point process

//    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson Point process
//    DistributionDirac d (1);//because the Poisson Point process has a surface equal to 0, we associate each point with mono-disperse sphere to display the result
//    RandomGeometry::sphere(grain,d);
//    Mat3RGBUI8 img = RandomGeometry::continuousToDiscrete(grain);
//    Scene3d scene;
//    pop::Visualization::marchingCube(scene,img);
//    pop::Visualization::lineCube(scene,img);
//    scene.display();
//}
//void testNonUniformPoissonPointProcess2D() {



//    Mat2UI8 img("../Lena.bmp");
//    Mat2F32 imgf(img);
//    imgf=imgf/255.;
//    imgf=imgf*0.05;
//    ModelGermGrain2 grain = RandomGeometry::poissonPointProcessNonUniform(imgf);
//    DistributionDirac d(4);
//    RandomGeometry::sphere(grain,d);
//    img = RandomGeometry::continuousToDiscrete(grain);
//    img.display();
//}

//void testMinOverlap(){

//    Vec2F32 domain(512,512);
//    ModelGermGrain<2> germgrain =  pop::RandomGeometry::poissonPointProcess(domain,0.002);

//    ModelGermGrain<2> germgrain2 = germgrain;
//    pop::RandomGeometry::minOverlapFilter(germgrain2,20);
//    DistributionDirac d(10);
//    pop::RandomGeometry::sphere(germgrain,d);
//    pop::RandomGeometry::sphere(germgrain2,d);
//    pop::RandomGeometry::continuousToDiscrete(germgrain).display("Boolean",false,false);
//    pop::RandomGeometry::continuousToDiscrete(germgrain2).display("Min Overlap",true,false);
//}
//void testVoronoiTesselation3D(){
//    Vec3F32 domain(200,200,200);
//    ModelGermGrain3 germ = RandomGeometry::poissonPointProcess(domain,0.0001);

//    Mat3UI32 label = RandomGeometry::germToMatrix(germ);

//    label = Processing::voronoiTesselationEuclidean(label);
//    Mat3RGBUI8 voronoicolor =  Visualization::labelToRandomRGB(label);
//    Scene3d scene;
//    Visualization::cube(scene,voronoicolor);
//    scene.display();
//}
//void testSingleSphere(){
//    GrainSphere<3> *sphere = new GrainSphere<3>;

//    double size=100;
//    sphere->x = Vec3F32(size/8,size/2,size/2);
//    sphere->radius = size/2;

//    ModelGermGrain3 germgrain;
//    germgrain.setDomain(Vec3F32(size,size,size));
//    germgrain.grains().push_back(sphere);
//    germgrain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_BOUNDED);
//    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
//    Scene3d scene;
//    Visualization::marchingCube(scene,m);
//    Visualization::lineCube(scene,m);
//    scene.display();
//}
//void testEllipse(){
//    GrainEllipsoid<3> *ellipse = new GrainEllipsoid<3>;
//    double size=100;
//    ellipse->x = Vec3F32(size/2,size/2,size/2);
//    ellipse->setRadius(Vec3F32(size/2,size/4,size/8));
//    ModelGermGrain3 germgrain;
//    germgrain.setDomain(Vec3F32(size,size,size));
//    germgrain.grains().push_back(ellipse);
//    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
//    Scene3d scene;
//    Visualization::marchingCube(scene,m);
//    Visualization::lineCube(scene,m);
//    scene.display();
//}
//void testCylinder(){
//    GrainCylinder *cylinder = new GrainCylinder;
//    double size=100;
//    cylinder->x = Vec3F32(size/2,size/2,size/2);
//    cylinder->radius = size/8;
//    cylinder->height = size*3./4.;
//    ModelGermGrain3 germgrain;
//    germgrain.setDomain(Vec3F32(size,size,size));
//    germgrain.grains().push_back(cylinder);
//    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
//    Scene3d scene;
//    Visualization::marchingCube(scene,m);
//    Visualization::lineCube(scene,m);
//    scene.display();
//}
//void testBox(){
//    GrainBox<3> *box = new GrainBox<3>;
//    double size=100;
//    box->x = Vec3F32(size/2,size/2,size/2);
//    box->radius = Vec3F32(size/2,size/4,size/8);
//    ModelGermGrain3 germgrain;
//    germgrain.setDomain(Vec3F32(size,size,size));
//    germgrain.grains().push_back(box);
//    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
//    Scene3d scene;
//    Visualization::marchingCube(scene,m);
//    Visualization::lineCube(scene,m);
//    scene.display();
//}
//void testRhombohedra(){
//    GrainEquilateralRhombohedron *rhombohedra = new GrainEquilateralRhombohedron;
//    double size=100;
//    rhombohedra->x = Vec3F32(size/2,size/2,size/2);
//    rhombohedra->radius = size/6;
//    rhombohedra->setAnglePlane(30.*pop::PI/180.);
//    ModelGermGrain3 germgrain;
//    germgrain.setDomain(Vec3F32(size,size,size));
//    germgrain.grains().push_back(rhombohedra);
//    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
//    Scene3d scene;
//    Visualization::marchingCube(scene,m);
//    Visualization::lineCube(scene,m);
//    scene.display();
//}

//void testRegularTetrahedron(){
//    GrainPolyhedra<3> *polyhedra = new GrainPolyhedra<3>;

//    //The following Cartesian coordinates define the four vertices of a tetrahedron with edge-length 2, centered at the origin:
//    Vec3F32 x1( 1, 0, -1./std::sqrt(2.));
//    Vec3F32 x2(-1, 0, -1./std::sqrt(2.));
//    Vec3F32 x3( 0, 1,  1./std::sqrt(2.));
//    Vec3F32 x4( 0,-1,  1./std::sqrt(2.));

//    double size=100;

//    Vec3F32 normal_x1_x2_x3 = productVectoriel(x1-x2,x1-x3);
//    normal_x1_x2_x3 *=sgn( productInner(normal_x1_x2_x3,x1));
//    normal_x1_x2_x3/=normal_x1_x2_x3.norm();
//    polyhedra->addPlane(size/6,normal_x1_x2_x3);

//    Vec3F32 normal_x1_x3_x4 = productVectoriel(x1-x3,x1-x4);
//    normal_x1_x3_x4 *=sgn( productInner(normal_x1_x3_x4,x1));
//    normal_x1_x3_x4/=normal_x1_x3_x4.norm();
//    polyhedra->addPlane(size/6,normal_x1_x3_x4);


//    Vec3F32 normal_x1_x4_x2 = productVectoriel(x1-x4,x1-x2);
//    normal_x1_x4_x2 *=sgn( productInner(normal_x1_x4_x2,x1));
//    normal_x1_x4_x2/=normal_x1_x4_x2.norm();
//    polyhedra->addPlane(size/6,normal_x1_x4_x2);

//    Vec3F32 normal_x2_x3_x4 = productVectoriel(x2-x3,x2-x4);
//    normal_x2_x3_x4 *=sgn( productInner(normal_x2_x3_x4,x2));
//    normal_x2_x3_x4/=normal_x2_x3_x4.norm();
//    polyhedra->addPlane(size/6,normal_x2_x3_x4);


//    polyhedra->x = Vec3F32(size/2,size/2,size/2);

//    ModelGermGrain3 germgrain;
//    germgrain.setDomain(Vec3F32(size,size,size));
//    germgrain.grains().push_back(polyhedra);
//    Mat3UI8 m = RandomGeometry::continuousToDiscrete(germgrain);
//    Scene3d scene;
//    Visualization::marchingCube(scene,m);
//    Visualization::lineCube(scene,m);
//    scene.display();
//}


//void testProbilityDistributionNormal(double porosity_volume_fraction=0.6){

//    DistributionNormal d(20,10);
//    double grain_volume_expectation = 4./3.*pop::PI*Statistics::moment(d,3,0,50);
//    double lambda = -std::log(porosity_volume_fraction)/grain_volume_expectation;


//    Vec3F32 domain(256,256,256);//3d field domain
//    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 3d Poisson point process
//    RandomGeometry::sphere(grain,d);
//    Mat3UI8 lattice = RandomGeometry::continuousToDiscrete(grain);
//    std::cout<<"realization porosity: "<<Analysis::histogram(lattice)(0,1)<<std::endl;
//    std::cout<<"expected porosity: "<<porosity_volume_fraction<<std::endl;
//    Scene3d scene;
//    pop::Visualization::marchingCube(scene,lattice);
//    pop::Visualization::lineCube(scene,lattice);
//    scene.display();
//}

//void testProbilityDistributionPowerLaw(double porosity_volume_fraction=0.6){
//    DistributionExpression dexp("1/x^(4.1)");
//    DistributionRegularStep d= Statistics::toProbabilityDistribution(dexp,5,512,0.1);
//    double grain_volume_expectation = 4./3.*pop::PI*Statistics::moment(d,3,0,50);
//    double lambda = -std::log(porosity_volume_fraction)/grain_volume_expectation;


//    Vec3F32 domain(256,256,256);//3d field domain
//    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 3d Poisson point process
//    RandomGeometry::sphere(grain,d);
//    Mat3UI8 lattice = RandomGeometry::continuousToDiscrete(grain);
//    std::cout<<"realization porosity: "<<Analysis::histogram(lattice)(0,1)<<std::endl;
//    std::cout<<"expected porosity: "<<porosity_volume_fraction<<std::endl;
//    Scene3d scene;
//    pop::Visualization::marchingCube(scene,lattice);
//    pop::Visualization::lineCube(scene,lattice);
//    scene.display();
//}

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

//void artAborigene(){
//    Mat2RGBUI8 img;
//    img.load(std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp");
//    img= GeometricalTransformation::scale(img,Vec2F32(1024./img.getDomain()(0)));
//    double radius=10;
//    Vec2F32 domain;
//    domain= img.getDomain();
//    ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,1);//generate the 2d Poisson point process
//    grain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_BOUNDED);
//    RandomGeometry::hardCoreFilter(grain,radius);
//    DistributionExpression d ("1/x^3");
//    DistributionRegularStep d_prob = Statistics::toProbabilityDistribution(d,5,256);
//    RandomGeometry::sphere(grain, d_prob);
//    RandomGeometry::RGBFromMatrix(grain,img);
//    grain.setModel( DeadLeave);
//    grain.setTransparency(0.5);
//    Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain);
//    Mat2RGBUI8::IteratorEDomain it = img.getIteratorEDomain();
//    RGBUI8 mean = AnalysisAdvanced::meanValue(img,it);
//    std::cout<<mean<<std::endl;
//    it.init();
//    while(it.next()){
//        if(aborigenart(it.x())==RGBUI8(0))
//            aborigenart(it.x())=mean;
//    }
//    aborigenart.display("art");
//    aborigenart.save("../../../art_aborigene.jpg");
//    // aborigenart.save("/home/vincent/Desktop/Population/doc/image/AborigenLena.bmp");
//}

#endif // ARTICLEGERMGRAIN_H
