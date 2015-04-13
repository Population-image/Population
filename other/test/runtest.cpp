#include"poptest.h"
#include"Population.h"
#include"poptest.h"
using namespace pop;
#include"data/mat/MatN.h"
#include"algorithm/Analysis.h"
#include <omp.h>








void testMatN(){

    pop::PopTest test;
    test._bool_write = true;

    Mat2F32 m1(2,3);
    m1(0,0)=1; m1(0,1)=2; m1(0,2)=0;
    m1(1,0)=4; m1(1,1)=3; m1(1,2)=-1;

    Mat2F32 m2(3,2);
    m2(0,0)=5; m2(0,1)=1;
    m2(1,0)=2; m2(1,1)=3;
    m2(2,0)=3; m2(2,1)=4;
    Mat2F32 m3 = m1*m2;

    std::cout<<m3<<std::endl;
    Mat2F32 mout(2,2);
    mout(0,0)=9; mout(0,1)=7;
    mout(1,0)=23; mout(1,1)=9;

    if(m3!=mout){
        std::cerr<<"[ERROR] Matrix multiplication"<<std::endl;
        exit(0);
    }

    VecF32 v1(3);
    v1(0)=2;v1(1)=-1;v1(2)=4;


    VecF32 v2 = m1*v1;

    VecF32 vout(2);
    vout(0)=0;vout(1)=1;
    if(v2!=vout){
        std::cerr<<"[ERROR] Matrix multiplication with vector"<<std::endl;
        exit(0);
    }




}

int testAnamysis(){

    return 1;
    std::vector<int> v(3);
    v[0]=2;v[1]=4;v[6]=2;



    Mat3UI8 porespace;
    porespace.load(std::string(POP_PROJECT_SOURCE_DIR)+"/image/spinodal.pgm");
    porespace = pop::Processing::greylevelRemoveEmptyValue(porespace);//the porespace label is now 1 (before 255)
    //    porespace = porespace(Vec3I32(0,0,0),Vec3I32(50,50,50));//for the first execution, you can test the process in small sample

    //    Mat2F32 mverpore = Analysis::REVPorosity(porespace,VecN<3,F64>(porespace.getDomain())*0.5,200);
    //    mverpore.saveAscii("spinodal_VER.m");

    //    Mat2F32 mhisto = Analysis::histogram(porespace);
    //    std::cout<<"porespace fraction:"<<mhisto(1,1)<<std::endl;
    //    Mat2F32 mchord = Analysis::chord(porespace);
    //    mchord.saveAscii("spinodal_chord.m");
    //    mchord.deleteCol(1);
    //    Distribution dchord_solid(mchord);
    //    std::cout<<"Charateristic length of solid space "<<Statistics::moment(dchord_solid,1,0,300,1)<<std::endl;// \sum_{i=0}^{300} i*d(i)=27.2

    //    Mat2F32 mcorre= Analysis::correlation(porespace);
    //    mcorre.saveAscii("spinodal_corre.m");

    //    Mat3F32 corre= Analysis::correlationDirectionByFFT(porespace);
    //    corre = GeometricalTransformation::translate(corre,corre.getDomain()/2,MATN_BOUNDARY_CONDITION_PERIODIC);//centered
    ////    corre = corre(corre.getDomain()/2-Vec3I32(20,20,20),corre.getDomain()/2+Vec3I32(20,20,20));//take only the core (the correlation length is near 20)
    //    Mat3UI8 dcorre;
    //    dcorre= pop::Processing::greylevelRange(corre,1,255);//0 is black color so start at 1
    //    Mat3RGBUI8 dcorregrad=Visualization::labelToRGBGradation(dcorre);//color gradation
    ////    dcorregrad.display();

    ////    Mat2UI8 dcorre = pop::Processing::greylevelRange(corre.getPlane(0,0),1,255);//0 is black color so start at 1
    ////    Visualization::labelToRGBGradation(dcorre).display();//color gradation
    ////    {
    ////    Scene3d scene;
    ////    Visualization::plane(scene,dcorregrad,dcorre.getDomain()(0)/2,0);
    ////    Visualization::plane(scene,dcorregrad,dcorre.getDomain()(1)/2,1);
    ////    Visualization::plane(scene,dcorregrad,dcorre.getDomain()(2)/2,2);
    ////    Visualization::lineCube(scene,dcorregrad);
    ////    scene.display();//if you close the opengl windows, that stop the program in linux. So comment this line in linux to execute the rest of the code
    ////    }
    //    Mat3UI8 fdistance;
    //    Mat2F32 mldistance= Analysis::ldistance(porespace,2,fdistance);//euclidean distance only in Population library ;-)
    //    mldistance.saveAscii("spinodal_ldistance.m");
    //    Mat3RGBUI8 dcorregrad1=Visualization::labelToRandomRGB(fdistance);//random color
    //    dcorregrad1.display();
    ////    Scene3d scene;
    ////    Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(0)/2,0);
    ////    Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(1)/2,1);
    ////    Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(2)/2,2);
    ////    Visualization::lineCube(scene,dcorregrad);
    ////    scene.display();

    //    Mat3UI8 fgranulo;
    //    //granulo of the solid space
    //    Mat2F32 mlgranulo= Analysis::granulometryMatheron(porespace,2,fgranulo);//quite long algorithm in euclidean distance
    //    mlgranulo.saveAscii("spinodal_granulo.m");
    //    Mat3RGBUI8 dcorregrad2=Visualization::labelToRandomRGB(fgranulo);//random color
    //    dcorregrad2.display();
    ////    Scene3d scene;
    ////    Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(0)/2,0);
    ////    Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(1)/2,1);
    ////    Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(2)/2,2);
    ////    Visualization::lineCube(scene,dcorregrad);
    ////    scene.display();

    //    Mat2F32 mgeometrical = Analysis::geometricalTortuosity(porespace);
    //    mgeometrical.saveAscii("spinodal_geometrical.m");


    //    //############ TOPOLOGY #######################
    //    double euler = Analysis::eulerPoincare(porespace,std::string(POP_PROJECT_SOURCE_DIR)+"/file/eulertab.dat");
    //    std::ofstream out("spinodal_euler.m");
    //    out<<euler;//euler
    //    out.close();
    //    Mat2F32 mpercolationopening = Analysis::percolationOpening(porespace,2);//->charactertic length related to permeability
    //    mpercolationopening.saveAscii("spinodal_percolationopening.m");//output is 6 in three direction-> the structure sill percolates after an opening of ball of radius of 6 but not with a radius of size 7
    //    Mat2F32 mpercolationerosion = Analysis::percolationErosion(porespace,2);
    //    mpercolationerosion.saveAscii("spinodal_percolationerosion.m");//output is 5 in three direction-> the structure sill percolates after an erosion of ball of radius of 5 but not with a radius of size 6
    //    Mat3UI8 porespace_hole= pop::Processing::holeFilling(porespace);
    //    Mat3UI8 skeleton= Analysis::thinningAtConstantTopology3d(porespace_hole,std::string(POP_PROJECT_SOURCE_DIR)+"/file/topo24.dat");
    ////    Scene3d scene;
    ////    pop::Visualization::voxelSurface(scene,skeleton);
    ////    pop::Visualization::lineCube(scene,skeleton);
    ////    scene.display();


    //    std::pair<Mat3UI8,Mat3UI8> vertex_edge = Analysis::fromSkeletonToVertexAndEdge (skeleton);
    //    Mat3UI32 verteces = pop::Processing::clusterToLabel(vertex_edge.first,0);
    //    Mat3UI32 edges = pop::Processing::clusterToLabel(vertex_edge.second,0);
    ////    Scene3d scene;
    ////    pop::Visualization::voxelSurface(scene,pop::Visualization::labelToRandomRGB(edges));
    ////    pop::Visualization::lineCube(scene,edges);
    ////    scene.display();

    //    int tore;
    //    GraphAdjencyList<Vec3I32> g = Analysis::linkEdgeVertex(verteces,edges,tore);
    ////    Scene3d scene;
    ////    pop::Visualization::graph(scene,g);
    ////    pop::Visualization::lineCube(scene,edges);
    ////    scene.display();

    //    std::cout<<euler/g.sizeVertex()<<std::endl;//N_3/alpha_0 normalised topogical characteristic (connectivity number in my phd)
    //    //############ PHYSICAL ###################
    Mat2F32 mdiffusion = PDE::randomWalk(porespace,500);
    mdiffusion.saveAscii("spinodal_self_diffusion.m","# time\t D_0/D(t)\t D_0/D_x(t)\t D_0/D_y(t)\t D_0/D_z(t)");

    MatN<3,Vec3F32> velocityfield;
    Mat2F32 K(3,3);
    VecF32 kx = PDE::permeability(porespace,velocityfield,0,0.01);//permeability in x-direction
    //    VecF32 ky = PDE::permeability(porespace,velocityfield,1,0.01);//permeability in y-direction
    //    VecF32 kz = PDE::permeability(porespace,velocityfield,2,0.01);//permeability in z-direction
    //merge the results in the permeability matrix
    K.setCol(0,kx);
    //    K.setCol(1,ky);
    //    K.setCol(2,kz);
    //display the norm of the last valocity field
    Mat3F32 velocityfield_norm(velocityfield);

    ForEachDomain3D(x,velocityfield)
    {
        velocityfield_norm(x)=normValue(velocityfield(x));
    }
    Mat3RGBUI8 velocityfield_norm_grad= pop::Visualization::labelToRGBGradation(velocityfield_norm);
    Scene3d scene;
    Visualization::plane(scene,velocityfield_norm_grad,velocityfield_norm_grad.getDomain()(0)/2,0);
    Visualization::plane(scene,velocityfield_norm_grad,velocityfield_norm_grad.getDomain()(1)/2,1);
    Visualization::plane(scene,velocityfield_norm_grad,velocityfield_norm_grad.getDomain()(2)/2,2);
    Visualization::lineCube(scene,velocityfield_norm_grad);
    scene.display();
}
void processingTest()
{
    pop::PopTest test;
    test._bool_write = true;
    pop::Mat2UI8 in,out;
    in.load(std::string(POP_PROJECT_SOURCE_DIR)+"/image/iex.png");

    test.start("threshold");
    double threshold=120;
    out = pop::Processing::threshold(in,threshold);
    test.end(out);


    test.start("thresholdKMeansVariation");
    out = pop::Processing::thresholdKMeansVariation(in);
    test.end(out);


    test.start("thresholdOtsuMethod");
    int value;
    out = pop::Processing::thresholdOtsuMethod(in,value);
    test.end(out);

    double radius1=2;
    test.start("thresholdToggleMappingMorphologicalFabrizio",pop::BasicUtility::Any2String(radius1));
    out = pop::Processing::thresholdToggleMappingMorphologicalFabrizio(in,30,1,radius1);
    test.end(out);


    test.start("integral");
    out = Processing::greylevelRange(pop::Processing::integral(Mat2UI32(in)),0,255);
    test.end(out);

    test.start("minimaLocalMap");
    out = Processing::threshold(pop::Processing::minimaLocalMap(in),1);
    test.end(out);

    double radius_erosion=2;
    test.start("erosion",pop::BasicUtility::Any2String(radius_erosion));
    out = pop::Processing::erosion(in,radius_erosion);
    test.end(out);


    test.start("median",pop::BasicUtility::Any2String(radius_erosion));
    out = pop::Processing::median(in,radius_erosion);
    test.end(out);


    test.start("meanShiftFilter",pop::BasicUtility::Any2String(radius_erosion));
    out = pop::Processing::meanShiftFilter(in);
    test.end(out);

    test.start("alternateSequentialCO",pop::BasicUtility::Any2String(radius_erosion));
    out = pop::Processing::alternateSequentialCO(in,radius_erosion);
    test.end(out);


    test.start("enhanceContrastToggleMapping",pop::BasicUtility::Any2String(radius_erosion));
    out = pop::Processing::enhanceContrastToggleMapping(in,radius_erosion,2);
    test.end(out);


    test.start("gradientMagnitudeSobel");
    out = pop::Processing::gradientMagnitudeSobel(in);
    test.end(out);

    double radius_deviation=2;
    test.start("gradientMagnitudeGaussian",pop::BasicUtility::Any2String(radius_deviation));
    out = pop::Processing::gradientMagnitudeGaussian(in,radius_deviation);
    test.end(out);

    test.start("gradientMagnitudeDeriche");
    out = pop::Processing::gradientMagnitudeDeriche(in,1);
    test.end(out);

    test.start("edgeDetectorCanny");
    out = pop::Processing::edgeDetectorCanny(in,1,1,10);
    test.end(out);

    {
        Mat2UI8 grad= pop::Processing::gradientMagnitudeGaussian(in,3);
        Mat2UI32 seed= pop::Processing::minimaLocalMap(grad,2,0);
        test.start("regionGrowingMergingLevel");
        seed = pop::Processing::regionGrowingMergingLevel(seed,in,50);
        test.end(pop::Visualization::labelToRandomRGB(seed));
    }

    {
        test.start("dynamic");
        out= pop::Processing::dynamic(in,20);
        test.end(out);
    }
    {
        Mat2UI8 grad= Processing::greylevelRange(pop::Processing::gradientMagnitudeGaussian(Mat2F32(in),5),0,255);
        grad= pop::Processing::dynamic(grad,70);
        Mat2UI32 seed= pop::Processing::minimaRegional(grad);
        test.start("watershed");
        seed = pop::Processing::watershed(seed,grad);
        test.end(pop::Visualization::labelToRandomRGB(seed));
    }
    {
        in.load(std::string(POP_PROJECT_SOURCE_DIR)+"/image/outil.bmp");
        in = GeometricalTransformation::scale(in,Vec2F32(10,10));
        in = pop::Processing::threshold(in,150);
        test.start("erosionRegionGrowing");
        out = Processing::erosionRegionGrowing(in,10,1);
        test.end(out);
    }
}



int main(){
    testMatN();
    processingTest();

    //    testAnamysis();
    return 1;
    Mat2UI8 img;
    //    img.load(std::string(POP_PROJECT_SOURCE_DIR)+"/image/outil.bmp");
    img.load("/home/vincent/Desktop/Gold.jpg");
    img.display();
    img = Processing::threshold(img,125);

    Mat2RGBUI8 mask(img);


    img = img.opposite();
    Mat2Vec2F32 vel;
    int dir=0;
    Vec2F32 kx = PDE::permeability(img,vel,dir,0.01);
    vel= GeometricalTransformation::scale(vel,Vec2F32(8));
    Mat2RGBUI8 c = Visualization::vectorField2DToArrows(vel,RGBUI8(0,0,255),RGBUI8(255,0,0));
    mask = GeometricalTransformation::scale(mask,Vec2F32(c.getDomain())/(mask.getDomain()));
    c = mask+c;
    c.display("velocity",false,false);
    dir=1;
    Vec2F32 ky = PDE::permeability(img,vel,dir,0.01);
    vel= GeometricalTransformation::scale(vel,Vec2F32(8));
    c = Visualization::vectorField2DToArrows(vel,RGBUI8(0,0,255),RGBUI8(255,0,0));
    c = mask+c;
    c.display("velocity",true,false);
    Mat2F32 K(2,2);
    K.setCol(0,kx);
    K.setCol(1,ky);
    std::cout<<K<<std::endl;


}
