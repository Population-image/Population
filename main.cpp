
#include"Population.h"//Single header
using namespace pop;//Population namespace

int main()
{
    {
        F32 porosity=0.8;
        F32 radius=10;
        DistributionDirac ddirac_radius(radius);

        F32 moment_order3 = pop::Statistics::moment(ddirac_radius,3,0,40);
        //std::sqrt(2)/12*(std::sqrt(24))^3
        F32 volume_expectation = moment_order3*std::sqrt(2.f)/12.f*std::pow(std::sqrt(24.f),3.f);
        Vec3F32 domain(200);//2d field domain
        F32 lambda=-std::log(porosity)/std::log(2.718f)/volume_expectation;
        std::cout<<lambda<<std::endl;
        ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process

        DistributionMultiVariateProduct dist_radius(ddirac_radius,ddirac_radius,ddirac_radius,ddirac_radius);

        DistributionDirac dplus(1/std::sqrt(3.f));
        DistributionDirac dminus(-1/std::sqrt(3.f));
        Vec<Distribution*> v_normal;
        v_normal.push_back(&dplus);
        v_normal.push_back(&dplus);
        v_normal.push_back(&dplus);

        v_normal.push_back(&dminus);
        v_normal.push_back(&dminus);
        v_normal.push_back(&dplus);

        v_normal.push_back(&dminus);
        v_normal.push_back(&dplus);
        v_normal.push_back(&dminus);

        v_normal.push_back(&dplus);
        v_normal.push_back(&dminus);
        v_normal.push_back(&dminus);

        DistributionMultiVariateProduct dist_normal(v_normal);
        DistributionMultiVariateProduct dist_dir(DistributionUniformReal(0,PI),DistributionUniformReal(0,PI),DistributionUniformReal(0,PI));

        RandomGeometry::polyhedra(grain,dist_radius,dist_normal,dist_dir);
        Mat3RGBUI8 img_germ = RandomGeometry::continuousToDiscrete(grain);
        Mat3UI8 img_germ_grey;
        img_germ_grey = img_germ;

        Mat2F32 m=  Analysis::histogram(img_germ_grey);
        std::cout<<"Realization porosity"<<m(0,1)<<std::endl;
        Scene3d scene;
        pop::Visualization::marchingCubeSmooth(scene,img_germ_grey);
        pop::Visualization::lineCube(scene,img_germ);
        scene.display();
    }
    Mat3UI8 m_rgb(4,4,8);
    m_rgb(1,1,1)=1;
    m_rgb(2,2,1)=1;
    m_rgb(2,1,1)=1;
    m_rgb(1,2,1)=1;
    m_rgb(1,1,2)=1;
    m_rgb(2,2,2)=1;
    m_rgb(2,1,2)=1;
    m_rgb(1,2,2)=1;


    m_rgb(1,1,4)=2;
    m_rgb(2,2,4)=2;
    m_rgb(2,1,4)=2;
    m_rgb(1,2,4)=2;
    m_rgb(1,1,5)=2;
    m_rgb(2,2,5)=2;
    m_rgb(2,1,5)=2;
    m_rgb(1,2,5)=2;
    Scene3d scene;
    Visualization::marchingCubeSmooth(scene,m_rgb);
    scene.display();
    Mat2UI8 img;//2d grey-level image object
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    int value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.pgm");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display();
    return 0;
}
