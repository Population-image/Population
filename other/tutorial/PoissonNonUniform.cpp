#include"Population.h"//Single header
using namespace pop;//Population namespace
int main()
{

    Mat2UI8 img;
    img.load("../Lena.bmp");
    //Initial porespacefield  with a local porosity equal to 1-(img(x)/255)
    Mat2F64 porespacefield =  Mat2F64(img)/255 ;
    std::cout<<"Macro porosity: "<<1-Analysis::meanValue(porespacefield)<<std::endl;//the
    double mean=5;
    double standard_deviation=2;
    DistributionNormal dnormal (mean,standard_deviation);//Normal generator
    double moment_order_2 = pop::Statistics::moment(dnormal,2,0,100);
    double surface_expectation = moment_order_2*3.14159265;
    //generation of lambda field
    Mat2F64 nonuniformlambdafield(porespacefield.getDomain());
    Mat2F64::IteratorEDomain it(nonuniformlambdafield.getIteratorEDomain());
    while(it.next()){
        double v = -std::log(1-porespacefield(it.x()))/std::log(2.718);
        nonuniformlambdafield(it.x())=v/surface_expectation;
    }
    //boolean model
    GermGrain2 grain = RandomGeometry::poissonPointProcessNonUniform(nonuniformlambdafield);//generate the 2d Poisson point process
    RandomGeometry::sphere(grain,dnormal);//dress the VecNs with disks
    Mat2UI8 lattice = RandomGeometry::continuousToDiscrete(grain);//conversion to lattice
    std::cout<<"Realization porosity: "<<1-Analysis::meanValue(lattice)/255.<<std::endl;
    lattice.display();
    lattice.save("../doc/image2/NonUniformLena.png");
    return 1;
}
