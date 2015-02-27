/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

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

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include"data/typeF/TypeF.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"
#include"data/distribution/DistributionAnalytic.h"
#include"data/typeF/TypeTraitsF.h"
#include"data/functor/FunctorF.h"
#include"algorithm/ForEachFunctor.h"
#include"data/mat/MatN.h"
#include"algorithm/ProcessingAdvanced.h"
#include"algorithm/Representation.h"
#include"data/notstable/graph/Graph.h"
#include"algorithm/Statistics.h"
#include"algorithm/AnalysisAdvanced.h"

/*!
     * \defgroup Analysis Analysis
     * \ingroup Algorithm
     * \brief Matrix In -> Measure (2-point correlation function, REV, histogram,...)
     *
     * After segmentation, space is partitioned into phases. One challenging problem deals with the ability to describe the geometrical organisation of these phases.
     * Quantitative knowledge of this organisation is important in many fields, for instance, in porous media, for the understanding of the role of geometric confinement in adsorption,
     * condensation, transport, reaction processes. As seen in my PhD, an analysis can be divided in two categories~:
     *   - geometrical information about forms, shapes and patterns like the average pore size, the mean curvature, the pore shape, the surface roughness, the structural correlation between pixels belonging either to the solid, to the interface or to the pore network,
     *   - topological information about connectivity like the percolation, the coordination number of the topological graph.

     * This class provides a collection of algorithms to analyse a 2d/3d matrix. A simple analysis can be as follows:

     * \code
     * #include"data/mat/MatN.h"
     * #include"algorithm/Analysis.h"
     * using namespace pop;
     * int main(){
     *     Mat2UI8 lena;
     *     lena.load("../image/Lena.bmp");
     *      std::cout<<Analysis::maxValue(lena)<<std::endl;
     *     return 0;
     * }
     * \endcode
     * Most of the algorithms return a matrix containing the information. To plot this information, you can convert it in DistributionRegularStep and display it as follows:
     * \code
     * Mat2F32 mverpore = Analysis::REVPorosity(grain,VecN<3,F32>(grain.getDomain())*0.5,200);
     * DistributionRegularStep dverpor(mverpore);
     * dverpor.display();
     * mverpore.saveAscii("ver.txt");
     * \endcode
     * Because this displayer is not nice, an another solution is to save the ascii data and to open it with gnuplot or matlab.\n
     * This code presents a extended analysis of a 3d matrix (to have the 3d visualization, uncomment the lines of code):
     * \code
     * Mat3UI8 porespace;
     * porespace.load("../image/spinodal.pgm");
     * porespace = pop::Processing::greylevelRemoveEmptyValue(porespace);//the porespace label is now 1 (before 255)
     * //porespace = porespace(Vec3I32(0,0,0),Vec3I32(50,50,50));//for the first execution, you can test the process in small sample
     * \endcode
     * \image html spinodal.png "Spinodal structure (see pop::Visualization::marchingCubeLevelSet)"
     * \code
     * //############ Representative Elementary Volume CHECKING ###########################
     * Mat2F32 mverpore = Analysis::REVPorosity(porespace,VecN<3,F32>(porespace.getDomain())*0.5,200);
     * mverpore.saveAscii("spinodal_VER.m");
     * \endcode
     * \image html spinodal_VER.png "The sample size is twice time larger than the stablilzation of the porosity measurement -> so REV oke"
     * \code
     * //############ METRIC ###################
     * Mat2F32 mhisto = Analysis::histogram(porespace);
     * std::cout<<"porespace fraction:"<<mhisto(1,1)<<std::endl;
     * Mat2F32 mchord = Analysis::chord(porespace);
     * mchord.saveAscii("spinodal_chord.m");
     * mchord = mchord.deleteCol(1);
     * DistributionRegularStep dchord_solid(mchord);
     * std::cout<<"Charateristic length of solid space "<<Statistics::moment(dchord_solid,1,0,300,1)<<std::endl;//  \sum_{i=0}^{300} i*d(i)=27.2
     * \endcode
     * \image html spinodal_chord.png "Peak following by exponantial decrease (for the meaning of this signature see Chap 2, p 37. In Handbook of Porous Media. P. Levitz)"
     * \code
     * Mat2F32 mcorre= Analysis::correlation(porespace);
     * mcorre.saveAscii("spinodal_corre.m");
     * \endcode
     * \image html spinodal_corr.png "No long-range oscillion=No periodic structure and REV oke as in REVPorosity"
     * \code
     * Mat3F32 corre= Analysis::correlationDirectionByFFT(porespace);
     * Mat3F32 corre= Analysis::correlationDirectionByFFT(porespace);
     * corre = GeometricalTransformation::translate(corre,corre.getDomain()/2);//centered
     * corre = corre(corre.getDomain()/2-Vec3I32(20,20,20),corre.getDomain()/2+Vec3I32(20,20,20));//take only the core (the correlation length is near 20)
     * Mat3UI8 dcorre;
     * dcorre= pop::Processing::greylevelRange(corre,1,255);//0 is black color so start at 1
     * Mat3RGBUI8 dcorregrad=Visualization::labelToRGBGradation(dcorre);//color gradation
     * Scene3d scene;
     * Visualization::plane(scene,dcorregrad,dcorre.getDomain()(0)/2,0);
     * Visualization::plane(scene,dcorregrad,dcorre.getDomain()(1)/2,1);
     * Visualization::plane(scene,dcorregrad,dcorre.getDomain()(2)/2,2);
     * Visualization::lineCube(scene,dcorregrad);
     * scene.display();//if you close the opengl windows, that stop the program in linux. So comment this line in linux to execute the rest of the code
     * \endcode
     * \image html spinodal_corr3d.png "We observe an isotropic 2-point correlation function->structure is istropic"
     * \code
     * Mat3UI8 fdistance;
     * Mat2F32 mldistance= Analysis::ldistance(porespace,2,fdistance);//euclidean distance only in Population library ;-)
     * mldistance.saveAscii("spinodal_ldistance.m");
     * Mat3RGBUI8 dcorregrad=Visualization::labelToRandomRGB(fdistance);//random color
     * Scene3d scene;
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(0)/2,0);
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(1)/2,1);
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(2)/2,2);
     * Visualization::lineCube(scene,dcorregrad);
     * scene.display();
     * \endcode
     * \image html spinodal_ldistance_pore.png "see Pore-Size Probability Density Function in Torquato's book"
     * \code
     * Mat3UI8 fgranulo;
     * //granulo of the solid space
     * Mat2F32 mlgranulo= Analysis::granulometryMatheron(porespace,2,fgranulo);//quite long algorithm in euclidean distance
     * mlgranulo.saveAscii("spinodal_granulo.m");
     * Mat3RGBUI8 dcorregrad=Visualization::labelToRandomRGB(fgranulo);//random color
     * Scene3d scene;
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(0)/2,0);
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(1)/2,1);
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(2)/2,2);
     * Visualization::lineCube(scene,dcorregrad);
     * scene.display();
     * \endcode
     * \image html spinodal_granulo.png "see X-ray microtomography characterisation of the changes in statistical homogeneity of an unsaturated sand during imbibitions"
     * \code
     * Mat2F32 mgeometrical = Analysis::geometricalTortuosity(porespace);
     * mgeometrical.saveAscii("spinodal_geometrical.m");
     * \endcode
     * We get  1.19465 in 3 directions // see Jeulin's paper estimation of tortuosity and reconstruction of geodesic paths in 3d
     * \code
     * //############ TOPOLOGY #######################
     * F32 euler = Analysis::eulerPoincare(porespace);
     * std::ofstream out("spinodal_euler.m");
     * out<<euler;//euler
     * out.close();
     * Mat2F32 mpercolationopening = Analysis::percolationOpening(porespace,2);//->charactertic length related to permeability
     * mpercolationopening.saveAscii("spinodal_percolationopening.m");//output is 6 in three direction-> the structure sill percolates after an opening of ball of radius of 6 but not with a radius of size 7
     * Mat2F32 mpercolationerosion = Analysis::percolationErosion(porespace,2);
     * mpercolationerosion.saveAscii("spinodal_percolationerosion.m");//output is 5 in three direction-> the structure sill percolates after an erosion of ball of radius of 5 but not with a radius of size 6


     * Mat3UI8 porespace_hole=   pop::Processing::holeFilling(porespace);
     * Mat3UI8 skeleton= Analysis::thinningAtConstantTopology(porespace_hole,"../file/topo24.dat");
     * Scene3d scene;
     * pop::Visualization::voxelSurface(scene,skeleton);
     * pop::Visualization::lineCube(scene,skeleton);
     * scene.display();
     * \endcode
     * \image html spinodal_skeleton.png "Topological skeleton"
     * \code
     * std::pair<Mat3UI8,Mat3UI8> vertex_edge = Analysis::fromSkeletonToVertexAndEdge (skeleton);
     * Mat3UI32 verteces = pop::Processing::clusterToLabel(vertex_edge.first,0);
     * Mat3UI32 edges = pop::Processing::clusterToLabel(vertex_edge.second,0);
     * pop::Visualization::voxelSurface(scene,pop::Visualization::labelToRandomRGB(edges));
     * pop::Visualization::lineCube(scene,edges);
     * scene.display();
     * \endcode
     * \image html spinodal_edge.png "Labelisation of the edges of the topological skeleton"
     * \code
     * int tore;
     * GraphAdjencyList<Vec3I32> g = Analysis::linkEdgeVertex(verteces,edges,tore);
     * Scene3d scene;
     * pop::Visualization::graph(scene,g);
     * pop::Visualization::lineCube(scene,edges);
     * scene.display();
     * \endcode
     * \image html spinodal_graph.png "Topological graph"
     * \code
     * std::cout<<euler/g.sizeVertex()<<std::endl;//N_3/alpha_0 normalised topogical characteristic (connectivity number in my phd)

     * //############ PHYSICAL ###################
     * Mat2F32 mdiffusion = PDE::randomWalk(porespace);
     * mdiffusion.saveAscii("spinodal_self_diffusion.m");
     * \endcode
     * \image html spinodal_self_diffusion.png "Coefficient of self diffusion"
     * \code
     * MatN<3,Vec3F32> velocityfield;
     * Mat2F32 K(3,3);
     * VecF32 kx = PDE::permeability(porespace,velocityfield,0,0.05);//permeability in x-direction
     * VecF32 ky = PDE::permeability(porespace,velocityfield,1,0.05);//permeability in y-direction
     * VecF32 kz = PDE::permeability(porespace,velocityfield,2,0.05);//permeability in z-direction
     * //merge the results in the permeability matrix
     * K.setCol(0,kx);
     * K.setCol(1,ky);
     * K.setCol(2,kz);

     * //display the norm of the last valocity field
     * Mat3F32 velocityfield_norm(velocityfield);
     * ForEachDomain3D(x,velocityfield)
     * {
     *     velocityfield_norm(x)=normValue(velocityfield(x));
     * }
     * Mat3RGBUI8 velocityfield_norm_grad= pop::Visualization::labelToRGBGradation(velocityfield_norm);
     * Scene3d scene;
     * Visualization::plane(scene,velocityfield_norm_grad,velocityfield_norm_grad.getDomain()(0)/2,0);
     * Visualization::plane(scene,velocityfield_norm_grad,velocityfield_norm_grad.getDomain()(1)/2,1);
     * Visualization::plane(scene,velocityfield_norm_grad,velocityfield_norm_grad.getDomain()(2)/2,2);
     * Visualization::lineCube(scene,velocityfield_norm_grad);
     * scene.display();
     * \endcode
     * \image html spinodal_permeability.png "Amplitude of the velocity field"


*/

namespace pop
{
struct POP_EXPORTS Analysis
{

    /*!
     * \class pop::Analysis
     * \ingroup Analysis
     * \brief Analyse a 2D/3D matrix
     * \author Tariel Vincent
     *
     * After segmentation, space is partitioned into phases. One challenging problem deals with the ability to describe the geometrical organisation of these phases.
     * Quantitative knowledge of this organisation is important in many fields, for instance, in porous media, for the understanding of the role of geometric confinement in adsorption,
     * condensation, transport, reaction processes. As seen in my PhD, an analysis can be divided in two categories~:
     *   - geometrical information about forms, shapes and patterns like the average pore size, the mean curvature, the pore shape, the surface roughness, the structural correlation between pixels belonging either to the solid, to the interface or to the pore network,
     *   - topological information about connectivity like the percolation, the coordination number of the topological graph.

     * This class provides a collection of algorithms to analyse a 2d/3d matrix. A simple analysis can be as follows:

     * \code
     * #include"data/mat/MatN.h"
     * #include"algorithm/Analysis.h"
     * using namespace pop;
     * int main(){
     *     Mat2UI8 lena;
     *     lena.load("../image/Lena.bmp");
     *      std::cout<<Analysis::maxValue(lena)<<std::endl;
     *     return 0;
     * }
     * \endcode
     * Most of the algorithms return a matrix containing the information. To plot this information, you can convert it in Distribution and display it as follows:
     * \code
     * Mat2F32 mverpore = Analysis::REVPorosity(grain,VecN<3,F32>(grain.getDomain())*0.5,200);
     * DistributionRegularStep dverpor(mverpore);
     * dverpor.display();
     * mverpore.saveAscii("ver.txt");
     * \endcode
     * Because this displayer is not nice, an another solution is to save the ascii data and to open it with gnuplot or matlab.\n
     * This code presents a extended analysis of a 3d matrix (to have the 3d visualization, uncomment the lines of code):
     * \code
     * Mat3UI8 porespace;
     * porespace.load("../image/spinodal.pgm");
     * porespace = pop::Processing::greylevelRemoveEmptyValue(porespace);//the porespace label is now 1 (before 255)
     * //porespace = porespace(Vec3I32(0,0,0),Vec3I32(50,50,50));//for the first execution, you can test the process in small sample
     * \endcode
     * \image html spinodal.png "Spinodal structure (see pop::Visualization::marchingCubeLevelSet)"
     * \code
     * //############ Representative Elementary Volume CHECKING ###########################
     * Mat2F32 mverpore = Analysis::REVPorosity(porespace,VecN<3,F32>(porespace.getDomain())*0.5,200);
     * mverpore.saveAscii("spinodal_VER.m");
     * \endcode
     * \image html spinodal_VER.png "The sample size is twice time larger than the size when the porosity measurement is stable->so REV oke"
     * \code
     * //############ METRIC ###################
     * Mat2F32 mhisto = Analysis::histogram(porespace);
     * std::cout<<"porespace fraction:"<<mhisto(1,1)<<std::endl;
     * Mat2F32 mchord = Analysis::chord(porespace);
     * mchord.saveAscii("spinodal_chord.m");
     * mchord = mchord.deleteCol(1);
     * DistributionRegularStep dchord_solid(mchord);
     * std::cout<<"Charateristic length of solid space "<<Statistics::moment(dchord_solid,1,0,300,1)<<std::endl;//  \sum_{i=0}^{300} i*d(i)=27.2
     * \endcode
     * \image html spinodal_chord.png "Peak following by exponantial decrease (for the meaning of this signature see Chap 2, p 37. In Handbook of Porous Media. P. Levitz)"
     * \code
     * Mat2F32 mcorre= Analysis::correlation(porespace);
     * mcorre.saveAscii("spinodal_corre.m");
     * \endcode
     * \image html spinodal_corr.png "No long-range oscillion=No periodic structure and REV oke as in REVPorosity"
     * \code
     * Mat3F32 corre= Analysis::correlationDirectionByFFT(porespace);
     * Mat3F32 corre= Analysis::correlationDirectionByFFT(porespace);
     * corre = GeometricalTransformation::translate(corre,corre.getDomain()/2);//centered
     * corre = corre(corre.getDomain()/2-Vec3I32(20,20,20),corre.getDomain()/2+Vec3I32(20,20,20));//take only the core (the correlation length is near 20)
     * Mat3UI8 dcorre;
     * dcorre= pop::Processing::greylevelRange(corre,1,255);//0 is black color so start at 1
     * Mat3RGBUI8 dcorregrad=Visualization::labelToRGBGradation(dcorre);//color gradation
     * Scene3d scene;
     * Visualization::plane(scene,dcorregrad,dcorre.getDomain()(0)/2,0);
     * Visualization::plane(scene,dcorregrad,dcorre.getDomain()(1)/2,1);
     * Visualization::plane(scene,dcorregrad,dcorre.getDomain()(2)/2,2);
     * Visualization::lineCube(scene,dcorregrad);
     * scene.display();//if you close the opengl windows, that stop the program in linux. So comment this line in linux to execute the rest of the code
     * \endcode
     * \image html spinodal_corr3d.png "We observe an isotropic 2-point correlation function->structure is istropic"
     * \code
     * Mat3UI8 fdistance;
     * Mat2F32 mldistance= Analysis::ldistance(porespace,2,fdistance);//euclidean distance only in Population library ;-)
     * mldistance.saveAscii("spinodal_ldistance.m");
     * Mat3RGBUI8 dcorregrad=Visualization::labelToRandomRGB(fdistance);//random color
     * Scene3d scene;
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(0)/2,0);
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(1)/2,1);
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(2)/2,2);
     * Visualization::lineCube(scene,dcorregrad);
     * scene.display();
     * \endcode
     * \image html spinodal_ldistance_pore.png "see Pore-Size Probability Density Function in Torquato's book"
     * \code
     * Mat3UI8 fgranulo;
     * //granulo of the solid space
     * Mat2F32 mlgranulo= Analysis::granulometryMatheron(porespace,2,fgranulo);//quite long algorithm in euclidean distance
     * mlgranulo.saveAscii("spinodal_granulo.m");
     * Mat3RGBUI8 dcorregrad=Visualization::labelToRandomRGB(fgranulo);//random color
     * Scene3d scene;
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(0)/2,0);
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(1)/2,1);
     * Visualization::plane(scene,dcorregrad,dcorregrad.getDomain()(2)/2,2);
     * Visualization::lineCube(scene,dcorregrad);
     * scene.display();
     * \endcode
     * \image html spinodal_granulo.png "see X-ray microtomography characterisation of the changes in statistical homogeneity of an unsaturated sand during imbibitions"
     * \code
     * Mat2F32 mgeometrical = Analysis::geometricalTortuosity(porespace);
     * mgeometrical.saveAscii("spinodal_geometrical.m");
     * \endcode
     * We get  1.19465 in 3 directions // see Jeulin's paper estimation of tortuosity and reconstruction of geodesic paths in 3d
     * \code
     * //############ TOPOLOGY #######################
     * F32 euler = Analysis::eulerPoincare(porespace);
     * std::ofstream out("spinodal_euler.m");
     * out<<euler;//euler
     * out.close();
     * Mat2F32 mpercolationopening = Analysis::percolationOpening(porespace,2);//->charactertic length related to permeability
     * mpercolationopening.saveAscii("spinodal_percolationopening.m");//output is 6 in three direction-> the structure sill percolates after an opening of ball of radius of 6 but not with a radius of size 7
     * Mat2F32 mpercolationerosion = Analysis::percolationErosion(porespace,2);
     * mpercolationerosion.saveAscii("spinodal_percolationerosion.m");//output is 5 in three direction-> the structure sill percolates after an erosion of ball of radius of 5 but not with a radius of size 6


     * Mat3UI8 porespace_hole=   pop::Processing::holeFilling(porespace);
     * Mat3UI8 skeleton= Analysis::thinningAtConstantTopology(porespace_hole,"../file/topo24.dat");
     * Scene3d scene;
     * pop::Visualization::voxelSurface(scene,skeleton);
     * pop::Visualization::lineCube(scene,skeleton);
     * scene.display();
     * \endcode
     * \image html spinodal_skeleton.png "Topological skeleton"
     * \code
     * std::pair<Mat3UI8,Mat3UI8> vertex_edge = Analysis::fromSkeletonToVertexAndEdge (skeleton);
     * Mat3UI32 verteces = pop::Processing::clusterToLabel(vertex_edge.first,0);
     * Mat3UI32 edges = pop::Processing::clusterToLabel(vertex_edge.second,0);
     * pop::Visualization::voxelSurface(scene,pop::Visualization::labelToRandomRGB(edges));
     * pop::Visualization::lineCube(scene,edges);
     * scene.display();
     * \endcode
     * \image html spinodal_edge.png "Labelisation of the edges of the topological skeleton"
     * \code
     * int tore;
     * GraphAdjencyList<Vec3I32> g = Analysis::linkEdgeVertex(verteces,edges,tore);
     * Scene3d scene;
     * pop::Visualization::graph(scene,g);
     * pop::Visualization::lineCube(scene,edges);
     * scene.display();
     * \endcode
     * \image html spinodal_graph.png "Topological graph"
     * \code
     * std::cout<<euler/g.sizeVertex()<<std::endl;//N_3/alpha_0 normalised topogical characteristic (connectivity number in my phd)

     * //############ PHYSICAL ###################
     * Mat2F32 mdiffusion = PDE::randomWalk(porespace);
     * mdiffusion.saveAscii("spinodal_self_diffusion.m");
     * \endcode
     * \image html spinodal_self_diffusion.png "Coefficient of self diffusion"
     * \code
     * MatN<3,Vec3F32> velocityfield;
     * Mat2F32 K(3,3);
     * VecF32 kx = PDE::permeability(porespace,velocityfield,0,0.05);//permeability in x-direction
     * VecF32 ky = PDE::permeability(porespace,velocityfield,1,0.05);//permeability in y-direction
     * VecF32 kz = PDE::permeability(porespace,velocityfield,2,0.05);//permeability in z-direction
     * //merge the results in the permeability matrix
     * K.setCol(0,kx);
     * K.setCol(1,ky);
     * K.setCol(2,kz);

     * //display the norm of the last valocity field
     * Mat3F32 velocityfield_norm(velocityfield);
     * ForEachDomain3D(x,velocityfield)
     * {
     *     velocityfield_norm(x)=normValue(velocityfield(x));
     * }
     * Mat3RGBUI8 velocityfield_norm_grad= pop::Visualization::labelToRGBGradation(velocityfield_norm);
     * Scene3d scene;
     * Visualization::plane(scene,velocityfield_norm_grad,velocityfield_norm_grad.getDomain()(0)/2,0);
     * Visualization::plane(scene,velocityfield_norm_grad,velocityfield_norm_grad.getDomain()(1)/2,1);
     * Visualization::plane(scene,velocityfield_norm_grad,velocityfield_norm_grad.getDomain()(2)/2,2);
     * Visualization::lineCube(scene,velocityfield_norm_grad);
     * scene.display();
     * \endcode
     * \image html spinodal_permeability.png "Amplitude of the velocity field"
     * For an extented analysis, you can decompose the structure in term of elmentary parts and to analyse statiscally this elements as follows:
     * \code
     * F32 porosity=0.2;
     * DistributionNormal dnormal(10,0.1);//Poisson generator
     * F32 moment_order_2 = pop::Statistics::moment(dnormal,2,0,1024);
     * F32 surface_expectation = moment_order_2*3.14159265;
     * Vec2F32 domain(2048);//2d field domain
     * F32 N=-std::log(porosity)/std::log(2.718)/surface_expectation;
     * ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,N);//generate the 2d Poisson Pointd process
     * RandomGeometry::sphere(grain,dnormal);
     * Mat2RGBUI8 lattice = RandomGeometry::continuousToDiscrete(grain);

     * //DECOMPOSITION OF THE PORE SPACE IN TERM OF ELEMENTARY PORES
     * Mat2UI8 porespace;
     * porespace = lattice;
     * Mat2UI8 inverse(porespace);
     * inverse = inverse.opposite();
     * Mat2F32 dist = pop::Processing::distanceEuclidean(inverse);
     * Mat2UI16 distl;
     * distl= dist;
     * distl = distl.opposite();
     * distl = pop::Processing::dynamic(distl,2);
     * Mat2UI32 minima = pop::Processing::minimaRegional(distl,0);
     * Mat2UI32 water = pop::Processing::watershed(minima,distl,porespace,1);
     * water.saveAscii("_label.pgm");
     * //ANALYSE THE PORE SPACE
     * PDE::allenCahn(water,porespace,5000);
     * water=pop::Processing::greylevelRemoveEmptyValue(water);
     * VecF32 varea = Analysis::areaByLabel(water);
     * DistributionRegularStep dvarea = pop::Statistics::computedStaticticsFromRealRealizations(varea,0.1);
     * Mat2F32 mvarea = dvarea.toMatrix();
     * mvarea.saveAscii("label_area.m");
     * VecF32 vcontact = Analysis::perimeterContactBetweenLabel(water);
     * DistributionRegularStep dcontact = pop::Statistics::computedStaticticsFromRealRealizations(vcontact,0.1);
     * Mat2F32 mcontact = dcontact.toMatrix();
     * mcontact.saveAscii("label_contact.m");
     * VecF32 vferet = Analysis::feretDiameterByLabel(water);
     * DistributionRegularStep dferet = pop::Statistics::computedStaticticsFromRealRealizations(vferet,0.1);
     * Mat2F32 mferet = dferet.toMatrix();
     * mferet.saveAscii("label_feret.m");
     * \endcode
     *
    */
    //-------------------------------------
    //
    //! \name Profile along a segment
    //@{
    //-------------------------------------
    /*!
     * \param f input matrix
     * \param x1 first segment point
     * \param x2 second segment point
     * \return  profile along the segment
     *
     * \code
     * Mat2UI8 img;
     * img.load("lena.pgm");
     * DistributionRegularStep(Analysis::profile(img,Vec2I32(800,0),Vec2I32(800,img.getDomain()(1)-1))).display();
     * \endcode
     */
    template<int DIM,typename Type>
    static Mat2F32 profile(const MatN<DIM,Type> f, const VecN<DIM,int>& x1,const VecN<DIM,int>& x2){
        F32 dist = (x2-x1).norm();
        VecN<DIM,F32> direction = VecN<DIM,F32>(x2-x1)/dist;
        VecN<DIM,F32> x=x1;
        Mat2F32 profile(std::floor(dist),2);
        for(int i =0;i<profile.getDomain()(0);i++){
            profile(i,0)=i;
            if(f.isValid(x)){
                profile(i,1)=f.interpolationBilinear(x);
            }
            x+=direction;
        }
        return profile;
    }

    //@}

    //-------------------------------------
    //
    //! \name Representative Elementary Volume
    //@{
    //-------------------------------------

    /*!
     * \param f input matrix
     * \param x center point
     * \param rmax max radius
     * \param norm distance norm
     * \return  Mat2F32 M
     *
     * Calculate the grey-level histogram inside the ball centered in x by progressively increased the radius from 0 to rmax\n
     * M(i,0)=i and M(i,j) = \f$\frac{|X_{j-1}\cap B(x,i)|}{|B(x,i)|}\f$ where \f$ B(x,i)=\{x': |x'-x|<i \}\f$ the ball centered in x of radius i
     * and \f$X_{j}=\{x:f(x)=j \}\f$ the level set of f
     * \code
        Mat2UI8 iex;
        iex.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));
        Mat2F32 mverpore = Analysis::REVHistogram(iex,VecN<2,F32>(iex.getDomain())*0.5,250);

        VecF32 vindex = mverpore.getCol(0);//get the first column containing the grey-level range
        VecF32 v100 = mverpore.getCol(100);//get the col containing the histogram for r=100
        VecF32 v150 = mverpore.getCol(150);
        VecF32 v200 = mverpore.getCol(200);

        Mat2F32 mhistoradius100(v100.size(),2);
        mhistoradius100.setCol(0,vindex);
        mhistoradius100.setCol(1,v100);

        Mat2F32 mhistoradius150(v150.size(),2);
        mhistoradius150.setCol(0,vindex);
        mhistoradius150.setCol(1,v150);

        Mat2F32 mhistoradius200(v200.size(),2);
        mhistoradius200.setCol(0,vindex);
        mhistoradius200.setCol(1,v200);



        DistributionRegularStep d100(mhistoradius100);
        DistributionRegularStep d150(mhistoradius150);
        DistributionRegularStep d200(mhistoradius200);
        DistributionDisplay::display(d100,d150,d200,d100.getXmin(),d150.getXmax());
     * \endcode
     * \image html REVhistogram.png "Histogram in balls of different radius"
    */

    template<int DIM,typename PixelType>
    static  Mat2F32 REVHistogram(const MatN<DIM,PixelType> & f, typename MatN<DIM,PixelType>::E x, int rmax=10000,int norm=2 )
    {
        typename MatN<DIM,PixelType>::IteratorEDomain ittotal(f.getIteratorEDomain());
        int max_value = Analysis::maxValue(f);
        int maxsize = NumericLimits<int>::maximumRange();
        for(int i =0;i<DIM;i++){
            maxsize = minimum(maxsize,f.getDomain()(i));
        }
        if(rmax>maxsize/2)
            rmax=maxsize/2;

        Mat2F32 m(max_value+1,1);
        for(unsigned int i =0;i<m.sizeI();i++)
            m(i,0)=i;

        for( int r =0;r<rmax;r++){
            m.resizeInformation(m.sizeI(),m.sizeJ()+1);
            typename MatN<DIM,PixelType>::E R( r*2+1);
            typename MatN<DIM,PixelType>::IteratorEDomain it(R);
            bool inside =true;
            typename MatN<DIM,PixelType>::E add =x- r;
            while(it.next()&&inside==true){
                typename MatN<DIM,PixelType>::E xprime = it.x()-r;
                if(xprime.norm(norm)<=r){
                    typename MatN<DIM,PixelType>::E xxprime = it.x()+add;
                    if(f.isValid(xxprime)){
                        m(f(xxprime),r+1) ++;
                    }else{
                        inside = false;
                    }
                }
            }
            if(inside==true){
                int count =0;
                for(unsigned int i =0;i<m.sizeI();i++){
                    count +=m(i,r+1);
                }
                for(unsigned int i =0;i<m.sizeI();i++){
                    m(i,r+1)/=count;
                }
            }else{
                r=rmax;
                m.resizeInformation(m.sizeI(),m.sizeJ()-1);
            }
        }
        return m;

    }
    /*!
     * \param bin input binary matrix
     * \param x center point
     * \param rmax max radius
      * \param norm distance norm
     * \return  Mat2F32 M
     *
     *
     *  M(i,0)=i, M(i,1)=\f$\frac{|X^c\cap B(x,i)|}{|B(x,i)|}\f$ such that  \f$X^c=\{x:bin(x)=0\}\f$ is the pore space and \f$B(x,i)\f$
     * is the ball centered in x of radius i.
    */

    template<int DIM>
    static  Mat2F32 REVPorosity(const MatN<DIM,UI8> & bin, typename MatN<DIM,UI8>::E x, int rmax=10000,int norm=2)
    {

        Mat2F32 m;
        int maxsize = NumericLimits<int>::maximumRange();
        for(int i =0;i<DIM;i++){
            maxsize = minimum(maxsize,bin.getDomain()(i));
        }
        if(rmax>maxsize/2)
            rmax=maxsize/2;


        for(int r =1;r<rmax;r++){
            F32 rr = r*r;
            m.resizeInformation(m.sizeI()+1,2);
            m(r-1,0) = r;
            typename MatN<DIM,UI8>::E R( r*2+1);
            typename MatN<DIM,UI8>::IteratorEDomain it(R);
            bool inside =true;
            int countpore=0;
            int count=0;

            typename MatN<DIM,UI8>::E add =x- r;
            while(it.next()&&inside==true){
                typename MatN<DIM,UI8>::E xprime = it.x()-r;
                F32 rrcurent = xprime.norm(norm);
                if(rrcurent<=rr){
                    typename MatN<DIM,UI8>::E xxprime = it.x()+add;
                    if(bin.isValid(xxprime)){
                        count++;
                        if(bin(xxprime)==0) countpore++;
                    }else{
                        inside = false;
                    }
                }
            }
            if(inside==true){

                m(r-1,1) = 1.0*countpore/count;
            }else
                r=rmax;
        }
        return m;
    }

    //@}

    //-------------------------------------
    //
    //! \name Metric
    //@{
    //-------------------------------------


    /*!
     *  \brief return the maximum value
     * \param f input function
     * \return max value with the pixel/voxel type
     *
     * Return the maximum value of the matrix, \f$\max_{\forall x\in E}f(x) \f$.\n
     * For instance, this code
     * \code
    Mat2RGBUI8 lena;
    lena.load("../image/Lena.bmp");
    std::cout<<Analysis::maxValue(lena)<<std::endl;
    return 1;
     \endcode
     produces this output 255\<C\>246\<C\>205\<C\>.
    */
    template<int DIM,typename PixelType>
    static  PixelType  maxValue(const MatN<DIM,PixelType> & f){

        typename MatN<DIM,PixelType>::IteratorEDomain it(f.getIteratorEDomain());
        return AnalysisAdvanced::maxValue(f,it);
    }

    /*!
     *  \brief return the minimum value
     * \param f input function
     * \return min value with the pixel/voxel type
     *
     * Return the minimum value of the matrix, \f$\min_{\forall x\in E}f(x) \f$.\n
     * For instance, this code
     * \code
    Mat2RGBUI8 lena;
    lena.load("../image/Lena.bmp");
    std::cout<<Analysis::minValue(lena)<<std::endl;
    return 1;
     \endcode
     produces this output 53<C\>0\<C\>46\<C\>.
    */
    template<int DIM,typename PixelType>
    static  PixelType  minValue(const MatN<DIM,PixelType> & f){

        typename MatN<DIM,PixelType>::IteratorEDomain it(f.getIteratorEDomain());
        return AnalysisAdvanced::minValue(f,it);
    }
    /*!
     * \brief return the mean value
     * \param f input function
     * \return mean value with a float type
     *
     * Return the mean value of the matrix:  \f$\frac{ \int_{x\in E} f(x)dx}{\int_{x\in E} 1 dx} \f$
     * For instance, this code
     * \code
    Mat2UI8 lena;
    lena.load("../image/Lena.bmp");
    F32 mean = Analysis::meanValue(lena);
    std::cout<<mean<<std::endl;
    \endcode
    */
    template<int DIM,typename PixelType>
    static F32 meanValue(const MatN<DIM,PixelType> & f)
    {
        typename MatN<DIM,PixelType>::IteratorEDomain it(f.getIteratorEDomain());
        return AnalysisAdvanced::meanValue( f,it);
    }
    /*!
     * \brief return the standard deviation value
     * \param f input function
     * \return mean value with a float type
     *
     * Return the standard deviation value of the matrix:  \f$\sigma=\sqrt{\frac{ \int_{x\in E} (f(x)-\mu)*(f(x)-\mu)dx}{\int_{x\in E} 1 dx}}= \sqrt{\operatorname E[(f(x) - \mu)^2]} \f$ with \f$\mu\f$ the mean value
     * For instance, this code
     * \code
    Mat2UI8 lena;
    lena.load("../image/Lena.bmp");
    F32 mean = Analysis::meanValue(lena);
    std::cout<<mean<<std::endl;
    \endcode

    */
    template<int DIM,typename PixelType>
    static F32 standardDeviationValue(const MatN<DIM,PixelType> & f)
    {
        typename MatN<DIM,PixelType>::IteratorEDomain it(f.getIteratorEDomain());
        return AnalysisAdvanced::standardDeviationValue( f,it);
    }

    /*!
     * \param f input grey-level matrix
     * \return  Mat2F32 M
     *
     *
     *  M(i,0)=i, M(i,1)=P(f(x)=i)
     * \code
     * Mat2UI8 img;
     * img.load("../image/Lena.bmp");
     * Analysis analysis;
     * Mat2F32 m = analysis.histogram(img);
     * DistributionRegularStep d(m);
     * d.display(0,255);
     * \endcode
     * \sa Matrix Distribution
    */
    template<int DIM,typename PixelType>
    static Mat2F32 histogram(const MatN<DIM,PixelType> & f)
    {
        typename MatN<DIM,PixelType>::IteratorEDomain it(f.getIteratorEDomain());
        return AnalysisAdvanced::histogram(f,it);
    }

    /*!
     * \param f input matrix
     * \return  Mat2F32 M
     *
     * M(i,0)=i and M(i,1) = \f$|X_{i}|\f$ where
     *  \f$X_{j}=\{x:f(x)=j \}\f$ is the level set of f and \f$|X|\f$  is the cardinal of the set
     * \code
     * Mat2UI8 img;
     * img.load("../image/Lena.bmp");
     * Analysis analysis;
     * Mat2F32 m = analysis.area(img);
     * DistributionRegularStep d(m);
     * d.display(0,255);
     * \endcode

    */
    template<int DIM,typename PixelType>
    static Mat2F32 area(const MatN<DIM,PixelType> & f)
    {
        typename MatN<DIM,PixelType>::IteratorEDomain it(f.getIteratorEDomain());
        return AnalysisAdvanced::area(f,it);
    }
    /*!
     * \param f input matrixE
     * \return  Mat2F32 M
     *
     * M(i,0)=i and M(i,1) = \f$|\partial X_{i}|\f$ where
     * \f$X_{j}=\{x:f(x)=j \}\f$ the level set of f and \f$ \partial X\f$  the set boundary. We count the number of edges where one adjacent pixel is occupied by the phase ''i''
     *  and at one other pixel adjacent by an other phase :
     * \f[s_i=\frac{\sum_{x\in\Omega}\sum_{y\in N(x)}1_{f(x)=i}1_{f(y)\neq i}   }{2|\Omega|}\f] where N(x) is the 4-connex pixel  of x in 2D,  the 6-connex voxel  of x in 3D.
     */
    template<int DIM,typename PixelType>
    static Mat2F32 perimeter(const MatN<DIM,PixelType> & f)
    {
        typename MatN<DIM,PixelType>::IteratorEDomain itg(f.getIteratorEDomain());
        typename MatN<DIM,PixelType>::IteratorENeighborhood itn(f.getIteratorENeighborhood(1,1));
        return AnalysisAdvanced::perimeter(f,itg,itn);
    }

    /*!
     * \param bin input binary matrix
     * \return  Mat2F32 M
     *
     *  Allow the evaluation of the fractal dimension that is the slot of the graph defined by the first column of the Mat2F32 for X-axis and the second one for the Y-axis \n
     *  M(i,0)=log(r), M(i,1) =log(N(r)), where N(r) is the number of boxes of a size r needed to cover the binary set.
     * \code
     * Mat2UI8 img = RandomGeometry::diffusionLimitedAggregation2D(1024,30000);
     * img.display();
     * Mat2F32 m=  Analysis::fractalBox(img);
     * m.saveAscii("fractal.m","#log(r)\tlog(N)");
     * return 0;
     * \endcode
     *
     * \image html DLA.png
     * \image html DLAplot.png
    */

    template<int DIM>
    static  Mat2F32 fractalBox(const MatN<DIM,UI8> & bin )
    {
        int mini=NumericLimits<int>::maximumRange();
        for(int i = 0; i<DIM;i++)
            mini=minimum(mini,bin.getDomain()(i));
        Mat2F32 m;
        int i = mini/10;
        while(i>1)
        {
            m.resizeInformation(m.sizeI()+1,2);
            typename MatN<DIM,UI8>::E boxlength=i;
            typename MatN<DIM,UI8>::IteratorEDomain itlocal(boxlength);

            typename MatN<DIM,UI8>::E sizeglobal;
            for(int k = 0; k<DIM;k++)
                sizeglobal(k) = std::floor(bin.getDomain()(k)*1.0/i);
            typename MatN<DIM,UI8>::IteratorEDomain itglobal(sizeglobal);
            int count = 0;
            itglobal.init();
            while(itglobal.next())
            {
                typename MatN<DIM,UI8>::E x = itglobal.x()*i;
                bool touch = false;
                itlocal.init();
                while(itlocal.next())
                {
                    typename MatN<DIM,UI8>::E z = itlocal.x() + x;
                    if(bin(z)!=0)
                        touch=true;
                }
                if(touch==true)
                    count++;
            }
            F32 vv = std::log(i*1.0)/std::log(2.);
            m(m.sizeI()-1,0)=vv;
            m(m.sizeI()-1,1)=std::log(count*1.0)/std::log(2.);
            //            std::cout<<m<<std::endl;
            if(i==1)
                i=0;
            else
                i/=1.25;
        }
        return m;
    }


    /*!
     * \param f input labelled matrix
     * \param nbrtest nbrtest for sampling
     * \param length max length for the correlation
     * \return  correlation ffunction
     *
     *
     *  M(i<length,0)=i, M(i,j) =P(f(x)=j and f(x+r)=j) with r any vector of size i. To estimate this value, we sample nbrtest*length*dim times.
     *
     *
     * \code
        Mat2UI8 img;//2d grey-level image object
        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));
        img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
        int value;
        Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
        threshold = Processing::greylevelRemoveEmptyValue(threshold);//scale the binary level from [0-255] to [0-1]
        Mat2F32 m_corr_simulated = Analysis::correlation(threshold,200);
        std::cout<<m_corr_simulated<<std::endl;
        m_corr_simulated.saveAscii("m_corre.m");
        return 1;
     * \endcode
     */

    template<int DIM,typename PixelType>
    static  Mat2F32 correlation(const MatN<DIM,PixelType> & f, int length=200, int nbrtest=100000 )
    {
        if(nbrtest<100)
            nbrtest=100;
        int maxsize = NumericLimits<int>::maximumRange();
        for(int i =0;i<DIM;i++){
            maxsize = minimum(maxsize,f.getDomain()(i));
        }
        if(length<0)
            length=maxsize/2;
        if(length>maxsize)
            length=maxsize-1;

        PixelType value = Analysis::maxValue(f);

        Mat2F32 m(length+1,value+2),mcount(length+1,value+2);
        std::vector<DistributionUniformInt> dist;
        for(int i=0;i<DIM;i++)
            dist.push_back(DistributionUniformInt(0,f.getDomain()(i)-1));

        for(int index_test=0;index_test<nbrtest;index_test++)
        {
            typename MatN<DIM,PixelType>::E  x ;
            for(int ii=0;ii<DIM;ii++)
                x(ii)=dist[ii].randomVariable();
            PixelType etat1 = f(x);
            for(int i=0;i<DIM;i++)
            {
                typename MatN<DIM,PixelType>::E y=x;
                y(i)=x(i)-length;
                for(;y(i)<=x(i)+length;y(i)++)
                {
                    int r = absolute(y(i)-x(i));
                    typename MatN<DIM,PixelType>::E z=y;
                    if(f.isValid(z))
                    {
                        PixelType etat2 = f(z);
                        for(int k=1;k<=(int)value+1;k++)
                        {
                            mcount(r,k)++;
                            if(k==etat1+1&& etat1 ==  etat2)
                                m(r,k)++;
                        }


                    }

                }


            }
        }
        for(unsigned int j=0;j<m.sizeJ();j++){
            for(unsigned int i=0;i<m.sizeI();i++){
                if(j==0)
                    m(i,j)=i;
                else
                    m(i,j)/=mcount(i,j);
            }
        }
        return m;
    }

    /*!
     * \param f input grey-level matrix
     * \param nbrtest nbrtest for sampling
     * \param length max length for the correlation
     * \return  Mat2F32 M
     *
     *
     *  M(i<length,0)=i, M(i,1)=\f$ \frac{\operatorname{E}[(f(x) - \mu)(f(x+i) - \mu)]}{\sigma^2}\f$ with \f$\mu\f$ the mean value and \f$\sigma\f$ the standard deviation see http://en.wikipedia.org/wiki/Autocorrelation. To estimate this value, we sample nbrtest*length*dim times.
    */
    template<int DIM,typename PixelType>
    static  Mat2F32 autoCorrelationFunctionGreyLevel(const MatN<DIM,PixelType> & f, int length=100, int nbrtest=100000 )
    {
        typename MatN<DIM,PixelType>::IteratorEDomain it(f.getIteratorEDomain());
        int maxsize = NumericLimits<int>::maximumRange();
        for(int i =0;i<DIM;i++){
            maxsize = minimum(maxsize,f.getDomain()(i));
        }
        if(length<0)
            length=maxsize/2;
        if(length>maxsize)
            length=maxsize-1;
        F32 mu = meanValue(f);
        F32 sigma = standardDeviationValue(f);
        Mat2F32 m(length+1,2),mcount(length+1,2);
        std::vector<DistributionUniformInt> dist;
        for(int i=0;i<DIM;i++)
            dist.push_back(DistributionUniformInt(0,f.getDomain()(i)-1));
        for(int index_test=0;index_test<nbrtest;index_test++)
        {
            typename MatN<DIM,PixelType>::E  x ;
            for(int i=0;i<DIM;i++)
                x(i)=dist[i].randomVariable();
            for(int i=0;i<DIM;i++)
            {
                typename MatN<DIM,PixelType>::E y=x;
                y(i)-=length;
                for(;y(i)<=x(i)+length;y(i)++)
                {
                    int r = absolute(y(i)-x(i));
                    typename MatN<DIM,PixelType>::E z=y;
                    if(f.isValid(z))
                    {
                        m(r,1)+= (f(x)-mu)*(f(z)-mu)/(sigma*sigma) ;
                        mcount(r,1)++;
                    }

                }

            }
        }
        for(unsigned int j=0;j<m.sizeJ();j++){
            for(unsigned int i=0;i<m.sizeI();i++){
                if(j==0)
                    m(i,j)=i;
                else
                    m(i,j)/=(mcount(i,j));
            }
        }
        return m;
    }
    /*!
     * \param f input matrix
     * \return output matrix with a float as pixel/voxel type
     *
     *  calculated the 2-VecNd correlation function in any direction by FFT  P = FFT^(-1)(FFT(f)FFT(f)^*)
    */

    template<int DIM,typename PixelType>
    static MatN<DIM,F32> correlationDirectionByFFT(const MatN<DIM,PixelType> & f){

        MatN<DIM,PixelType> bint;
        bint = pop::Representation::truncateMulitple2(f);
        MatN<DIM,F32> binfloat(bint);
        typename MatN<DIM,PixelType>::IteratorEDomain it (binfloat.getIteratorEDomain());
        binfloat = pop::ProcessingAdvanced::greylevelRange(binfloat,it,0,1);


        MatN<DIM,ComplexF32>  bin_complex(bint.getDomain());
        Convertor::fromRealImaginary(binfloat,bin_complex);
        MatN<DIM,ComplexF32>  fft = pop::Representation::FFT(bin_complex,1);

        it.init();
        while(it.next()){
            ComplexF32 c = fft(it.x());
            ComplexF32 c1 = fft(it.x());
            fft(it.x()).real() = (c*c1.conjugate()).real();
            fft(it.x()).img() =0;
        }
        fft  = pop::Representation::FFT(fft,0);
        MatN<DIM,F32>  fout(bint.getDomain());
        Convertor::toRealImaginary(fft,fout);
        return  fout;

    }
    /*!
     * \param f input matrix
     * \param nbrchord  number of sampling
     * \return  Mat2F32 M
     *
     * P(i,0)=i and P(i,1) = Proba(|c|=i) with c a random chord, To estimate it, we random n chords\n
    */
    template<int DIM,typename PixelType>
    static Mat2F32 chord(const MatN<DIM,PixelType> & f, int nbrchord=20000000)
    {
        Mat2F32 m;
        std::vector<DistributionUniformInt> random_dist;
        for(int i=0;i<DIM;i++)
            random_dist.push_back(DistributionUniformInt(0,f.getDomain()(i)-1));
        DistributionUniformInt c (0,DIM-1);
        for(int index_test=0;index_test<nbrchord;index_test++)
        {
            int direction = c.randomVariable();
            typename MatN<DIM,PixelType>::E  x ;
            for(int i=0;i<DIM;i++)
                x(i)=random_dist[i].randomVariable();
            int phase = f(x);
            bool boundary =false;
            int dist =-1;
            int phasetemp = phase;
            typename MatN<DIM,PixelType>::E y1 =x;
            while(phase==phasetemp)
            {
                y1(direction)++;
                dist++;
                if(f.isValid(y1)==true)
                {
                    phasetemp = f(y1);
                }
                else
                {
                    boundary= true;
                    phasetemp=phase+1;
                }
            }
            typename MatN<DIM,PixelType>::E y2 =x;
            phasetemp = phase;
            while(phase==phasetemp)
            {
                y2(direction)--;
                dist++;
                if(f.isValid(y2)==true)
                {
                    phasetemp = f(y2);
                }
                else
                {
                    boundary= true;
                    phasetemp=phase+1;
                }
            }
            if(boundary==false)
            {
                if(dist>=static_cast<int>(m.sizeI()))
                    m.resizeInformation(dist+1,m.sizeJ());
                if(phase>=static_cast<int>(m.sizeJ())-1)
                    m.resizeInformation(m.sizeI(),phase+2);

                m(dist,phase+1)++;
            }
        }
        for(unsigned int j=0;j<m.sizeJ();j++)
        {
            if(j==0)
            {
                for(unsigned int i=0;i<m.sizeI();i++)
                    m(i,j)=i;
            }
            else
            {
                int count=0;
                for(unsigned int i=0;i<m.sizeI();i++)
                {
                    count+=m(i,j);
                }
                for(unsigned int i=0;i<m.sizeI();i++)
                {
                    m(i,j)/=count;
                }
            }
        }
        return m;
    }

    /*!
     * \param bin input binary matrix
     * \param norm norm of the ball
     * \param distance distance map
     * \return Mat2F32 M
     *
     *  ldistance allows the evaluation of the size distribution between any VecNd and the complementary space
     *  This algorithm works in any dimension and in any norms (in particular the euclidean norm for norm=2)\n
     *  M(i,0)=i, M(i,1) =P(d(x,X^c)) where X^c is the complementary set of the input binary set
    */

    template<int DIM>
    static Mat2F32 ldistance(const MatN<DIM,UI8> & bin,int norm,MatN<DIM,UI8> & distance){
        MatN<DIM,UI8> bin_minus(bin.getDomain());
        typename MatN<DIM,UI8>::IteratorEDomain it(bin_minus.getIteratorEDomain());
        while(it.next()){
            if(bin(it.x())!=0)
                bin_minus(it.x())=0;
            else
                bin_minus(it.x())=1;
        }
        if(norm!=2)
            distance = pop::ProcessingAdvanced::voronoiTesselation(bin_minus,  bin.getIteratorENeighborhood(1,norm)).second;
        else{
            distance =pop::ProcessingAdvanced::voronoiTesselationEuclidean(bin_minus).second;
        }
        it.init();
        Mat2F32 m;
        while(it.next()==true){
            if(static_cast<int>(distance(it.x()))>=m.sizeI()){
                m.resizeInformation(distance(it.x())+1,2);
            }
            m(distance(it.x()),1)++;
        }
        int sum=0;
        for(unsigned int i=0;i<m.sizeI();i++){
            sum+=m(i,1);
            m(i,0)=i;
        }
        for(unsigned int i=0;i<m.sizeI();i++){
            m(i,1)/=sum;
        }
        return m;
    }
    /*!
     * \param bin input binary matrix
     * \param norm norm of the ball
     * \param fgranulo granulometric map
     * \return granulometric Mat2F32 M
     *
     *  Granulometry allows the evaluation of the size distribution of grains in binary matrixs (<a href=http://en.wikipedia.org/wiki/Granulometry_%28morphology%29>wiki</a> ).
     *  This algorithm works in any dimension and in any norms (in particular the euclidean norm for norm=2)\n
     *  M(i,0)=i, M(i,1) =\f$ | (X\circ B(i,norm)|\f$, M(i,2)=M(i,1)-M(i-1,1), where \f$X\f$ is the binary set defined by the input binary matrix,
     *  \f$\circ\f$ the opening operator, \f$B(i,norm)=\{x:|x|_n\leq i\}\f$ the ball centered in 0 of radius i with the given norm and \f$|X|\f$ is the cardinality of the set X
     *
    */

    template<int DIM>
    static Mat2F32 granulometryMatheron(const MatN<DIM,UI8> & bin, F32 norm ,MatN<DIM,UI8> & fgranulo){
      FunctorF::FunctorThreshold<UI8,UI8, UI8> func(1,NumericLimits<UI8>::maximumRange());
        MatN<DIM,UI8> bin2(bin.getDomain());
        fgranulo.resize(bin.getDomain());

        forEachFunctorUnaryF(bin,bin2,func);
        int size = bin2.getDomain().multCoordinate();
        Mat2F32 marea = Analysis::area(bin2);
        int area = size - marea(0,1);
        Mat2F32 m(1,3);
        m(0,0) = 0;
        m(0,1) = area;
        m(0,2) = 0;
        int radius =1;
        typename MatN<DIM,UI8>::IteratorEDomain it_total(bin2.getIteratorEDomain());
        MatN<DIM,UI8> opening(bin2.getDomain());
        while(area!=0){
            std::cout<<"radius Matheron "<<radius<<" and cardinality "<<area<<std::endl;
            it_total.init();
            opening =pop::ProcessingAdvanced::openingRegionGrowing(bin2,radius,norm);
            it_total.init();
            area = 0;
            while(it_total.next()){
                if(opening(it_total.x())!=0){
                    fgranulo(it_total.x())=radius;
                    area++;
                }
            }
            m.resizeInformation(m.sizeI()+1,3);
            m(m.sizeI()-1,0) = m.sizeI()-1;
            m(m.sizeI()-1,1) = area;
            if(m.sizeI()>1&&m(m.sizeI()-1,1)>m(m.sizeI()-2,1))
                m(m.sizeI()-1,1) =  m(m.sizeI()-2,1);
            m(m.sizeI()-1,2) =  m(m.sizeI()-2,1)-m(m.sizeI()-1,1);
         radius++;

        }

        return m;
    }
    /*!
     *
     * \param bin input binary matrix
     * \param norm norm of the ball (1=4-connectivity, 0=8-connectivity in 2D)
     * \return Mat2F32 containing the geometrical tortuosity following each coordinate
     *
     *  Calculated the geometrical tortuosity (geodesical path) following each coordinate
     *
     */

    template<int DIM>
    static Mat2F32 geometricalTortuosity( const MatN<DIM,UI8> & bin, int norm=1)
    {
        typename MatN<DIM,UI8>::IteratorEDomain it(bin.getIteratorEDomain());
        Mat2F32 m(DIM,2);
        for(int i=0;i<DIM;i++){
            int count=0;
            int sum=0;
            for(int j=0;j<=1;j++){
                MatN<DIM,UI8> seed(bin.getDomain());
                it.init();
                while(it.next()){
                    if(j==0&&it.x()(i)==0&&bin(it.x())!=0)
                        seed(it.x())=1;
                    if(j==1&&it.x()(i)==bin.getDomain()(i)-1&&bin(it.x())!=0)
                        seed(it.x())=1;
                }
                MatN<DIM,UI16> dist = pop::ProcessingAdvanced::voronoiTesselation(seed,bin,bin.getIteratorENeighborhood(1,norm)).second;
                it.init();
                while(it.next()){
                    if(j==0&&it.x()(i)==bin.getDomain()(i)-1&&dist(it.x())!=0){
                        count++;
                        sum+=dist(it.x())+1;

                    }
                    if(j==1&&it.x()(i)==0                    &&dist(it.x())!=0){
                        count++;
                        sum+=dist(it.x())+1;
                    }

                }
            }
            m(i,0)=i;
            m(i,1)=sum*1.0/(count*(bin.getDomain()(i)-1));
        }
        return m;
    }
    /*!
     * \param bin input binary matrix
     * \param norm norm of the ball (1=4-connectivity, 0=8-connectivity in 2D)
     * \return median medial-axis
     *
     * \f$ m= \cup_{i=0}^\infty (X\ominus i\lambda)\setminus((X\ominus i\lambda )\circ\lambda    )\f$  where \f$X\f$ is the binary set defined by the input binary matrix,  \f$\lambda\f$  is the structural element
     * , \f$\ominus\f$ the erosion operator, \f$i\lambda\f$ the scale of lambda by i, \f$\circ\f$ the opening operator.

     \code
    Mat2UI8 img;
    img.load("../image/outil.bmp");
    img =Analysis::medialAxis(img);
    img.display();
    \endcode
    */

    template<int DIM>
    static MatN<DIM,UI8> medialAxis(const MatN<DIM,UI8> & bin,int norm=1){
      MatN<DIM,UI8> bin_minus(bin.getDomain());
        typename MatN<DIM,UI8>::IteratorEDomain it(bin_minus.getIteratorEDomain());
        while(it.next()){
            if(bin(it.x())!=0)
                bin_minus(it.x())=0;
            else
                bin_minus(it.x())=1;
        }
        MatN<DIM,UI8> distance(bin.getDomain());
        if(norm!=2)
            distance = pop::ProcessingAdvanced::voronoiTesselation(bin_minus,  bin.getIteratorENeighborhood(1,norm)).second;
        else{
            distance =pop::ProcessingAdvanced::voronoiTesselationEuclidean(bin_minus).second;
        }

        MatN<DIM,UI8> medial(bin.getDomain());
        MatN<DIM,UI8> zero(bin.getDomain());
        MatN<DIM,UI8> erosion(bin.getDomain());
        MatN<DIM,UI8> opening(bin.getDomain());


        typename MatN<DIM,UI8>::IteratorENeighborhood itn (bin_minus.getIteratorENeighborhood(1,norm));
        bool isempty=false;
        int radius=0;
        while(isempty==false)
        {
           it.init();
            FunctorF::FunctorThreshold<UI8,UI8,UI8> func(radius+1,NumericLimits<UI8>::maximumRange());
            forEachFunctorUnaryF(distance,erosion,func);

            if(zero==erosion)
                isempty = true;
            it.init();
            opening = pop::ProcessingAdvanced::opening(erosion,it,itn);
            erosion -=opening;
            medial= maximum(medial,erosion);
            radius++;
        }
      return medial;
    }


    //@}

    //-------------------------------------
    //
    //! \name Topology
    //@{
    //-------------------------------------

    /*!
     * \brief percolation (1=yes, 0=no) in the given coordinate
     * \param bin input binary matrix
     * \param norm norm of the ball
     * \return Mat2F32 M
     *
     *  A(i,0)=i and A(i,1) = r if it exists  path included in the binary set eroded touching the two opposites faces for the i-coordinate, 0 otherwise\n
     * Numerically, we compute the max cluster of the binary test and we test if it touch the opposite direction for each coordinate
     * \sa pop::Processing::clusterMax(const FunctionBinary & bin, int norm)
    */

    template<int DIM>
    static Mat2F32 percolation(const MatN<DIM,UI8> & bin,int norm=1){
        MatN<DIM,UI8> max_cluster = pop::ProcessingAdvanced::clusterMax( bin,   bin.getIteratorENeighborhood(1,norm));
        Mat2F32 m(DIM,2);
        for(int i = 0;i<DIM;i++)
            m(i,0)=i;
        for(int i =0;i<DIM;i++)
        {
            bool percolleft=false;
            bool percolright=false;
            typename MatN<DIM-1,UI8>::IteratorEDomain b(bin.getDomain().removeCoordinate(i));
            typename MatN<DIM,UI8>::E x;
            while(b.next()==true)
            {
                for(int j =0;j<DIM;j++){
                    if(j<i)
                        x(j)=b.x()(j);
                    else if(j>i)
                        x(j)=b.x()(j-1);
                }
                x(i)=0;
                if(max_cluster(x)!=0)
                    percolleft=true;
                x(i)=bin.getDomain()(i)-1;
                if(max_cluster(x)!=0)
                    percolright=true;
            }
            if(percolleft==true && percolright==true)
                m(i,1)=1;
            else
                m(i,1)=0;
        }
        return m;
    }

    /*!
     * \brief Percolation until the erosion with a ball of radius r (diameter 2*r+1) in the given coordinate, -1 means no percolation for the original binary matrix
     * \param bin input binary matrix
     * \param norm norm of the ball
     * \return Mat2F32 M
     *
     *  A(i,0)=i and A(i,1) = R for the maximum erosion of the binary set with tha ball of radius R such that it exists  path included
     * in the eroded binary set touching the two opposites faces for the i-coordinate
     * Numerically, by increasing the radius of ball, we compute the erosion of the binary set until no percolation
     * \sa percolation(const MatN<DIM,PixelType> & bin,int norm,MatN<DIM,PixelType> & max_cluster) pop::Processing::clusterMax(const FunctionBinary & bin, int norm)
    */
    template<int DIM>
    static Mat2F32 percolationErosion(const  MatN<DIM,UI8>   & bin,int norm=1){
       MatN<DIM,UI8> bin_minus(bin.getDomain());
        typename MatN<DIM,UI8>::IteratorEDomain it(bin.getIteratorEDomain());
        while(it.next()){
            if(bin(it.x())!=0)
                bin_minus(it.x())=0;
            else
                bin_minus(it.x())=255;
        }
        MatN<DIM,UI8> erosion(bin.getDomain());
        MatN<DIM,UI16> dist(erosion.getDomain());
        if(norm!=2)
            dist = pop::ProcessingAdvanced::voronoiTesselation(bin_minus, bin_minus.getIteratorENeighborhood(1,norm)).second;
        else
            dist = pop::ProcessingAdvanced::voronoiTesselationEuclidean(bin_minus).second;
        Mat2F32 m(DIM,2);
        for(int i = 0;i<DIM;i++){
            m(i,0)=i;
            m(i,1)=-1;
        }
        bool ispercol=true;
        int radius=0;
        while(ispercol==true){
            it.init();
            erosion = pop::ProcessingAdvanced::threshold(dist,radius+1,NumericLimits<UI16>::maximumRange(),it);
            Mat2F32 mradius =  Analysis::percolation(erosion, norm);
            ispercol=false;
            for(int i = 0;i<DIM;i++){
                if(mradius(i,1)==1){
                    m(i,1)=radius+1;
                    ispercol=true;
                }
            }
            radius++;

        }
       return m;
    }

    /*!
     * \brief Percolation until the opening with a ball of radius r (diameter 2*r+1) in the given coordinate, -1 means no percolation for the original binary matrix
     * \param bin input binary matrix
     * \param norm norm of the ball
     * \return Mat2F32 M
     *
     *  A(i,0)=i and A(i,1) = R for the maximum opening of the binary set with tha ball of radius R such that it exists  path included
     * in the opened binary set touching the two opposites faces for the i-coordinate
     * Numerically, by increasing the radius of ball, we compute the opening of the binary set until no percolation
     * \sa clusterMax
    */
    template<int DIM>
    static Mat2F32 percolationOpening(const MatN<DIM,UI8> & bin,int norm=1){
        MatN<DIM,UI8> bin_minus(bin.getDomain());
        typename MatN<DIM,UI8>::IteratorEDomain it(bin.getIteratorEDomain());
        while(it.next()){
            if(bin(it.x())!=0)
                bin_minus(it.x())=0;
            else
                bin_minus(it.x())=255;
        }
        MatN<DIM,UI8> opening(bin.getDomain());

        MatN<DIM,UI16> dist(opening.getDomain());
        if(norm!=2)
            dist = pop::ProcessingAdvanced::voronoiTesselation(bin_minus, bin_minus.getIteratorENeighborhood(1,norm)).second;
        else
            dist = pop::ProcessingAdvanced::voronoiTesselationEuclidean(bin_minus).second;

        Mat2F32 m(DIM,2);
        for(int i = 0;i<DIM;i++){
            m(i,0)=i;
            m(i,1)=-1;
        }
        bool ispercol=true;
        int radius=0;
        while(ispercol==true){
            it.init();
            opening = pop::ProcessingAdvanced::threshold(dist,radius+1,NumericLimits<UI16>::maximumRange(),it);
            opening = pop::ProcessingAdvanced::dilationRegionGrowing(opening,radius,norm);
            Mat2F32 mradius =  Analysis::percolation(opening, norm);
            ispercol=false;
            for(int i = 0;i<DIM;i++){
                if(mradius(i,1)==1){
                    m(i,1)=radius+1;
                    ispercol=true;
                }
            }
            radius++;
         }
         return m;
    }
    /*!
     * \param bin input binary matrix
     * \return euler-poincar number
     *
     *  compute the euler-poincare number
     * \code
     * Mat2UI8 img;
     * img.load("../image/outil.bmp");
     * img = pop::Processing::threshold(img,120);
     * F32 euler = Analysis::eulerPoincare(img);
     * std::cout<<euler<<std::endl;
     * \endcode
    */
    template<int DIM>
    static F32 eulerPoincare(const MatN<DIM,UI8> & bin ){
            return AnalysisAdvanced::eulerPoincare(bin);
    }
    /*!
     * \param bin input binary matrix
     * \param file_topo24  only for the 3d case, the lock-up table is saved in a file named  topo24.dat "Your_Directory/Population/file/topo24.dat"
     * \return topological skeleton
     *
     *  compute the thining at constant topology thanks to a lock-up table named topo24.dat
     * \code
     * Mat2UI8 img;
     * img.load("../image/outil.bmp");
     * img = pop::Processing::threshold(img,120);
     * Mat2UI8 skeleton= Analysis::thinningAtConstantTopology(img,"../file/topo24.dat");
     * skeleton.display();
     * \endcode
    */
    template<int DIM>
    static MatN<DIM,UI8>  thinningAtConstantTopology( const MatN<DIM,UI8> & bin,std::string file_topo24="")
    {
        return AnalysisAdvanced::thinningAtConstantTopology(bin,file_topo24);
    }

    template<int DIM>
    static MatN<DIM,UI8>  thinningAtConstantTopologyWire( const MatN<DIM,UI8> & bin,F32 ratio_filter=0.5,int length_edge=3)
    {
        MatN<DIM,UI8>  img(bin);
        img = img.opposite();
        MatN<DIM,UI8>  dist = pop::ProcessingAdvanced::voronoiTesselation(img,img.getIteratorENeighborhood(1,0)).second;;
        img = img.opposite();
        MatN<DIM,UI8>  granulo;
        MatN<DIM,F32> m = Analysis::granulometryMatheron(img,0,granulo);
        m = m.deleteCol(1);
        DistributionRegularStep d(m);
        F32 mean = Statistics::moment(d,1,d.getXmin(),d.getXmax(),1);
        typename MatN<DIM,UI8>::IteratorEDomain it(dist.getIteratorEDomain());
        MatN<DIM,UI8> fixed_VecNd(img.getDomain());
        while(it.next()){
            if( img(it.x())>0 && dist(it.x())>mean*ratio_filter && dist(it.x())>=granulo(it.x()) )
                fixed_VecNd(it.x())=255;
            else
                fixed_VecNd(it.x())=0;
        }
        MatN<DIM,UI8> skeleton = AnalysisAdvanced::thinningAtConstantTopologyWire(img,fixed_VecNd);

        std::pair<MatN<DIM,UI8>,MatN<DIM,UI8> > p_vertex_edge = Analysis::fromSkeletonToVertexAndEdge(skeleton);
        MatN<DIM,UI32> label_edge = ProcessingAdvanced::clusterToLabel(p_vertex_edge.second,img.getIteratorENeighborhood(1,0),img.getIteratorEDomain());

        MatN<DIM,UI32> label_vertex = ProcessingAdvanced::clusterToLabel(p_vertex_edge.first,img.getIteratorENeighborhood(1,0),img.getIteratorEDomain());
        typename MatN<DIM,UI32>::IteratorEDomain itg(label_edge.getIteratorEDomain());
        typename MatN<DIM,UI32>::IteratorENeighborhood itn(label_edge.getIteratorENeighborhood(1,0));
        VecI32 v_length;
        std::vector<int> v_neight_edge;
        while(itg.next()){
            if(label_edge(itg.x())!=0){
                if((int)label_edge(itg.x())>=(int)v_length.size()){
                    v_length.resize(label_edge(itg.x())+1);
                    v_neight_edge.resize(label_edge(itg.x())+1,-1);
                }
                v_length(label_edge(itg.x())) ++;

                itn.init(itg.x());
                while(itn.next()){
                    if(label_vertex(itn.x())!=0&& (v_neight_edge[label_edge(itg.x())]!=static_cast<int>(label_vertex(itn.x())))){
                        if(v_neight_edge[label_edge(itg.x())]==-1)
                            v_neight_edge[label_edge(itg.x())] = label_vertex(itn.x());
                        else
                            v_neight_edge[label_edge(itg.x())] =0;
                    }
                }
            }
        }
        pop::Private::Topology<DIM> topo(POP_PROJECT_SOURCE_DIR+std::string("/file/topo24.dat"));
        itg.init();
        while(itg.next()){
            if(label_edge(itg.x())!=0&&v_neight_edge[label_edge(itg.x())]>0&&v_length(label_edge(itg.x()))<length_edge){
                skeleton(itg.x())=0;
            }

        }
        std::vector<pop::VecN<DIM,I32> > v_VecNd;
        itg.init();
        while(itg.next()){
            if(p_vertex_edge.first(itg.x())!=0)
                if(topo.isIrrecductible(skeleton,itg.x())==false)
                    v_VecNd.push_back(itg.x());
        }
        for(unsigned int i=0;i<v_VecNd.size();i++){
            if(topo.isIrrecductible(skeleton,v_VecNd[i])==false)
                skeleton(v_VecNd[i])=0;
        }
        return skeleton;
    }
    /*!
     * \param skeleton skeleton
     * \return the first element contain the matrix with the verteces and the second one with the edges
     * extract the vecteces and the edges from the topological skeleton
     *
     *  extract the vecteces and the edges from the topological skeleton
     \sa thinningAtConstantTopology( const MatN<DIM,PixelType> & bin,const char * file_topo24)
    */
    template<int DIM>
    static std::pair<MatN<DIM,UI8>,MatN<DIM,UI8> > fromSkeletonToVertexAndEdge(const MatN<DIM,UI8> & skeleton)
    {
        MatN<DIM,UI8> edge(skeleton.getDomain());
        MatN<DIM,UI8> vertex(skeleton.getDomain());

        typename MatN<DIM,UI8>::IteratorEDomain b(skeleton.getDomain());
        typename MatN<DIM,UI8>:: IteratorENeighborhood V(skeleton.getIteratorENeighborhood(1,0));

        b.init();
        while(b.next()==true )
        {
            (edge)(b.x())=0;
            (vertex)(b.x())=0;
            if(skeleton(b.x())!=0)
            {
                V.init(b.x());
                int neighbor=-1;

                while(V.next()==true ){
                    if(skeleton(V.x())!=0)neighbor++;
                }
                if(neighbor<=2) edge(b.x())=NumericLimits<UI8>::maximumRange();
                else vertex(b.x())=NumericLimits<UI8>::maximumRange();
            }
        }
        return std::make_pair(vertex,edge);
    }

    /*! GraphAdjencyList<Vec3I32> linkEdgeVertex(  const MatN<DIM,PixelType> & vertex,const MatN<DIM,PixelType> & edge,int & tore)
     * \param vertex labelled verteces
     * \param edge labelled edge
     * \param tore number of tores
     * \return the topological graph
     *
     *  After the extraction of  the vecteces and the edges from the topological skeleton with the VertexAndEdgeFromSkeleton algorithm
     * you affect a label at each vectex/edge cluster with cluster2Label, then you apply this procedure to get the topological graph
     * and because a tore edge is not connected to any verteces, you return the number of tore edges
     *
     \sa VertexAndEdgeFromSkeleton Cluster2Label GraphAdjencyList
    */
    template<int DIM,typename PixelType>
    static GraphAdjencyList<Vec3I32> linkEdgeVertex(  const MatN<DIM,PixelType> & vertex,const MatN<DIM,PixelType> & edge,int & tore)
    {

        typename MatN<DIM,PixelType>::IteratorEDomain b(vertex.getIteratorEDomain());
        typename MatN<DIM,PixelType>::IteratorENeighborhood V(edge.getIteratorENeighborhood(1,0));
        GraphAdjencyList<Vec3I32> g;

        std::vector<std::pair<int,int> > v_link;
        b.init();
        while(b.next()){
            if((edge)(b.x())>0){
                while((int)(edge)(b.x())>(int)v_link.size()){
                    v_link.push_back(std::make_pair(-1,-1));
                }
                //edge_length[(*edge)[b.x()]]++;
                V.init(b.x());
                while(V.next()==true)
                {
                    if(vertex(V.x())>0)
                    {

                        while((int)vertex(V.x())>(int)g.sizeVertex())
                            g.addVertex();
                        int edgeindex = ((edge)(b.x()) -1);
                        int vertexindex = (vertex)(V.x()) -1;
                        Vec3I32 vv;
                        vv= V.x();
                        //                    std::cout<<vv<<std::endl;
                        g.vertex(vertexindex) = vv;
                        int link = v_link[edgeindex].first;
                        //                    std::cout<<"hit vertex "<<(vertex)[V.x()]<<"by edge "<< (edge)[b.x()] <<std::endl;
                        //                    getchar();
                        if( link==-1 )
                            v_link[edgeindex].first= vertexindex;
                        else {
                            v_link[edgeindex].second= vertexindex;
                        }
                        //                    std::cout<<"link "<<v_link[edgeindex].first <<" and "<<v_link[edgeindex].second  <<std::endl;
                    }
                }
            }
        }
        b.init();
        while(b.next()){
            if((vertex)(b.x())>0)
            {
                int vertexindex = (vertex)(b.x()) -1;
                Vec3I32 vv;
                vv= b.x();
                while((int)(vertex)(b.x())>(int)g.sizeVertex())
                    g.addVertex();
                g.vertex(vertexindex) = vv;
            }

        }
        tore=0;
        for(int i =0;i<(int)v_link.size();i++)
        {
            //      std::cout<<"i "<<v_link[i].first<<" et "<< v_link[i].second<<std::endl;
            if(v_link[i].first!=-1&& v_link[i].second!=-1){
                int label_edge = g.addEdge();
                g.connection(label_edge,v_link[i].first,  v_link[i].second);
            }
            else if(v_link[i].first==-1&& v_link[i].second==-1){
                tore++;
            }
            else
                std::cerr<<"Error LinkEdgeVertex "<<std::endl;

        }
        return g;

    }


    template<typename Vertex, typename Edge>
    static DistributionRegularStep coordinationNumberStatistics(GraphAdjencyList<Vertex,Edge>&g ){
        VecI32 v;
        for(int i=0;i<g.sizeEdge();i++)
        {
            std::pair<int,int> p  =g.getLink(i);
            int vv = maximum(p.first,p.second);
            if(vv>=(int)v.size()){
                v.resize(vv+1);
            }
            v(p.first)++;
            v(p.second)++;
        }
        return pop::Statistics::computedStaticticsFromIntegerRealizations(v);
    }

    //@}



    //-------------------------------------
    //
    //! \name Labelling statistics
    //@{
    //-------------------------------------

    /*!
     * \param label input label matrix
     * \return  Vec M
     *
     * V(i)=|{x:f(x)=i-1}| with area=M(i,0) where we count the areas of each label
    */
    template<int DIM,typename PixelType>
    static VecI32 areaByLabel(const MatN<DIM,PixelType> & label)
    {
        return AnalysisAdvanced::areaByLabel(label,label.getIteratorEDomain());
    }
    /*!
     * \param label input label matrix
     * \return  Mat2F32 M
     *
     * V(i)=Perimter({x:f(x)=i-1}) where we count the perimeters of each label
    */
    template<int DIM,typename PixelType>
    static VecI32 perimeterByLabel(const MatN<DIM,PixelType> & label)
    {
        typename MatN<DIM,PixelType>::IteratorEDomain itg(label.getIteratorEDomain());
        typename MatN<DIM,PixelType>::IteratorENeighborhood itn(label.getIteratorENeighborhood(1,1));
        VecI32 v;
        while(itg.next()){
            if(label(itg.x())!=0){
                if(label(itg.x())>(PixelType)v.size()){
                    v.resize(label(itg.x()));
                }
                itn.init(itg.x());
                while(itn.next()){
                    if(label(itg.x())!=label(itn.x())){
                        v(label(itg.x())-1) ++;
                    }
                }
            }
        }
        return v;

    }
    /*!
     * \param label input label matrix
     * \return  Mat2F32 M
     *
     * we count the perimemter between contact labels
    */
    template<int DIM,typename PixelType>
    static VecI32 perimeterContactBetweenLabel(const MatN<DIM,PixelType> & label)
    {
        typename MatN<DIM,PixelType>::IteratorEDomain itg(label.getIteratorEDomain());
        typename MatN<DIM,PixelType>::IteratorENeighborhood itn(label.getIteratorENeighborhood(1,1));
        VecI32 v;
        while(itg.next()){
            if(label(itg.x())!=0){
                if((int)label(itg.x())>(int)v.size()){
                    v.resize(label(itg.x()));
                }
                itn.init(itg.x());
                while(itn.next()){
                    if(label(itn.x())!=0 && label(itg.x())!=label(itn.x())){
                        v(label(itg.x())-1) ++;
                    }
                }
            }
        }
       return v;


    }

    /*!
     * \param label input label matrix
     * \param norm norm
     * \return  Mat2F32 M
     *
     *  We count the feret dimater for each label where for norm=1 , D= 1/n*sum_i diameter(i) and otherwise D= (mult_i diameter(i))^{1/n}
    */
    template<int DIM,typename PixelType>
    static VecF32 feretDiameterByLabel(const MatN<DIM,PixelType> & label, int norm=1)
    {
        typename MatN<DIM,PixelType>::IteratorEDomain itg(label.getIteratorEDomain());
        std::vector<typename MatN<DIM,PixelType>::E> v_xmin;
        std::vector<typename MatN<DIM,PixelType>::E> v_xmax;
        while(itg.next()){

            if(label(itg.x())!=0){
                if(    (int)label(itg.x())  >= (int) v_xmin.size() ){
                    v_xmax.resize(label(itg.x())+1);
                    v_xmin.resize(label(itg.x())+1,label.getDomain());
                }
                v_xmin[label(itg.x())]=minimum(v_xmin[label(itg.x())],itg.x());
                v_xmax[label(itg.x())]=maximum(v_xmax[label(itg.x())],itg.x());
            }

        }
        VecF32 v;
        for(int index=0; index<(int)v_xmin.size();index++){
            if( v_xmin[index]!=label.getDomain() ){
                if(norm==0){
                    int sum=0;
                    for(int i=0;i<DIM;i++){
                        sum+=v_xmax[index](i)-v_xmin[index](i);
                    }
                    v.push_back(sum*1.0/DIM);
                }else {
                    int mult=1;
                    for(int i=0;i<DIM;i++){
                        mult*=v_xmax[index](i)-v_xmin[index](i);
                    }
                    v.push_back(    std::pow(mult*1.0, 1.0/DIM));
                }
            }
        }
        return v;
    }




    /*!
     * \param label input label matrix
     * \param v_xmin vector of xmin positions
     * \param v_xmax vector of xmax positions
     * \return std::vector of binary matrix
     *
     *  Extract each label of the input matrix to form a std::vector of binary matrixs
    */

    template<int DIM,typename PixelType>
    static pop::Vec<MatN<DIM,UI8>  > labelToMatrices(const MatN<DIM,PixelType> & label,  pop::Vec<typename MatN<DIM,PixelType>::E> & v_xmin,pop::Vec<typename MatN<DIM,PixelType>::E>&  v_xmax)
    {
        typename MatN<DIM,PixelType>::IteratorEDomain it (label.getIteratorEDomain());
        return AnalysisAdvanced::labelToMatrices(label,  v_xmin, v_xmax, it);
    }
    //@}
};
}
#endif // ANALYSIS_H
