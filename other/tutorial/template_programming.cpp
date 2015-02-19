#include"Population.h"//Single header
using namespace pop;//Population namespace


//erosion of 2d matrix with UI8 as pixel type and the structural emlement is a square
Mat2UI8 erosion_Level_0(Mat2UI8 m,double radius)
{
    Mat2UI8 m_erosion(m.getDomain());
    //scan the domain of the matrix
    for(unsigned int i = 0;i<m.sizeI();i++){
        for(unsigned int j = 0;j<m.sizeJ();j++){
            UI8 value = m(i,j);
            //scan the neighborhood
            for( int ki = -radius;ki<=radius;ki++){
                for( int kj = -radius;kj<=radius;kj++){
                    if(m.isValid(i+ki,j+kj)==true){
                        value = (std::min)(value, m(i+ki,j+kj));
                    }
                }
            }
            m_erosion(i,j)=value;
        }
    }
    return m_erosion;
}
//erosion of 2d matrix with UI8 as pixel type and the structural element is a ball
Mat2UI8 erosion_Level_1(Mat2UI8 m,double radius,int norm)
{
    Mat2UI8 m_erosion(m.getDomain());
    Mat2UI8::IteratorEDomain it_global = m.getIteratorEDomain();
    Mat2UI8::IteratorENeighborhood it_local = m.getIteratorENeighborhood(radius,norm);//
    //scan the image domain
    while(it_global.next()){//.next()->iterate over the domain
        UI8 value = m(it_global.x()); //.x()->access a 2d-vector (Vec2I32) of the domain as a pixel position
        //scan the neighborhood of a point
        it_local.init(it_global.x());//.init(Vec2I32)->init the neighbohood at the given 2d-vector
        while(it_local.next()){//.next()->iterate over the neighborhood
            value = (std::min)(value, m(it_local.x()));//.x()->access a 2d-vector (Vec2I32) of the neighbohood as a pixel position
        }
        m_erosion(it_global.x())=value;
    }
    return m_erosion;
}
//erosion of a Nd matrix with any pixel type and the structural element is a ball
template<int Dim, typename PixelType>
MatN<Dim,PixelType> erosion_Level_2(MatN<Dim,PixelType> m,double radius,int norm)
{
    MatN<Dim,PixelType> m_erosion(m.getDomain());
    typename MatN<Dim,PixelType>::IteratorEDomain it_global = m.getIteratorEDomain();
    typename MatN<Dim,PixelType>::IteratorENeighborhood it_local = m.getIteratorENeighborhood(radius,norm);
    //scan the image domain
    while(it_global.next()){
        PixelType value = m(it_global.x());
        //scan the neighborhood of a point
        it_local.init(it_global.x());
        while(it_local.next()){
            value = pop::minimum(value, m(it_local.x()));//use pop::minimum rather than std::min because of template specilatization
        }
        m_erosion(it_global.x())=value;
    }
    return m_erosion;
}
//erosion  of a Nd matrix with any pixel type with any gobal iterationE and local iterationE
template<int Dim, typename PixelType,typename IteratorEGlobal,typename IteratorELocal>
MatN<Dim,PixelType> erosion_Level_3(const MatN<Dim,PixelType> & m,IteratorEGlobal it_global,IteratorELocal it_local )
{
    MatN<Dim,PixelType> m_erosion(m);
    //Global scan
    while(it_global.next()){
        it_local.init(it_global.x());
        PixelType value = m(it_global.x());
        //Local scan
        while(it_local.next()){
            value = pop::minimum(value, m(it_local.x()));
        }
        m_erosion(it_global.x())=value;
    }
    return m_erosion;
}
//erosion of any functions  with any gobal iterationE and any local iterationE
template<typename Function,typename IteratorEGlobal,typename IteratorELocal>
Function erosion_Level_4(const Function& m,IteratorEGlobal it_global,IteratorELocal it_local )
{
    Function m_erosion(m.getDomain());
    //Global scan
    while(it_global.next()){
        it_local.init(it_global.x());
        typename Function::F value = m(it_global.x());
        //Local scan
        while(it_local.next()){
            value = pop::minimum(value, m(it_local.x()));
        }
        m_erosion(it_global.x())=value;
    }
    return m_erosion;
}
//Algorithm of local accumulation in global loop of any functions  with any gobal iterationE and any local iterationE
template<typename Function,typename FunctorAccumulator, typename IteratorEGlobal,typename IteratorELocal>
Function neighbordhood_Algorithm_Level_5(const Function& m,FunctorAccumulator func,IteratorEGlobal it_global,IteratorELocal it_local )
{
    Function m_neigh(m.getDomain());
    //Global scan
    while(it_global.next()){
        it_local.init(it_global.x());
        func.init();
        //Local scan
        while(it_local.next()){
            //accumulate values
            func(m(it_local.x()));
        }
        m_neigh(it_global.x())=func.getValue();
    }
    return m_neigh;
}
int main(){
    Mat2UI8 m;
    m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
    Mat2UI8 m_erosion = erosion_Level_0(m,3);
    m_erosion.display("Level 0",false);
    m_erosion = erosion_Level_1(m,3,2);
    m_erosion.display("Level 1",false);
    m_erosion = erosion_Level_2(m,3,2);
    m_erosion.display("Level 2",false);
    pop::Mat2UI8::IteratorERectangle it_global = m.getIteratorERectangle(m.getDomain()/4,m.getDomain()*3/4);
    pop::Mat2UI8::IteratorENeighborhoodAmoebas it_local = m.getIteratorENeighborhoodAmoebas(15,0.05);
    m_erosion = erosion_Level_3(m,it_global,it_local);
    m_erosion.display("Level 3",false);

    GraphAdjencyList<UI8> graph;//Adgency graph with a vertex type coded in 1 byte
    int v0 = graph.addVertex();
    int v1 = graph.addVertex();
    int v2 = graph.addVertex();
    int v3 = graph.addVertex();
    graph(v0)=0;
    graph(v1)=100;
    graph(v2)=175;
    graph(v3)=125;
    int e0 = graph.addEdge();
    int e1 = graph.addEdge();
    int e2 = graph.addEdge();
    graph.connection(e0,v0,v1);
    graph.connection(e1,v1,v2);
    graph.connection(e2,v2,v3);
    std::cout<<"graph"<<std::endl;
    std::cout<<(int)graph(v0)<<std::endl;
    std::cout<<(int)graph(v1)<<std::endl;
    std::cout<<(int)graph(v2)<<std::endl;
    std::cout<<(int)graph(v3)<<std::endl;
    GraphAdjencyList<UI8>::IteratorEDomain it_graph_global = graph.getIteratorEDomain();
    GraphAdjencyList<UI8>::IteratorENeighborhood it_graph_local = graph.getIteratorENeighborhood();
    graph = erosion_Level_4(graph,it_graph_global,it_graph_local);
    std::cout<<"graph erosion"<<std::endl;
    std::cout<<(int)graph(v0)<<std::endl;
    std::cout<<(int)graph(v1)<<std::endl;
    std::cout<<(int)graph(v2)<<std::endl;
    std::cout<<(int)graph(v3)<<std::endl;

    FunctorF::FunctorAccumulatorMax<UI8> func_max;
    Mat2UI8 m_dilation = neighbordhood_Algorithm_Level_5(m, func_max , m.getIteratorEDomain(), m.getIteratorENeighborhood(3,2));
    m_dilation.display("Level 5");
}
