#ifndef ANALYSISADVANCED_H
#define ANALYSISADVANCED_H
#include<vector>
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

#include"data/population/PopulationData.h"
namespace pop
{
namespace Private{
inline bool save(const char * file ,const std::vector<bool> &dat );
inline bool load(const char * file,std::vector<bool> &dat);

template<int DIM>
class Topology
{
private:
    static bool _lock_up_table2d[256];
    static std::vector<bool> _lock_up_table3d;
    static F32  _euler_tab[];
public:

    Topology(std::string inlockup=""){
        if(DIM==3&&_lock_up_table3d.size()==0&&inlockup.size()!=0)
            load(inlockup.c_str(),_lock_up_table3d);//1<<26,
    }
    F32 eulerPoincare(const MatN<3,UI8> & img,const VecN<3,int>& x){
        VecN<3,int> xx;
        int value=0;
        for(int ii=0;ii<=1;ii++){
            for(int jj=0;jj<=1;jj++){
                for(int kk=0;kk<=1;kk++){
                    xx(0)=x(0)+ii-1;
                    xx(1)=x(1)+jj-1;
                    xx(2)=x(2)+kk-1;
                    if(img.isValid(xx)==true){
                        if(img(xx)!=0){
                            value += 1<<(ii+jj*2+kk*4);
                        }
                    }
                }
            }
        }
        return _euler_tab[value];
    }
    F32 eulerPoincare(const MatN<2,UI8> & img,const VecN<2,int>& x){
        VecN<2,int> xx;
        int value=0;
        for(int ii=0;ii<=1;ii++){
            for(int jj=0;jj<=1;jj++){
                xx(0)=x(0)+ii-1;
                xx(1)=x(1)+jj-1;
                if(img.isValid(xx)==true){
                    if(img(xx)!=0){
                        value += 1<<(ii+jj*2);
                    }
                }
            }
        }
        return _euler_tab[value];
    }

    bool isIrrecductible(const MatN<2,UI8> & img,const VecN<2,int>& x ){
        VecN<2,int> y;
         int power=1;
         int value=0;
        for(y(0)=-1+x(0);y(0)<=1+x(0);y(0)++){
            for(y(1)=-1+x(1);y(1)<=1+x(1);y(1)++){
                if(y(0)!=x(0)||y(1)!=x(1)){
                    if(img.isValid(y(0),y(1))==true){
                        if(img(y)!=0){
                            value+=power;
                        }
                    }
                    power=power<<1;
                }
            }
        }
        return _lock_up_table2d[value];
    }
    bool isIrrecductible(const MatN<3,UI8> & img,const VecN<3,int>& x ){
        VecN<3,int> y,z;
         int power=1;
         int value=0;
        y=3;
        for(y(2)=-1;y(2)<=1;y(2)++){
            for(y(1)=-1;y(1)<=1;y(1)++){
                for(y(0)=-1;y(0)<=1;y(0)++){
                    if(y(0)!=0||y(1)!=0||y(2)!=0){
                        z=x+y;
                        if(img.isValid(z)==true){
                            if(img(x+y)!=0){
                                value+=power;
                            }
                        }
                        power=power<<1;
                    }
                }
            }
        }
        return _lock_up_table3d[value];
    }
};
template<int DIM>
bool Topology<DIM>::_lock_up_table2d[]={1,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1};
template<int DIM>
F32 Topology<DIM>::_euler_tab[]={0,0.125,0.125,0,0.125,0,-0.25,-0.125,0.125,-0.25,0,-0.125,0,-0.125,-0.125,0,0.125,0,-0.25,-0.125,-0.25,-0.125,-0.125,-0.25,-0.75,-0.375,-0.375,-0.25,-0.375,-0.25,0,-0.125,0.125,-0.25,0,-0.125,-0.75,-0.375,-0.375,-0.25,-0.25,-0.125,-0.125,-0.25,-0.375,0,-0.25,-0.125,0,-0.125,-0.125,0,-0.375,-0.25,0,-0.125,-0.375,0,-0.25,-0.125,0,0.125,0.125,0,0.125,-0.25,-0.75,-0.375,0,-0.125,-0.375,-0.25,-0.25,-0.125,-0.375,0,-0.125,-0.25,-0.25,-0.125,0,-0.125,-0.375,-0.25,-0.125,0,0,-0.125,-0.375,0,0,0.125,-0.25,-0.125,0.125,0,-0.25,-0.125,-0.375,0,-0.375,0,0,0.125,-0.125,0.5,0,0.375,0,0.375,0.125,0.25,-0.125,-0.25,-0.25,-0.125,-0.25,-0.125,0.125,0,0,0.375,0.125,0.25,0.125,0.25,0.25,0.125,0.125,-0.75,-0.25,-0.375,-0.25,-0.375,-0.125,0,0,-0.375,-0.125,-0.25,-0.125,-0.25,-0.25,-0.125,-0.25,-0.375,-0.125,0,-0.125,0,0.5,0.375,-0.375,0,0,0.125,0,0.125,0.375,0.25,0,-0.375,-0.125,-0.25,-0.375,0,0,0.125,-0.125,0,0,-0.125,-0.25,0.125,-0.125,0,-0.125,-0.25,-0.25,-0.125,0,0.125,0.375,0.25,-0.25,0.125,-0.125,0,0.125,0.25,0.25,0.125,0,-0.375,-0.375,0,-0.125,-0.25,0,0.125,-0.125,0,-0.25,0.125,0,-0.125,-0.125,0,-0.125,-0.25,0,0.125,-0.25,-0.125,0.375,0.25,-0.25,0.125,0.125,0.25,-0.125,0,0.25,0.125,-0.125,0,-0.25,0.125,-0.25,0.125,0.125,0.25,-0.25,0.375,-0.125,0.25,-0.125,0.25,0,0.125,0,-0.125,-0.125,0,-0.125,0,0.25,0.125,-0.125,0.25,0,0.125,0,0.125,0.125,0};
template<int DIM>
std::vector<bool> Topology<DIM>::_lock_up_table3d;
}



struct AnalysisAdvanced
{
    template<typename Function,typename Iterator>
    static VecI32 areaByLabel(const Function & label, Iterator itg)
    {
        VecI32 v;
        while(itg.next()){
            if(label(itg.x())!=0){
                if((int)label(itg.x())>=(int)v.size()){
                    v.resize(label(itg.x())+1);
                }
                v(label(itg.x())) ++;
            }
        }
        return v;

    }
    template<int DIM,typename TypePixel,typename Iterator>
    static pop::Vec<MatN<DIM,UI8>  > labelToMatrices(const MatN<DIM,TypePixel> & label,  pop::Vec<typename MatN<DIM,TypePixel>::E> & v_xmin,pop::Vec<typename MatN<DIM,TypePixel>::E>&  v_xmax, Iterator & it)
    {

        v_xmin.clear();
        v_xmax.clear();
        while(it.next()){
            if(static_cast<unsigned int>(label(it.x()))>v_xmin.size()){
                typename MatN<DIM,TypePixel>::E xmin(+NumericLimits<int>::maximumRange());
                typename MatN<DIM,TypePixel>::E xmax(-NumericLimits<int>::maximumRange());
                v_xmin.resize(label(it.x()),xmin);
                v_xmax.resize(label(it.x()),xmax);
            }
            if(label(it.x())>0){
                v_xmin[label(it.x())-1]= minimum(it.x(),v_xmin[label(it.x())-1]);
                v_xmax[label(it.x())-1]= maximum(it.x(),v_xmax[label(it.x())-1]) ;
            }
        }
        pop::Vec<MatN<DIM,UI8> > v(static_cast<int>(v_xmin.size()));
        for(int i =0;i<static_cast<int>(v_xmin.size());i++){
            if(v_xmin[i][0]!=NumericLimits<int>::maximumRange()){
                typename MatN<DIM,TypePixel>::E size = (v_xmax[i]-v_xmin[i])+1;
                v[i] = MatN<DIM,UI8>(size);
            }
        }
        it.init();
        while(it.next()){
            if(label(it.x())>0){
                int index = label(it.x())-1;
                typename MatN<DIM,TypePixel>::E pos = it.x()-v_xmin[label(it.x())-1];
                v(index)(pos)=255;
            }
        }
        return v;

    }
    template<typename Function,typename Iterator>
    static typename Function::F maxValue(const Function & f,  Iterator & it)
    {
        FunctorF::FunctorAccumulatorMax<typename Function::F > func;
        return forEachFunctorAccumulator(f,func,it);
    }
    template<typename Function,typename Iterator>
    static typename Function::F minValue(const Function & f,  Iterator & it)
    {
        FunctorF::FunctorAccumulatorMin<typename Function::F > func;
        return forEachFunctorAccumulator(f,func,it);
    }

    template<typename Function1,typename Iterator>
    static typename FunctionTypeTraitsSubstituteF<typename Function1::F,F32>::Result meanValue(const Function1 & f, Iterator & it)
    {
        FunctorF::FunctorAccumulatorMean<typename Function1::F> func;
        return forEachFunctorAccumulator(f,func,it);
    }

    template<typename Function1,typename Iterator>
    static F32 standardDeviationValue(const Function1 & f, Iterator & it)
    {

        F32 mean = meanValue(f,it);
        it.init();
        FunctorF::FunctorAccumulatorVariance<typename Function1::F> func(mean);
        return std::sqrt(forEachFunctorAccumulator(f,func,it));
    }

    template<typename Function,typename IteratorGlobal>
    static Mat2F32 histogram(const Function & f, IteratorGlobal & itg)
    {
        Mat2F32 m;
        while(itg.next()){
            int value = static_cast<int>(normValue(f(itg.x())));
            if(value>=static_cast<int>(m.sizeI())){
                m.resizeInformation(value+1,2);
            }
            m(value,1) ++;

        }
        int count =0;
        for(unsigned int i =0;i<m.sizeI();i++){
            count +=static_cast<int>(m(i,1));
        }
        for(unsigned int i =0;i<m.sizeI();i++){
            m(i,1)/=count;
            m(i,0)=static_cast<F32>(i);
        }
        return m;
    }
    template<typename Function,typename IteratorGlobal>
    static Mat2F32 area(const Function & f, IteratorGlobal & itg)
    {
        Mat2F32 m(maxValue(f,itg)+1,2);
        itg.init();
        while(itg.next()){
            m(f(itg.x()),1) ++;

        }
        for(unsigned int i =0;i<m.sizeI();i++)
            m(i,0)=i;
        return m;
    }
    template<typename Function,typename IteratorGlobal, typename IteratorNeighborhood>
    static Mat2F32 perimeter(const Function & f, IteratorGlobal & itg, IteratorNeighborhood itn)
    {
        Mat2F32 m;
        while(itg.next()){
            if(f(itg.x())>=(typename Function::F)m.sizeI()){
                m.resizeInformation(f(itg.x())+1,2);
            }
            itn.init(itg.x());
            while(itn.next()){
                if(f(itg.x())!=f(itn.x())){
                    m(f(itg.x()),1) ++;
                }
            }
        }
        for(unsigned int i =0;i<m.sizeI();i++)
            m(i,0)=i;
        return m;
    }

    template<int DIM>
    static inline F32 eulerPoincare(const MatN<DIM,UI8> & img)
    {

        Private::Topology<DIM> topo;

        F32 e_p=0;
        typename MatN<DIM,UI8>::IteratorEDomain it(img.getDomain()+1);
        while(it.next()==true){
            e_p+=topo.eulerPoincare(img,it.x());
        }
        if(DIM==2)
            return e_p*2;
        else
            return e_p;
    }


    template<int DIM>
    static MatN<DIM,UI8> thinningAtConstantTopology(const MatN<DIM,UI8> & bin,std::string file_topo24)
    {
        MatN<DIM,UI8> thinning(bin);
        Private::Topology<DIM> topo(file_topo24);
        typename MatN<DIM,UI8>::E  x;
        int indice=0,sens;
        sens=1;
        typename MatN<DIM,UI8>::IteratorEOrder b2(thinning.getIteratorEOrder());
        int tour=0;
        bool nil=true,bsens=true;
        bool suite=true;
        //thinning until irrecductible skeleton
        while(nil==true){
            nil=false;
            bsens=true;
            tour++;
            //all ways and directions
            while(bsens==true){
                b2.setDirection(sens);
                b2.setLastLoop(indice);
                while(b2.next())
                {
                    x= b2.x();
                    if( (thinning) (x) ==0)
                        suite=true;
                    else{
                        if(suite==true){
                            suite=false;
                            if(topo.isIrrecductible(thinning,x)==true)
                                thinning(x)=255;
                            else
                            {
                                thinning(x)=0;
                                nil =true;

                            }
                        }
                    }
                }
                if(sens==1)
                    sens=-1;
                else{
                    sens=1;
                    if(indice<DIM-1){
                        indice++;
                    }
                    else{
                        indice=0;
                        bsens =false;
                    }
                }
            }
        }
        return thinning;
    }
    template<int DIM>
    static MatN<DIM,UI8> thinningAtConstantTopologyWire(const MatN<DIM,UI8> & bin,const MatN<DIM,UI8>& f_allow_branch,const char * file_topo24="")
    {
        MatN<DIM,UI8> thinning(bin);
        Private::Topology<DIM> topo(file_topo24);
        typename MatN<DIM,UI8>::E  x;
        int indice=0,sens;
        sens=1;
        typename MatN<DIM,UI8>::IteratorEOrder b2(thinning.getIteratorEOrder());
        typename MatN<DIM,UI8>::IteratorENeighborhood itneigh(thinning.getIteratorENeighborhood(1,1));
        int tour=0;
        bool nil=true,bsens=true;
        bool suite=true;
        //thinning until irrecductible skeleton
        while(nil==true){
            nil=false;
            bsens=true;
            //all ways and directions
            while(bsens==true){
                b2.setDirection(sens);
                b2.setLastLoop(indice);
                while(b2.next())
                {
                    x= b2.x();
                    if( (thinning) (x) ==0)
                        suite=true;
                    else{
                        if(suite==true){
                            suite=false;
                            int count=0;
                            if(f_allow_branch(x)!=0){
                                itneigh.init(x);
                                while(itneigh.next())
                                {
                                    if(thinning(itneigh.x())!=0)
                                        count++;
                                }
                            }else{
                                count=3;
                            }
                            if(count>2){

                                if(topo.isIrrecductible(thinning,x)==true)
                                    thinning(x)=255;
                                else
                                {
                                    thinning(x)=0;
                                    nil =true;

                                }
                            }
                        }
                    }
                }
                if(sens==1)
                    sens=-1;
                else{
                    sens=1;
                    if(indice<DIM-1){
                        indice++;
                    }
                    else{
                        tour++;
                        indice=0;
                        bsens =false;
                    }
                }
            }
        }
        return thinning;
    }


};

namespace Private{
bool save(const char * file ,const std::vector<bool> &dat ){
    std::fstream outFile (file, std::ios::out | std::ios::binary) ;
    char buffer ;
    long int i,j;
    if (outFile.is_open()){
        buffer=0;
        for( i = 0 ; i <(int)dat.size()/8  ; i++){
            for(j=7;j>=0;j--){
                if( dat[i*8+j]== false){
                    buffer=buffer<<1;
                }
                else{
                    buffer=(buffer<<1)+1;
                }
            }
            outFile.write((char*)&buffer, sizeof(char)) ;
        }
    }
    else{
        return false;
    }
    outFile.close() ;
    return true;
}
bool load(const char * file,std::vector<bool> &dat)
{
    dat.clear();
    std::fstream myFile (file, std::ios::in | std::ios::binary) ;
    if (myFile.is_open()){
        while (!myFile.eof( ))      //if not at end of file, continue reading numbers
        {
            char buffer ;
            myFile.read((char*)&buffer, sizeof(char)) ;
            for(int j=0;j<8;j++)
            {
                if( (buffer>>j)%2==false)dat.push_back(false);
                else dat.push_back(true);

            }
        }
    }
    else
        return false;
    myFile.close() ;
    return true;
}
//template<int DIM>
//bool testTopologyMatrix33(MatN<DIM,UI8> img){
//    MatN<DIM,UI8> label1 = Processing::clusterToLabel(img,0);
//    int nbrlabel1 = Analysis::maxValue(label1);
//    img = img.opposite();
//    MatN<DIM,UI8> label2 = Processing::clusterToLabel(img,1);
//    int nbrlabel2 = Analysis::maxValue(label2);
//    img = img.opposite();
//    VecN<DIM,int> x;
//    x=1;
//    img(x)=0;
//    MatN<DIM,UI8> label3 = Processing::clusterToLabel(img,0);
//    int nbrlabel3 = Analysis::maxValue(label3);
//    img = img.opposite();
//    MatN<DIM,UI8> label4 = Processing::clusterToLabel(img,1);
//    int nbrlabel4 = Analysis::maxValue(label4);

//    if(nbrlabel1==nbrlabel3&&nbrlabel2==nbrlabel4){
//        std::cout<<"false"<<std::endl;
//        return false;
//    }
//    else{
//        std::cout<<"true"<<std::endl;
//        return true;
//    }
//}

//static std::vector<bool> createLockUpTable2D(){
//    Mat2UI8 img(3,3);
//    std::vector<bool> topo2d;

//    for(unsigned int index=0;index<1<<8;index++)
//    {
//        int k=index;
//        img=0;
//        img(1,1)=255;
//        for(int i=0;i<3;i++){
//            for(int j=0;j<3;j++){
//                if(i!=1||j!=1){
//                    if(k%2==1)
//                        img(i,j)=255;
//                    k=k>>1;
//                }

//            }
//        }
//        std::cout<<img;
//        topo2d.push_back(testTopologyMatrix33(img));
//    }
//    return topo2d;
//}
//static std::vector<bool> createLockUpTable3D(){
//    Mat3UI8 img(3,3,3);
//    std::vector<bool> topo2d;

//    for(unsigned int index=0;index<1<<26;index++)
//    {
//        int k=index;
//        img=0;
//        img(1,1,1)=255;
//        for(int z=0;z<3;z++){
//        for(int j=0;j<3;j++){
//        for(int i=0;i<3;i++){
//                    if(i!=1||j!=1||z!=1){
//                        if(k%2==1)
//                            img(i,j,z)=255;
//                        k=k>>1;
//                    }
//                }
//            }
//        }
//        std::cout<<img;
//        topo2d.push_back(testTopologyMatrix33(img));
//    }
//    return topo2d;
//}
}
}

#endif // ANALYSISADVANCED_H
