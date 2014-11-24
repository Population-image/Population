#ifndef ANALYSISADVANCED_H
#define ANALYSISADVANCED_H
#include<vector>
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

#include"data/population/PopulationData.h"
namespace pop
{
namespace Private{
//old code with low c++ skill
class BooleanField
{
public:

    static void save(const char * file ,const std::vector<bool> &dat )throw(pexception)
    {
        std::fstream outFile (file, std::ios::out | std::ios::binary) ;
        char buffer ;
        long int i,j;

        if (outFile.is_open())
        {
            buffer=0;
            for( i = 0 ; i <(int)dat.size()/8  ; i++)
            {
                for(j=7;j>=0;j--)
                {
                    if( dat[i*8+j]== false)
                    {
                        buffer=buffer<<1;
                    }
                    else
                    {
                        buffer=(buffer<<1)+1;
                    }

                }
                outFile.write((char*)&buffer, sizeof(char)) ;
            }
        }
        else
        {
            throw(pexception("Cannot save file of the BooleanField"));

        }
        outFile.close() ;
    }
    static void load(const char * file,std::vector<bool> &dat)throw(pexception)
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
            throw(pexception(std::string("Cannot open file of the BooleanField")+file));
        myFile.close() ;
    }
};

template<typename Function>
class Topology
{
private:
    static bool _lock_up_table2d[256];
    static std::vector<bool> _lock_up_table3d;
    //Code is good but to avoid cross dependency between Analasys and Processing I comment it
    //    static std::vector<bool> createLockUpTable2D(){
    //        Mat2UI8 img(3,3);
    //        std::vector<bool> topo2d;

    //        for(unsigned int index=0;index<1<<8;index++)
    //        {
    //            int k=index;
    //            img=0;
    //            img(1,1)=255;
    //            for(int i=0;i<3;i++)
    //                for(int j=0;j<3;j++){
    //                    if(i!=1||j!=1){
    //                        if(k%2==1)
    //                            img(i,j)=255;
    //                        k=k>>1;
    //                    }

    //                }
    //            Mat2UI8 label1 = Processing::clusterToLabel(img,0);
    //            int nbrlabel1 = Analysis::maxValue(label1);
    //            img = img.opposite();
    //            Mat2UI8 label2 = Processing::clusterToLabel(img,1);
    //            int nbrlabel2 = Analysis::maxValue(label2);
    //            img = img.opposite();

    //            img(1,1)=0;
    //            Mat2UI8 label3 = Processing::clusterToLabel(img,0);
    //            int nbrlabel3 = Analysis::maxValue(label3);
    //            img = img.opposite();
    //            Mat2UI8 label4 = Processing::clusterToLabel(img,1);
    //            int nbrlabel4 = Analysis::maxValue(label4);
    //            if(nbrlabel1==nbrlabel3&&nbrlabel2==nbrlabel4)
    //                topo2d.push_back(false);
    //            else{
    //                topo2d.push_back(true);
    //            }
    //        }
    //        return topo2d;
    //    }

//    Code is good but to avoid cross dependency between Analasys and Processing I comment it
public:

    Topology(std::string inlockup="")
    {
        if(_lock_up_table3d.size()==0&&inlockup.size()!=0)
            BooleanField::load(inlockup.c_str(),_lock_up_table3d);//1<<26,
    }
    template<int DIM>
    static bool isIrrecductibleVecN(const Function & img,const typename Function::E& x,Loki::Int2Type<DIM> );


    static bool isIrrecductibleVecN(const Function & img,const typename Function::E& x,Loki::Int2Type<2> )
    {
        typename Function::E y;
        long int power=1;
        long int value=0;
        for(y(1)=-1+x(1);y(1)<=1+x(1);y(1)++)
        {
            for(y(0)=-1+x(0);y(0)<=1+x(0);y(0)++)
            {
                if(y(0)!=x(0)||y(1)!=x(1))
                {
                    if(img.isValid(y(0),y(1))==true){
                        if(img(y)!=0)
                        {
                            value+=power;
                        }
                    }
                    power=power<<1;

                }
            }

        }
        return _lock_up_table2d[value];
    }
    static bool isIrrecductibleVecN(const Function & img,const typename Function::E& x,Loki::Int2Type<3> )
    {

        typename Function::E y,z;
        long int power=1;
        long int value=0;

        y=3;
        for(y(2)=-1;y(2)<=1;y(2)++)
        {
            for(y(1)=-1;y(1)<=1;y(1)++)
            {
                for(y(0)=-1;y(0)<=1;y(0)++)
                {
                    if(y(0)!=0||y(1)!=0||y(2)!=0)
                    {
                        z=x+y;
                        if(img.isValid(z)==true)
                            if(img(x+y)!=0)
                            {
                                value+=power;
                            }
                        power=power<<1;
                    }

                }
            }

        }
        return _lock_up_table3d[value];

    }
};
template<typename Function>
//bool Topology<Function>::_lock_up_table2d[]={1,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1};
bool Topology<Function>::_lock_up_table2d[]={1,1,1,0,1,1,0,0,1,0,0,0,1,1,0,0,1,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1};
template<typename Function>
std::vector<bool> Topology<Function>::_lock_up_table3d;
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
        pop::Vec<MatN<DIM,UI8> > v(v_xmin.size());
        for(int i =0;i<(int)v_xmin.size();i++){
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
        return FunctionProcedureFunctorAccumulatorF(f,func,it);
    }
    template<typename Function,typename Iterator>
    static typename Function::F minValue(const Function & f,  Iterator & it)
    {
        FunctorF::FunctorAccumulatorMin<typename Function::F > func;
        return FunctionProcedureFunctorAccumulatorF(f,func,it);
    }

    template<typename Function1,typename Iterator>
    static typename FunctionTypeTraitsSubstituteF<typename Function1::F,F64>::Result meanValue(const Function1 & f, Iterator & it)throw(pexception)
    {
        FunctorF::FunctorAccumulatorMean<typename Function1::F> func;
        return FunctionProcedureFunctorAccumulatorF(f,func,it);
    }

    template<typename Function1,typename Iterator>
    static F64 standardDeviationValue(const Function1 & f, Iterator & it)throw(pexception)
    {

        F64 mean = meanValue(f,it);
        it.init();
        FunctorF::FunctorAccumulatorVariance<typename Function1::F> func(mean);
        return std::sqrt(FunctionProcedureFunctorAccumulatorF(f,func,it));
    }

    template<typename Function,typename IteratorGlobal>
    static Mat2F64 histogram(const Function & f, IteratorGlobal & itg)
    {
        CollectorExecutionInformationSingleton::getInstance()->startExecution("Histogram");
        Mat2F64 m;
        while(itg.next()){
            int value = normValue(f(itg.x()));
            if(value>=static_cast<int>(m.sizeI())){
                m.resizeInformation(value+1,2);
            }
            m(value,1) ++;

        }
        int count =0;
        for(unsigned int i =0;i<m.sizeI();i++){
            count +=m(i,1);
        }
        for(unsigned int i =0;i<m.sizeI();i++){
            m(i,1)/=count;
            m(i,0)=i;
        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("End Histogram");
        return m;
    }
    template<typename Function,typename IteratorGlobal>
    static Mat2F64 area(const Function & f, IteratorGlobal & itg)
    {
        CollectorExecutionInformationSingleton::getInstance()->startExecution("Area");
        Mat2F64 m(maxValue(f,itg)+1,2);
        itg.init();
        while(itg.next()){
            m(f(itg.x()),1) ++;

        }
        for(unsigned int i =0;i<m.sizeI();i++)
            m(i,0)=i;
        CollectorExecutionInformationSingleton::getInstance()->endExecution("Area");
        return m;
    }
    template<typename Function,typename IteratorGlobal, typename IteratorNeighborhood>
    static Mat2F64 perimeter(const Function & f, IteratorGlobal & itg, IteratorNeighborhood itn)
    {
        CollectorExecutionInformationSingleton::getInstance()->startExecution("perimeter");
        Mat2F64 m;
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
        CollectorExecutionInformationSingleton::getInstance()->endExecution("perimeter");
        return m;
    }

    template<typename Function>
    static F64 eulerPoincare3D(const Function & img,std::string file_eulertab)throw(pexception)
    {

        std::fstream filestr(file_eulertab.c_str());
        std::vector<F64> N3_masq;

        if (filestr.is_open()){
            while (!filestr.eof( )){
                double data;
                filestr >> data;
                N3_masq.push_back(data);
            }
        }
        else
            throw(pexception(std::string("In Analysis::eulerPoincare, we cannot opent the file: ")+file_eulertab+ "\n The good file is eulertab.dat in the directory file Your_Directory/Population/file"));
        filestr.close();
        Function dat(img.getDomain()+4);
        F64 e_p=0;

        typename Function::E x;

        typename Function::IteratorEDomain b(img.getIteratorEDomain());
        while(b.next()==true)
        {
            dat( (b.x()+2) )=img(b.x());
        }

        typename Function::IteratorEDomain bb(img.getDomain()+2);
        bb.init();
        while(bb.next()==true)
        {
            int value =0;
            for(int ii=0;ii<2;ii++)for(int jj=0;jj<2;jj++)for(int kk=0;kk<2;kk++)
            {
                x(0)=bb.x()(0)+1+ii;
                x(1)=bb.x()(1)+1+jj;
                x(2)=bb.x()(2)+1+kk;
                if(dat(x)!=0){
                    value += 1<<(ii+jj*2+kk*4);
                }
            }
            e_p+=N3_masq[value];
        }
        return e_p;
    }
    template<typename Function>
    static Function thinningAtConstantTopologyGrowingRegion(const Function & bin,const char * file_topo24)
    {
        Function thinning(bin);
        Private::Topology<Function> topo(file_topo24);
        FunctorZero f;
        Population<Function,FunctorZero> pop(bin.getDomain(),f,bin.getIteratorENeighborhood(1,1));
        typename Function::IteratorEDomain it(bin.getIteratorEDomain());
        while(it.next()){
            if(bin(it.x())==0)
                pop.setRegion(1,it.x());
        }
        it.init();
        while(it.next()){
            if(bin(it.x())==0)
                pop.growth(1,it.x());
        }
        while(pop.next()){
            if(topo.isIrrecductibleVecN(thinning,pop.x().second,Loki::Int2Type<Function::DIM>())==false){
                thinning(pop.x().second)=0;
                pop.growth(pop.x().first,pop.x().second);
            }else{
                pop.pop();
            }

        }
        it.init();
        while(it.next()){
            if(topo.isIrrecductibleVecN(thinning,it.x(),Loki::Int2Type<Function::DIM>())==false){
                thinning(it.x())=0;
            }
        }
        return thinning;
    }

    template<typename Function>
    static Function thinningAtConstantTopology(const Function & bin,std::string file_topo24)
    {
        CollectorExecutionInformationSingleton::getInstance()->startExecution("thinningAtConstantTopology3d",COLLECTOR_EXECUTION_INFO);
        Function thinning(bin);
        try{
            Private::Topology<Function> topo(file_topo24);
            typename Function::E  x;
            int indice=0,sens;
            sens=1;
            typename Function::IteratorEOrder b2(thinning.getIteratorEOrder());
            int tour=0;
            bool nil=true,bsens=true;
            bool suite=true;
            //thinning until irrecductible skeleton
            while(nil==true){
                nil=false;
                bsens=true;
                CollectorExecutionInformationSingleton::getInstance()->progression(tour,"Iteration number");
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
                                if(topo.isIrrecductibleVecN(thinning,x,Loki::Int2Type<Function::DIM>())==true)
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
                        if(indice<Function::DIM-1){
                            indice++;
                        }
                        else{
                            indice=0;
                            bsens =false;
                        }
                    }
                }
            }
        }catch(const pexception&  ){
            throw(pexception("In Analysis::thinningAtConstantTopology, we cannot open the file: " +std::string(file_topo24)+"\n The lock-up table for this algorithm is topo24.dat the the folder file" ));

        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("thinningAtConstantTopology3d");
        return thinning;
    }
    template<typename Function>
    static Function thinningAtConstantTopologyWire(const Function & bin,const Function& f_allow_branch,const char * file_topo24="")
    {
        CollectorExecutionInformationSingleton::getInstance()->startExecution("thinningAtConstantTopology3d",COLLECTOR_EXECUTION_INFO);
        Function thinning(bin);
        try{
            Private::Topology<Function> topo(file_topo24);
            typename Function::E  x;
            int indice=0,sens;
            sens=1;
            typename Function::IteratorEOrder b2(thinning.getIteratorEOrder());
            typename Function::IteratorENeighborhood itneigh(thinning.getIteratorENeighborhood(1,1));
            int tour=0;
            bool nil=true,bsens=true;
            bool suite=true;
            //thinning until irrecductible skeleton
            while(nil==true){
                nil=false;
                bsens=true;
                CollectorExecutionInformationSingleton::getInstance()->progression(tour,"Iteration number");
                //                maximum(thinning/2,f_allow_branch).display();
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

                                    if(topo.isIrrecductibleVecN(thinning,x,Loki::Int2Type<Function::DIM>())==true)
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
                        if(indice<Function::DIM-1){
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
        }catch(const pexception&  ){
            throw(pexception("In Analysis::thinningAtConstantTopology, we cannot open the file: " +std::string(file_topo24)+"\n The lock-up table for this algorithm is topo24.dat the the folder file" ));

        }
        CollectorExecutionInformationSingleton::getInstance()->endExecution("thinningAtConstantTopology3d");
        return thinning;
    }


};
}
#endif // ANALYSISADVANCED_H
