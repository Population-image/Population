#include<algorithm>
#include"algorithm/Statistics.h"
#include"data/distribution/DistributionFromDataStructure.h"
#include"data/distribution/DistributionMultiVariateFromDataStructure.h"
#include"data/utility/BasicUtility.h"
#include"data/vec/VecN.h"
namespace pop{
class DistributionIteratorERegularInterval
{
private:
    F32 _xminima;
    F32 _xmaxima;
    F32 _step;
    int _nbrstep;
    int _index;
public:
    DistributionIteratorERegularInterval( F32 xmin, F32 xmax,F32 step);
    bool isValid();
    void init();
    bool next();
    int size();
    F32 x();
};
class DistributionMultiVariateIteratorE
{
private:


    int _nbrstep;
    int _index;
public:
    VecF32 _xminima;
    VecF32 _xmaxima;
    VecF32 _domain;
    F32 _step;

    DistributionMultiVariateIteratorE( const VecF32& xmin, const VecF32& xmax, F32 step);
    bool isValid();
    void init();
    bool next();
    VecF32 x();
    VecF32 xInteger();
};
DistributionIteratorERegularInterval::DistributionIteratorERegularInterval(F32 xmin, F32 xmax,F32 step){
    _xminima = xmin;
    if(xmax<xmin)
        xmax=xmin;
    _xmaxima = xmax;
    _step    = step;
    _nbrstep = std::floor((_xmaxima-_xminima-0.0000001)/_step);
    init();
}
int DistributionIteratorERegularInterval::size(){
    return _nbrstep+1;
}
bool DistributionIteratorERegularInterval::isValid(){
    if(_xmaxima>_xminima)
        return true;
    else
        return false;
}

void DistributionIteratorERegularInterval::init(){
    _index =-1;
}
bool DistributionIteratorERegularInterval::next(){
    _index++;

    if(_index<=_nbrstep)
        return true ;
    else
        return false ;

}

F32 DistributionIteratorERegularInterval::x(){
    return _xminima + _index*_step;
}

DistributionMultiVariateIteratorE::DistributionMultiVariateIteratorE(const VecF32& xmin, const VecF32& xmax,F32 step){
    _step = step;
    _xminima = xmin;
    _xmaxima = xmax;
    _step = step;
    int multdim=1;
    _domain.resize(_xminima.size());
    for(unsigned int i=0;i<_xminima.size();i++){
        _domain(i)=std::ceil((_xmaxima(i)-_xminima(i)+0.0001) /_step );
        multdim *=_domain(i);
    }

    _nbrstep = multdim;
    init();
}

bool DistributionMultiVariateIteratorE::isValid(){
    if(_xmaxima>_xminima)
        return true;
    else
        return false;

}

void DistributionMultiVariateIteratorE::init(){
    _index =-1;
}

bool DistributionMultiVariateIteratorE::next(){
    _index++;

    if(_index<_nbrstep)
        return true ;
    else
        return false ;
}

VecF32 DistributionMultiVariateIteratorE::x(){
    int indice = _index;
    VecF32 xx(_xminima.size());
    for(unsigned int i=0;i<_xminima.size();i++)
    {
        int temp=1;
        for(unsigned int j=0;j<_xminima.size()-(i+1);j++)
            temp*=_domain(i);
        xx(_xminima.size()-(i+1)) = (indice/temp);
        indice -= (xx(_xminima.size()-(i+1)) *temp);
        xx(_xminima.size()-(i+1)) =xx(_xminima.size()-(i+1)) *_step + _xminima(_xminima.size()-(i+1)) ;
    }
    return xx;
}
VecF32 DistributionMultiVariateIteratorE::xInteger(){
    int indice = _index;
    VecF32 xx(_xminima.size());
    for(unsigned int i=0;i<_xminima.size();i++)
    {
        int temp=1;
        for(unsigned int j=0;j<_xminima.size()-(i+1);j++)
            temp*=_domain(i);
        xx(_xminima.size()-(i+1)) = (indice/temp);
        indice -= (xx(_xminima.size()-(i+1)) *temp);
    }
    return xx;
}
DistributionRegularStep  Statistics::integral(const Distribution &f, F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax+step,step);

    F32 integral=0;
    Mat2F32 m(it.size(),2);
    int i =0;
    while(it.next()){
        integral+=f.operator ()(it.x())*step;
        m(i,0)=it.x();
        m(i,1)=integral;
        i++;
    }

    return DistributionRegularStep(m);
}

namespace Private{
const VecN<2,F32> d=VecN<2,F32>(1,1)/normValue(VecN<2,F32>(1,1));
}
void Statistics::__inverseVecN(F32 x,F32 y, F32 &x_minus_one, F32& y_minus_two){
    VecN<2,F32> vx;
    vx(0)=x;
    vx(1)=y;
    VecN<2,F32> p;
    p=  pop::productInner(pop::Private::d,vx)*pop::Private::d;
    VecN<2,F32> vx_symmetry_axis;
    vx_symmetry_axis = p*2-vx;
    x_minus_one = vx_symmetry_axis(0);
    y_minus_two = vx_symmetry_axis(1);
}

DistributionRegularStep Statistics::inverse(const Distribution &f, F32 xmin, F32 xmax,F32 step,F32 error_step){


    Mat2F32 m;
    F32 x = xmin;

    F32 xminus;
    F32 yminus;
    __inverseVecN(x,f(x),xminus,yminus);
    F32  xstep =step;
    while(x<xmax){
        m.resizeInformation(m.sizeI()+1,2);
        m(m.sizeI()-1,0)=xminus;
        m(m.sizeI()-1,1)=yminus;
        //dicotomie method to find the next x (can be based on taylor expansion first order)
        xminus +=step;
        F32 xminustemp;
        F32 xnextmin=x;
        F32 xbefore=x;
        F32 xnextmax=xstep+x;
        bool error = false;
        while(error==false){

            __inverseVecN(xnextmax,f(xnextmax),xminustemp,yminus);
            if(xminustemp<xminus){
                xnextmax=(xnextmax-x)*2+x;
            }else{
                error = true;
            }
        }

        x=(xnextmax-xnextmin)/2+xnextmin;

        __inverseVecN(x,f(x),xminustemp,yminus);
        int k=0;
        while(absolute((xminustemp-xminus)/xminus)>error_step&&k<100){
            k++;
            if(xminustemp>xminus)
                xnextmax=x;
            else
                xnextmin=x;
            x=(xnextmax-xnextmin)/2+xnextmin;
            __inverseVecN(x,f(x),xminustemp,yminus);
        }
        xstep = x - xbefore+0.0001;

    }
    return DistributionRegularStep(m);
}

F32 Statistics::moment(const Distribution &f, int moment,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F32 integral=0;
    F32 normalization=0;
    while(it.next()){

        integral +=f.operator ()(it.x())*step*std::pow(it.x(),moment);
        normalization+=f.operator ()(it.x())*step;
    }
    return integral/normalization;
}

F32 Statistics::norm(const Distribution &f, F32 norm,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F32 integral=0;
    while(it.next()){
        integral +=std::pow(absolute(f.operator ()(it.x())),norm)*step;
    }
    return std::pow(integral,1.f/norm);
}
F32 Statistics::productInner(const Distribution &f,const Distribution &g, F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);
    g.setStep(step);
    F32 sum=0;
    while(it.next()){
        sum +=f.operator ()(it.x())*step*g.operator ()(it.x());
    }
    return sum;
}


F32 Statistics::FminusOneOfYMonotonicallyIncreasingFunction(const Distribution &f, F32 y,F32 xmin, F32 xmax,F32 mindiff)
{

    if(const DistributionRegularStep* dist =dynamic_cast<const DistributionRegularStep*>(f.___getPointerImplementation()))
    {
        return dist->fMinusOneForMonotonicallyIncreasing(y);
    }
    F32 xmincurrent=xmin;
    F32 xmaxcurrent=xmax;
    F32 ytemp;
    do{
        F32 step =  (xmaxcurrent - xmincurrent)/2+xmincurrent ;
        ytemp = f.operator ()(step);
        if(ytemp<y)
            xmincurrent = step;
        else
            xmaxcurrent = step;
    }while(xmaxcurrent-xmincurrent>=mindiff);
    return ytemp;
}

DistributionRegularStep Statistics::derivate(const Distribution &f, F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);


    Mat2F32 m(it.size()-1,2);
    F32 x=NumericLimits<F32>::maximumRange();
    int i=0;
    while(it.next()){
        if(x!=NumericLimits<F32>::maximumRange()){
            m(i-1,0)=x;
            m(i-1,1)=(f.operator ()(it.x())-f.operator ()(x))/step;
        }
        x = it.x();
        i++;
    }
    return DistributionRegularStep(m);
}



F32 Statistics::argMin(const Distribution &f,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F32 arg=0;
    F32 min=NumericLimits<F32>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())<min)
        {
            min = f.operator ()(it.x());
            arg = it.x();
        }
    }
    return arg;
}

std::vector<F32> Statistics::argMaxLocal(const Distribution &f,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);
    std::vector<F32> v_argmax;
    it.next();
    F32 max_minus_1=f.operator ()(it.x());

    it.next();
    F32 arg_max=it.x();
    F32 max=f.operator ()(it.x());

    bool perhaps_max=false;
    F32 arg_perhaps_max=false;
    while(it.next()){
        F32 arg_max_plus_1=it.x();
        F32 max_plus_1=f.operator ()(it.x());
        if(perhaps_max==false&&max>max_minus_1&&max>max_plus_1){
            v_argmax.push_back(arg_max);
        }
        if(perhaps_max==false&&max==max_minus_1&&max==max_plus_1){
            perhaps_max=true;
            arg_perhaps_max = arg_max;
        }
        if(perhaps_max==true&&max>max_plus_1){
            perhaps_max=false;
            v_argmax.push_back((arg_max+arg_perhaps_max)/2);
        }
        if(perhaps_max==true&&max<max_plus_1){
            perhaps_max=false;
        }
        arg_max = arg_max_plus_1;
        max_minus_1 = max;
        max = max_plus_1;
    }
    return v_argmax;
}


//F32 Statistics::argMin(const Distribution &f,F32 xmin, F32 xmax,F32 step)
//{

//    DistributionIteratorERegularInterval it(xmin,xmax,step);

//    F32 arg=0;
//    F32 min=NumericLimits<F32>::maximumRange();
//    while(it.next()){
//        if(f.operator ()(it.x())<min)
//        {
//            min = f.operator ()(it.x());
//            arg = it.x();
//        }
//    }
//    return arg;
//}

Vec<F32> Statistics::argMinLocal(const Distribution &f,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);
    Vec<F32> v_argmin;
    it.next();
    F32 min_minus_1=f.operator ()(it.x());

    it.next();
    F32 arg_min=it.x();
    F32 min=f.operator ()(it.x());

    bool perhaps_min=false;
    F32 arg_perhaps_min=false;
    while(it.next()){
        F32 arg_min_plus_1=it.x();
        F32 min_plus_1=f.operator ()(it.x());
        if(perhaps_min==false&&min<min_minus_1&&min<min_plus_1){
            v_argmin.push_back(arg_min);
        }
        if(perhaps_min==false&&min==min_minus_1&&min==min_plus_1){
            perhaps_min=true;
            arg_perhaps_min = arg_min;
        }
        if(perhaps_min==true&&min<min_plus_1){
            perhaps_min=false;
            v_argmin.push_back((arg_min+arg_perhaps_min)/2);
        }
        if(perhaps_min==true&&min>min_plus_1){
            perhaps_min=false;
        }
        arg_min = arg_min_plus_1;
        min_minus_1 = min;
        min = min_plus_1;
    }
    return v_argmin;
}

F32 Statistics::argMax(const Distribution &f,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F32 arg=0;
    F32 max=-NumericLimits<F32>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())>max)
        {
            max = f.operator ()(it.x());
            arg = it.x();
        }
    }
    return arg;
}

F32 Statistics::minValue( const Distribution &f,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F32 min=NumericLimits<F32>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())<min){
            min = f.operator ()(it.x());
        }
    }
    return min;
}
F32 Statistics::maxValue( const Distribution &f,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F32 max=-NumericLimits<F32>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())>max){
            max = f.operator ()(it.x());
        }
    }
    return max;
}
DistributionRegularStep Statistics::toProbabilityDistribution( const Distribution &f,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F32 sum=0;

    while(it.next()){
        sum += absolute(f.operator ()(it.x()))*step;
    }
    it.init();
    Mat2F32 m(it.size(),2);
    int i =0;
    while(it.next()){
        m(i,0)=it.x();
        m(i,1)=absolute(f.operator ()(it.x()))/sum;
        //        std::cout<<absolute(f.operator ()(it.x()))/sum/step<<std::endl;
        i++;
    }
    return DistributionRegularStep(m);
}
DistributionRegularStep Statistics::toCumulativeProbabilityDistribution( const Distribution &f,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F32 sum=0;

    while(it.next()){
        sum += absolute(f.operator ()(it.x()));
    }
    it.init();
    Mat2F32 m(it.size(),2);
    int i =0;
    F32 sumtemp =0;
    while(it.next()){
        m(i,0)=it.x();
        sumtemp += absolute(f.operator ()(it.x()));
        m(i,1)=sumtemp/sum;
        i++;
    }
    return DistributionRegularStep(m);
}
DistributionRegularStep Statistics::toStepFunction(const Distribution &f,F32 xmin, F32 xmax,F32 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);
    it.init();
    Mat2F32 m(it.size(),2);
    int i =0;
    while(it.next()){
        m(i,0)=it.x();
        m(i,1)=f.operator ()(it.x());
        i++;
    }
    DistributionRegularStep d;
    d.fromMatrix(m);
    return d;
}
Mat2F32 Statistics::toMatrix(const Distribution &f,F32 xmin, F32 xmax,F32 step)
{
    DistributionIteratorERegularInterval it(xmin,xmax,step);

    it.init();
    Mat2F32 m(it.size(),2);
    int i =0;
    while(it.next()){
        m(i,0)=it.x();
        m(i,1)=f.operator ()(it.x());
        i++;
    }
    return m;
}



DistributionIntegerRegularStep Statistics::computedStaticticsFromIntegerRealizations( const VecI32 & v){
    VecI32 vv(v);
    std::sort (vv.begin(), vv.end());
    Mat2F32 m(vv(vv.size()-1)-vv(0)+1,2);

    for(unsigned int i=0;i<vv.size();i++){
        m(vv(i)-vv(0),1)++;
    }
    for(unsigned int i=0;i<m.sizeI();i++){
        m(i,0)=vv(0)+i;
        m(i,1)/=vv.size();
    }
    DistributionIntegerRegularStep dd(m);
    return dd;

}
DistributionRegularStep Statistics::computedStaticticsFromRealRealizationsWithWeight( const VecF32 & v,const VecF32 & weight,F32 step,F32 min,F32 max)
{
    POP_DbgAssertMessage(v.size()!=0,"Vecs should contain at least one element");
    POP_DbgAssertMessage(v.size()==weight.size(),"Two vectors should have the same size");
    if(min==NumericLimits<F32>::minimumRange()){
        min = *std::min_element(v.begin(),v.end());
    }
    if(max==NumericLimits<F32>::maximumRange()){
        max = *std::max_element(v.begin(),v.end());
    }
    int nbr_step = std::floor((max-min)/step);
    Mat2F32 m(nbr_step,2);
    for(unsigned int i=0;i<v.size();i++){
        int value = std::floor((v(i)-min)/step);
        if(value<static_cast<int>(m.sizeI())&&value>=0)
            m(value,1)+=weight(i);
    }
    F32 sum=0;
    F32 incr = min;
    for(unsigned int i=0;i<m.sizeI();i++){
        m(i,0)=incr;
        sum+=m(i,1);
        incr+=step;
    }
    for(unsigned int i=0;i<m.sizeI();i++){
        m(i,1)/=(sum*step);
    }

    return DistributionRegularStep(m);
}
DistributionRegularStep Statistics::computedStaticticsFromRealRealizations(  const VecF32 & v,F32 step,F32 min,F32 max)
{
    if(min==NumericLimits<F32>::minimumRange()){
        min = *std::min_element(v.begin(),v.end());
    }
    if(max==NumericLimits<F32>::maximumRange()){
        max = *std::max_element(v.begin(),v.end());
    }
    int nbr_step = std::floor((max-min)/step);
    Mat2F32 m(nbr_step,2);
    for(unsigned int i=0;i<v.size();i++){
        int value = std::floor((v(i)-min)/step);
        if(value<static_cast<int>(m.sizeI())&&value>=0)
            m(value,1)++;
    }
    int sum=0;
    F32 incr = min;
    for(unsigned int i=0;i<m.sizeI();i++){
        m(i,0)=incr;
        sum+=m(i,1);
        incr+=step;
    }
    for(unsigned int i=0;i<m.sizeI();i++){
        m(i,1)/=(sum*step);
    }

    return DistributionRegularStep(m);
}


F32  Statistics::maxRangeIntegral(const Distribution & f,F32 integralvalue,F32 min,F32 maxsupremum ,F32 step  ){
    DistributionIteratorERegularInterval it(min,maxsupremum,step);
    it.init();
    F32 sum=0;
    while(it.next()){
        sum+=f.operator ()(it.x())*step;
        if(sum>integralvalue)
            return it.x();
    }
    return maxsupremum;
}


F32 Statistics::moment(const DistributionMultiVariate &f, VecF32 moment,VecF32 xmin, VecF32 xmax,F32 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    F32 integral=0;
    F32 normalization=0;

    while(it.next()){
        F32 sum=1;
        for(unsigned int i =0;i<moment.size();i++)
            sum *= std::pow(it.x()(i),moment(i));
        integral +=f.operator ()(it.x())*sum;
        normalization+=f.operator ()(it.x());
    }
    return integral/normalization;
}


VecF32 Statistics::argMin(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    VecF32 arg;
    F32 min=NumericLimits<F32>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())<min)
        {
            min = f.operator ()(it.x());
            arg = it.x();
        }
    }
    return arg;
}
VecF32 Statistics::argMax(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    VecF32 arg;
    F32 max=-NumericLimits<F32>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())>max)
        {
            max = f.operator ()(it.x());
            arg = it.x();
        }
    }
    return arg;
}

F32 Statistics::minValue( const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    F32 min=NumericLimits<F32>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())<min){
            min = f.operator ()(it.x());
        }
    }
    return min;
}
F32 Statistics::maxValue( const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    F32 max=-NumericLimits<F32>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())>max){
            max = f.operator ()(it.x());
        }
    }
    return max;
}
DistributionMultiVariateRegularStep  Statistics::integral(const DistributionMultiVariate &f, VecF32 xmin, VecF32 xmax,F32 step)
{
    if(f.getNbrVariable()!=2)
        std::cerr<<"Only for two variate distribution";
    DistributionMultiVariateIteratorE it(xmin,xmax,step);
    Mat2F32 m( Vec2I32(it._domain(0),it._domain(1)));
    F32 steppower2 = step*step;
    while(it.next()){
        VecF32 x = it.xInteger();
        if(x(0)>0&&x(1)>0)
            m (x(0),x(1)) = f.operator ()(it.x())*steppower2+m (x(0)-1,x(1))+m (x(0),x(1)-1)-m (x(0)-1,x(1)-1);
        else if(x(0)>0)
            m (x(0),x(1)) = f.operator ()(it.x())*steppower2+m (x(0)-1,x(1));
        else if(x(1)>0)
            m (x(0),x(1)) = f.operator ()(it.x())*steppower2+m (x(0),x(1)-1);
        else
            m (x(0),x(1)) = f.operator ()(it.x())*steppower2;
    }
    DistributionMultiVariateRegularStep d(m,xmin,step);
    return d;
}
DistributionMultiVariateRegularStep Statistics::toProbabilityDistribution( const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    F32 sum=0;
    while(it.next()){
        sum += absolute(f.operator ()(it.x()));
    }
    it.init();
    Mat2F32 m( Vec2I32(it._domain(0),it._domain(1)));
    while(it.next()){
        VecF32 x = it.xInteger();
        m (x(0),x(1)) = absolute(f.operator ()(it.x()))/sum;
    }
    DistributionMultiVariateRegularStep d(m,xmin,step);
    return d;
}

Mat2F32 Statistics::toMatrix( const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step){
    DistributionMultiVariateIteratorE it(xmin,xmax,step);
    Mat2F32 m( Vec2I32(it._domain(0),it._domain(1)));
    while(it.next()){
        VecF32 x = it.xInteger();
        m (x(0),x(1)) = f.operator ()(it.x());
    }
    return m;

}

}
