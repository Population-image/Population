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
    F64 _xminima;
    F64 _xmaxima;
    F64 _step;
    int _nbrstep;
    int _index;
public:
    DistributionIteratorERegularInterval( F64 xmin, F64 xmax,F64 step);
    bool isValid();
    void init();
    bool next();
    int size();
    F64 x();
};
class DistributionMultiVariateIteratorE
{
private:


    int _nbrstep;
    int _index;
public:
    VecF64 _xminima;
    VecF64 _xmaxima;
    VecF64 _domain;
    F64 _step;

    DistributionMultiVariateIteratorE( const VecF64& xmin, const VecF64& xmax, F64 step);
    bool isValid();
    void init();
    bool next();
    VecF64 x();
    VecF64 xInteger();
};
DistributionIteratorERegularInterval::DistributionIteratorERegularInterval(F64 xmin, F64 xmax,F64 step){
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

F64 DistributionIteratorERegularInterval::x(){
    return _xminima + _index*_step;
}

DistributionMultiVariateIteratorE::DistributionMultiVariateIteratorE(const VecF64& xmin, const VecF64& xmax,F64 step){
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

VecF64 DistributionMultiVariateIteratorE::x(){
    int indice = _index;
    VecF64 x(_xminima.size());
    for(unsigned int i=0;i<_xminima.size();i++)
    {
        int temp=1;
        for(unsigned int j=0;j<_xminima.size()-(i+1);j++)
            temp*=_domain(i);
        x(_xminima.size()-(i+1)) = (indice/temp);
        indice -= (x(_xminima.size()-(i+1)) *temp);
        x(_xminima.size()-(i+1)) =x(_xminima.size()-(i+1)) *_step + _xminima(_xminima.size()-(i+1)) ;
    }
    return x;
}
VecF64 DistributionMultiVariateIteratorE::xInteger(){
    int indice = _index;
    VecF64 x(_xminima.size());
    for(unsigned int i=0;i<_xminima.size();i++)
    {
        int temp=1;
        for(unsigned int j=0;j<_xminima.size()-(i+1);j++)
            temp*=_domain(i);
        x(_xminima.size()-(i+1)) = (indice/temp);
        indice -= (x(_xminima.size()-(i+1)) *temp);
    }
    return x;
}
DistributionRegularStep  Statistics::integral(const Distribution &f, F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax+step,step);

    F64 integral=0;
    Mat2F64 m(it.size(),2);
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
const VecN<2,double> d=VecN<2,double>(1,1)/normValue(VecN<2,double>(1,1));
}
void inverseVecN(double x,double y, double &x_minus_one, double& y_minus_two){
    VecN<2,double> vx;
    vx(0)=x;
    vx(1)=y;
    VecN<2,double> p;
    p=  productInner(pop::Private::d,vx)*pop::Private::d;
    VecN<2,double> vx_symmetry_axis;
    vx_symmetry_axis = p*2-vx;
    x_minus_one = vx_symmetry_axis(0);
    y_minus_two = vx_symmetry_axis(1);
}

DistributionRegularStep Statistics::inverse(const Distribution &f, F64 xmin, F64 xmax,F64 step,F64 error_step){


    Mat2F64 m;
    double x = xmin;

    double xminus;
    double yminus;
    inverseVecN(x,f(x),xminus,yminus);
    double  xstep =step;
    while(x<xmax){
        m.resizeInformation(m.sizeI()+1,2);
        m(m.sizeI()-1,0)=xminus;
        m(m.sizeI()-1,1)=yminus;
        //dicotomie method to find the next x (can be based on taylor expansion first order)
        xminus +=step;
        double xminustemp;
        double xnextmin=x;
        double xbefore=x;
        double xnextmax=xstep+x;
        bool error = false;
        while(error==false){

            inverseVecN(xnextmax,f(xnextmax),xminustemp,yminus);
            if(xminustemp<xminus){
                xnextmax=(xnextmax-x)*2+x;
            }else{
                error = true;
            }
        }

        x=(xnextmax-xnextmin)/2+xnextmin;

        inverseVecN(x,f(x),xminustemp,yminus);
        int k=0;
        while(absolute((xminustemp-xminus)/xminus)>error_step&&k<100){
            k++;
            if(xminustemp>xminus)
                xnextmax=x;
            else
                xnextmin=x;
            x=(xnextmax-xnextmin)/2+xnextmin;
            inverseVecN(x,f(x),xminustemp,yminus);
        }
        xstep = x - xbefore+0.0001;

    }
    return DistributionRegularStep(m);
}

F64 Statistics::moment(const Distribution &f, int moment,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F64 integral=0;
    F64 normalization=0;
    while(it.next()){

        integral +=f.operator ()(it.x())*step*std::pow(it.x(),moment);
        normalization+=f.operator ()(it.x())*step;
    }
    return integral/normalization;
}

F64 Statistics::norm(const Distribution &f, F64 norm,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F64 integral=0;
    while(it.next()){
        integral +=std::pow(absolute(f.operator ()(it.x())),norm)*step;
    }
    return std::pow(integral,1./norm);
}
F64 Statistics::productInner(const Distribution &f,const Distribution &g, F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);
    g.setStep(step);
    F64 sum=0;
    while(it.next()){
        sum +=f.operator ()(it.x())*step*g.operator ()(it.x());
    }
    return sum;
}


F64 Statistics::FminusOneOfYMonotonicallyIncreasingFunction(const Distribution &f, F64 y,F64 xmin, F64 xmax,F64 mindiff)
{

    if(const DistributionRegularStep* dist =dynamic_cast<const DistributionRegularStep*>(f.___getPointerImplementation()))
    {
        return dist->fMinusOneForMonotonicallyIncreasing(y);
    }
    F64 xmincurrent=xmin;
    F64 xmaxcurrent=xmax;
    F64 ytemp;
    do{
        F64 step =  (xmaxcurrent - xmincurrent)/2+xmincurrent ;
        ytemp = f.operator ()(step);
        if(ytemp<y)
            xmincurrent = step;
        else
            xmaxcurrent = step;
    }while(xmaxcurrent-xmincurrent>=mindiff);
    return ytemp;
}

DistributionRegularStep Statistics::derivate(const Distribution &f, F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);


    Mat2F64 m(it.size()-1,2);
    F64 x=NumericLimits<F64>::maximumRange();
    int i=0;
    while(it.next()){
        if(x!=NumericLimits<F64>::maximumRange()){
            m(i-1,0)=x;
            m(i-1,1)=(f.operator ()(it.x())-f.operator ()(x))/step;
        }
        x = it.x();
        i++;
    }
    return DistributionRegularStep(m);
}



F64 Statistics::argMin(const Distribution &f,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F64 arg=0;
    F64 min=NumericLimits<F64>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())<min)
        {
            min = f.operator ()(it.x());
            arg = it.x();
        }
    }
    return arg;
}

std::vector<F64> Statistics::argMaxLocal(const Distribution &f,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);
    std::vector<F64> v_argmax;
    it.next();
    F64 max_minus_1=f.operator ()(it.x());

    it.next();
    F32 arg_max=it.x();
    F64 max=f.operator ()(it.x());

    bool perhaps_max=false;
    F64 arg_perhaps_max=false;
    while(it.next()){
        F32 arg_max_plus_1=it.x();
        F64 max_plus_1=f.operator ()(it.x());
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


//F64 Statistics::argMin(const Distribution &f,F64 xmin, F64 xmax,F64 step)
//{

//    DistributionIteratorERegularInterval it(xmin,xmax,step);

//    F64 arg=0;
//    F64 min=NumericLimits<F64>::maximumRange();
//    while(it.next()){
//        if(f.operator ()(it.x())<min)
//        {
//            min = f.operator ()(it.x());
//            arg = it.x();
//        }
//    }
//    return arg;
//}

Vec<F64> Statistics::argMinLocal(const Distribution &f,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);
    Vec<F64> v_argmin;
    it.next();
    F64 min_minus_1=f.operator ()(it.x());

    it.next();
    F32 arg_min=it.x();
    F64 min=f.operator ()(it.x());

    bool perhaps_min=false;
    F64 arg_perhaps_min=false;
    while(it.next()){
        F32 arg_min_plus_1=it.x();
        F64 min_plus_1=f.operator ()(it.x());
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

F64 Statistics::argMax(const Distribution &f,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F64 arg=0;
    F64 max=-NumericLimits<F64>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())>max)
        {
            max = f.operator ()(it.x());
            arg = it.x();
        }
    }
    return arg;
}

F64 Statistics::minValue( const Distribution &f,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F64 min=NumericLimits<F64>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())<min){
            min = f.operator ()(it.x());
        }
    }
    return min;
}
F64 Statistics::maxValue( const Distribution &f,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F64 max=-NumericLimits<F64>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())>max){
            max = f.operator ()(it.x());
        }
    }
    return max;
}
DistributionRegularStep Statistics::toProbabilityDistribution( const Distribution &f,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F64 sum=0;

    while(it.next()){
        sum += absolute(f.operator ()(it.x()))*step;
    }
    it.init();
    Mat2F64 m(it.size(),2);
    int i =0;
    while(it.next()){
        m(i,0)=it.x();
        m(i,1)=absolute(f.operator ()(it.x()))/sum;
        //        std::cout<<absolute(f.operator ()(it.x()))/sum/step<<std::endl;
        i++;
    }
    return DistributionRegularStep(m);
}
DistributionRegularStep Statistics::toCumulativeProbabilityDistribution( const Distribution &f,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);

    F64 sum=0;

    while(it.next()){
        sum += absolute(f.operator ()(it.x()));
    }
    it.init();
    Mat2F64 m(it.size(),2);
    int i =0;
    F64 sumtemp =0;
    while(it.next()){
        m(i,0)=it.x();
        sumtemp += absolute(f.operator ()(it.x()));
        m(i,1)=sumtemp/sum;
        i++;
    }
    return DistributionRegularStep(m);
}
DistributionRegularStep Statistics::toStepFunction(const Distribution &f,F64 xmin, F64 xmax,F64 step)
{

    DistributionIteratorERegularInterval it(xmin,xmax,step);
    it.init();
    Mat2F64 m(it.size(),2);
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
Mat2F64 Statistics::toMatrix(const Distribution &f,F64 xmin, F64 xmax,F64 step)
{
    DistributionIteratorERegularInterval it(xmin,xmax,step);

    it.init();
    Mat2F64 m(it.size(),2);
    int i =0;
    while(it.next()){
        m(i,0)=it.x();
        m(i,1)=f.operator ()(it.x());
        i++;
    }
    return m;
}



DistributionIntegerRegularStep Statistics::computedStaticticsFromIntegerRealizations( const VecI32 & v)throw(pexception){
    VecI32 vv(v);
    std::sort (vv.begin(), vv.end());
    Mat2F64 m(vv(vv.size()-1)-vv(0)+1,2);

    for(unsigned int i=0;i<vv.size();i++){
        m(vv(i)-vv(0),1)++;
    }
    for(unsigned int i=0;i<m.sizeI();i++){
        m(i,0)=vv(0)+i;
        m(i,1)/=vv.size();
    }
    try{
        DistributionIntegerRegularStep dd(m);
        return dd;
    }catch(const pexception & ){
        throw(pexception("In Statistics::computedStaticticsFromRealRealizations, not enough realization in your collection"));
    }

}
DistributionRegularStep Statistics::computedStaticticsFromRealRealizationsWithWeight( const VecF64 & v,const VecF64 & weight,F64 step,F64 min,F64 max)
{
    POP_DbgAssertMessage(v.size()!=0,"Vecs should contain at least one element");
    POP_DbgAssertMessage(v.size()==weight.size(),"Two vectors should have the same size");
    if(min==NumericLimits<F64>::minimumRange()){
        min = *std::min_element(v.begin(),v.end());
    }
    if(max==NumericLimits<F64>::maximumRange()){
        max = *std::max_element(v.begin(),v.end());
    }
    int nbr_step = std::floor((max-min)/step);
    Mat2F64 m(nbr_step,2);
    for(unsigned int i=0;i<v.size();i++){
        int value = std::floor((v(i)-min)/step);
        if(value<static_cast<int>(m.sizeI())&&value>=0)
            m(value,1)+=weight(i);
    }
    double sum=0;
    double incr = min;
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
DistributionRegularStep Statistics::computedStaticticsFromRealRealizations(  const VecF64 & v,F64 step,F64 min,F64 max)
{
    if(min==NumericLimits<F64>::minimumRange()){
        min = *std::min_element(v.begin(),v.end());
    }
    if(max==NumericLimits<F64>::maximumRange()){
        max = *std::max_element(v.begin(),v.end());
    }
    int nbr_step = std::floor((max-min)/step);
    Mat2F64 m(nbr_step,2);
    for(unsigned int i=0;i<v.size();i++){
        int value = std::floor((v(i)-min)/step);
        if(value<static_cast<int>(m.sizeI())&&value>=0)
            m(value,1)++;
    }
    int sum=0;
    double incr = min;
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


F64  Statistics::maxRangeIntegral(const Distribution & f,F64 integralvalue,F64 min,F64 maxsupremum ,F64 step  ){
    DistributionIteratorERegularInterval it(min,maxsupremum,step);
    it.init();
    F64 sum=0;
    while(it.next()){
        sum+=f.operator ()(it.x())*step;
        if(sum>integralvalue)
            return it.x();
    }
    return maxsupremum;
}


F64 Statistics::moment(const DistributionMultiVariate &f, VecF64 moment,VecF64 xmin, VecF64 xmax,F64 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    F64 integral=0;
    F64 normalization=0;

    while(it.next()){
        F64 sum=1;
        for(unsigned int i =0;i<moment.size();i++)
            sum *= std::pow(it.x()(i),moment(i));
        integral +=f.operator ()(it.x())*sum;
        normalization+=f.operator ()(it.x());
    }
    return integral/normalization;
}


VecF64 Statistics::argMin(const DistributionMultiVariate &f,VecF64 xmin, VecF64 xmax,F64 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    VecF64 arg;
    F64 min=NumericLimits<F64>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())<min)
        {
            min = f.operator ()(it.x());
            arg = it.x();
        }
    }
    return arg;
}
VecF64 Statistics::argMax(const DistributionMultiVariate &f,VecF64 xmin, VecF64 xmax,F64 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    VecF64 arg;
    F64 max=-NumericLimits<F64>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())>max)
        {
            max = f.operator ()(it.x());
            arg = it.x();
        }
    }
    return arg;
}

F64 Statistics::minValue( const DistributionMultiVariate &f,VecF64 xmin, VecF64 xmax,F64 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    F64 min=NumericLimits<F64>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())<min){
            min = f.operator ()(it.x());
        }
    }
    return min;
}
F64 Statistics::maxValue( const DistributionMultiVariate &f,VecF64 xmin, VecF64 xmax,F64 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    F64 max=-NumericLimits<F64>::maximumRange();
    while(it.next()){
        if(f.operator ()(it.x())>max){
            max = f.operator ()(it.x());
        }
    }
    return max;
}
DistributionMultiVariateRegularStep  Statistics::integral(const DistributionMultiVariate &f, VecF64 xmin, VecF64 xmax,F64 step)throw(pexception)
{
    if(f.getNbrVariable()!=2)
        throw(pexception("Only for two variate distribution"));
    DistributionMultiVariateIteratorE it(xmin,xmax,step);
    Mat2F64 m( Vec2I32(it._domain(0),it._domain(1)));
    double steppower2 = step*step;
    while(it.next()){
        VecF64 x = it.xInteger();
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
DistributionMultiVariateRegularStep Statistics::toProbabilityDistribution( const DistributionMultiVariate &f,VecF64 xmin, VecF64 xmax,F64 step)
{

    DistributionMultiVariateIteratorE it(xmin,xmax,step);

    F64 sum=0;
    while(it.next()){
        sum += absolute(f.operator ()(it.x()));
    }
    it.init();
    Mat2F64 m( Vec2I32(it._domain(0),it._domain(1)));
    while(it.next()){
        VecF64 x = it.xInteger();
        m (x(0),x(1)) = absolute(f.operator ()(it.x()))/sum;
    }
    DistributionMultiVariateRegularStep d(m,xmin,step);
    return d;
}

Mat2F64 Statistics::toMatrix( const DistributionMultiVariate &f,VecF64 xmin, VecF64 xmax,F64 step){
    DistributionMultiVariateIteratorE it(xmin,xmax,step);
    Mat2F64 m( Vec2I32(it._domain(0),it._domain(1)));
    while(it.next()){
        VecF64 x = it.xInteger();
        m (x(0),x(1)) = f.operator ()(it.x());
    }
    return m;

}

}
