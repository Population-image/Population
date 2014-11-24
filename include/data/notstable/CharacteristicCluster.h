#ifndef CARACTERISTICCLUSTER_H
#define CARACTERISTICCLUSTER_H
#include"data/vec/VecN.h"
#include"data/mat/MatN.h"
#include"data/ocr/OCR.h"
#include"data/mat/MatNDisplay.h"
namespace pop {

//TODO remove this part

class POP_EXPORTS CharacteristicCluster
{
public:
    CharacteristicCluster();
    void setLabel(int label);
    void addPoint(const Vec2I32 & x);
    Vec2I32 size()const;
    Vec2I32 center()const;
    int _label;
    int _mass;
    Vec2I32 _min;
    Vec2I32 _max;
    Vec2F64 _barycentre;
};



struct POP_EXPORTS CharacteristicClusterDistance{
    virtual double operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b)=0;
};
struct POP_EXPORTS CharacteristicClusterDistanceMass : public CharacteristicClusterDistance{
    double operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b);
};
struct POP_EXPORTS CharacteristicClusterDistanceHeight : public CharacteristicClusterDistance{
    double operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b);
};
struct POP_EXPORTS CharacteristicClusterDistanceWidth : public CharacteristicClusterDistance{
    double operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b);
};
struct POP_EXPORTS CharacteristicClusterDistanceWidthInterval : public CharacteristicClusterDistance{
    double operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b);
};
struct POP_EXPORTS CharacteristicClusterDistanceHeightInterval : public CharacteristicClusterDistance{
    double operator ()(const CharacteristicCluster& a,const CharacteristicCluster& b);
};

struct POP_EXPORTS CharacteristicClusterFilter{
    double _min;
    double _max;
    CharacteristicClusterFilter();
    virtual ~CharacteristicClusterFilter();
    virtual bool operator ()(const CharacteristicCluster& a);
};
struct POP_EXPORTS CharacteristicClusterFilterMass : public CharacteristicClusterFilter{

    bool operator ()(const CharacteristicCluster& a);
};
struct POP_EXPORTS CharacteristicClusterFilterHeight : public CharacteristicClusterFilter{

    bool operator ()(const CharacteristicCluster& a);
};
struct POP_EXPORTS CharacteristicClusterFilterWidth : public CharacteristicClusterFilter{

    bool operator ()(const CharacteristicCluster& a);
};
struct POP_EXPORTS CharacteristicClusterFilterAsymmetryHeightPerWidth : public CharacteristicClusterFilter{

    bool operator ()(const CharacteristicCluster& a);
};
Vec<CharacteristicCluster> applyCharacteristicClusterFilter(const Vec<CharacteristicCluster>& v_cluster, CharacteristicClusterFilter * filter);

pop::Mat2UI32 POP_EXPORTS applyClusterFilter(const pop::Mat2UI32& labelled_image,const Vec<CharacteristicClusterFilter*> v_filter  );
Vec<Vec<Mat2UI8> > POP_EXPORTS applyGraphCluster(const pop::Mat2UI32& labelled_image, pop::Vec<CharacteristicClusterDistance*> v_dist,  pop::Vec<double> v_weight,double threshold  );




//version 2



class CharacteristicMass
{
private:
    int _mass;
public:
    CharacteristicMass();
    template<typename Position>
    void addPoint(const Position & ){
        _mass++;
    }
    int getMass()const;
};


template<typename Position=Vec2I32>
class CharacteristicBoundingBox
{
private:
    Position _min,_max;
public:
    CharacteristicBoundingBox()
        :_min(NumericLimits<Position>::maximumRange()),_max(NumericLimits<Position>::minimumRange())
    {
    }

    void addPoint(const Position & x){

        _min = pop::minimum(x,_min);
        _max = pop::maximum(x,_max);
    }
    Position getSize()const{
        return _max-_min+1;
    }
    Position getCenter()const{
        return (_max +_min)/2;
    }
    Position getMax()const{
        return _max;
    }
    Position getMin()const{
        return _min;
    }
    double getAsymmetryHeighPerWidth()const{
        return static_cast<double>(getSize()(0))/getSize()(1);
    }

};
template<typename Function=Mat2UI8>
class CharacteristicGreyLevel
{
    const Function * _m;
        int _nbr_point;
    double  _mean;
    double  _variance;
    mutable double  _standart_deviation;

public:
    CharacteristicGreyLevel()
        :_m(NULL),_nbr_point(0),_mean(0),_variance(0),_standart_deviation(0){}
    void addPoint(const typename Function::E & x){
        POP_DbgAssertMessage(_m==NULL,"Set an sMatrix to the CharacteristicGreyLevelMean");
        _mean= (_mean *_nbr_point +_m->operator ()(x))/(_nbr_point+1);
        _variance = (_mean *_nbr_point + (_m->operator ()(x)-_mean)*(_m->operator ()(x)-_mean) )/(_nbr_point+1);
        _nbr_point++;
    }
    void setMatrix(const Function *m){
        _m = m;
    }

    double getMean()const{
        return _mean;
    }
    double getStandartDeviation()const{
        if(_standart_deviation==0)
            _standart_deviation = std::sqrt(_variance);
        return _standart_deviation;
    }
};

class CharacteristicLabel
{
private:
    int _label;
public:
    void setLabel(int label);
    int getLabel()const;
    //void addPoint(const typename Function::E & ){
//    }
};

template<typename TypeCharacteristic >
struct DistanceCharacteristic{
    virtual double operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b)=0;
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicMass : public DistanceCharacteristic<TypeCharacteristic>{
    double operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return (static_cast<double>(std::max(a.getMass(),b.getMass()))/std::min(a.getMass(),b.getMass()))-1;
    }
};

template<typename TypeCharacteristic >
struct DistanceCharacteristicGreyLevel : public DistanceCharacteristic<TypeCharacteristic>{
    double operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return std::abs(a.getMean()-b.getMean())/std::min(a.getStandartDeviation(),b.getStandartDeviation());
    }
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicHeigh : public DistanceCharacteristic<TypeCharacteristic>{
    double operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return (static_cast<double>(std::max(a.getSize()(0),b.getSize()(0))))/std::min(a.getSize()(0),b.getSize()(0))-1;
    }
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicWidth : public DistanceCharacteristic<TypeCharacteristic>{
    double operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return (static_cast<double>(std::max(a.getSize()(1),b.getSize()(1))))/std::min(a.getSize()(1),b.getSize()(1))-1;
    }
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicWidthInterval : public DistanceCharacteristic<TypeCharacteristic>{
    double operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return static_cast<double>(std::abs(a.getCenter()(1)-b.getCenter()(1)))/ std::min(a.getSize()(1),b.getSize()(1));
    }
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicHeightInterval : public DistanceCharacteristic<TypeCharacteristic>{
    double operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return static_cast<double>(std::abs(a.getCenter()(0)-b.getCenter()(0)))/  std::min(a.getSize()(0),b.getSize()(0));
    }
};
template<typename TypeCharacteristic>
struct DistanceSumCharacteristic  : public DistanceCharacteristic<TypeCharacteristic>
{

    std::vector<Distribution *> _v_dist;
    std::vector<DistanceCharacteristic<TypeCharacteristic> * > _v_carac;

    void addDistance(Distribution* function ,DistanceCharacteristic<TypeCharacteristic> * dist ){
        _v_dist.push_back(function);
        _v_carac.push_back(dist);
    }

    double operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        double sum=0;
        for(unsigned int index = 0; index<_v_dist.size(); index++){
            sum+=_v_dist[index]->operator()(_v_carac[index]->operator()(a,b));
        }
        return sum;
    }
};
template<typename Function>
struct CharacteristicClusterMix :  CharacteristicMass, CharacteristicBoundingBox<typename Function::E>, CharacteristicGreyLevel< Function>
{
    void addPoint(const typename Function::E & x){
        CharacteristicMass::addPoint(x);
        CharacteristicBoundingBox<typename Function::E>::addPoint(x);
        CharacteristicGreyLevel<Function>::addPoint(x);
    }
};




}
#endif // CARACTERISTICCLUSTER_H
