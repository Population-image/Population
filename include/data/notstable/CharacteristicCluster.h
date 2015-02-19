#ifndef CARACTERISTICCLUSTER_H
#define CARACTERISTICCLUSTER_H
#include"data/vec/VecN.h"
#include"data/mat/MatN.h"
#include"data/ocr/OCR.h"
#include"data/mat/MatNDisplay.h"
namespace pop {
class CharacteristicMass
{
private:
    int _mass;
public:
    inline CharacteristicMass()    :_mass(0){}
    template<typename Position>
    void addPoint(const Position & ){
        _mass++;
    }
    inline int getMass()const{ return _mass;}
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
    F32 getAsymmetryHeighPerWidth()const{
        return static_cast<F32>(getSize()(0))/getSize()(1);
    }

};
template<typename Function=Mat2UI8>
class CharacteristicGreyLevel
{
    const Function * _m;
        int _nbr_point;
    F32  _mean;
    F32  _variance;
    mutable F32  _standart_deviation;

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

    F32 getMean()const{
        return _mean;
    }
    F32 getStandartDeviation()const{
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
    inline void setLabel(int label){_label = label; }
    inline int getLabel()const{return _label;
}
};

template<typename TypeCharacteristic >
struct DistanceCharacteristic{
    virtual F32 operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b)=0;
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicMass : public DistanceCharacteristic<TypeCharacteristic>{
    F32 operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return (static_cast<F32>(((std::max))(a.getMass(),b.getMass()))/(std::min)(a.getMass(),b.getMass()))-1;
    }
};

template<typename TypeCharacteristic >
struct DistanceCharacteristicGreyLevel : public DistanceCharacteristic<TypeCharacteristic>{
    F32 operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return std::abs(a.getMean()-b.getMean())/(std::min)(a.getStandartDeviation(),b.getStandartDeviation());
    }
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicHeigh : public DistanceCharacteristic<TypeCharacteristic>{
    F32 operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return (static_cast<F32>((std::max)(a.getSize()(0),b.getSize()(0))))/(std::min)(a.getSize()(0),b.getSize()(0))-1;
    }
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicWidth : public DistanceCharacteristic<TypeCharacteristic>{
    F32 operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return (static_cast<F32>((std::max)(a.getSize()(1),b.getSize()(1))))/(std::min)(a.getSize()(1),b.getSize()(1))-1;
    }
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicWidthInterval : public DistanceCharacteristic<TypeCharacteristic>{
    F32 operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return static_cast<F32>(std::abs(a.getCenter()(1)-b.getCenter()(1)))/ (std::min)(a.getSize()(1),b.getSize()(1));
    }
};
template<typename TypeCharacteristic >
struct DistanceCharacteristicHeightInterval : public DistanceCharacteristic<TypeCharacteristic>{
    F32 operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        return static_cast<F32>(std::abs(a.getCenter()(0)-b.getCenter()(0)))/  (std::min)(a.getSize()(0),b.getSize()(0));
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

    F32 operator ()(const TypeCharacteristic& a,const TypeCharacteristic& b){
        F32 sum=0;
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
