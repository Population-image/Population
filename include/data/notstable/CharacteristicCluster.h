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

template<typename Position=Vec2I32 >
class ListOfPoints {
private:
    Vec<Position> _list_points;
public:
    inline Vec<Position>& getListPoints() {
        return _list_points;
    }

    void addPoint(const Position& x) {
        _list_points.push_back(x);
    }
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

    virtual void addPoint(const Position & x){

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

template<typename Position=Vec2I32>
class CharacteristicBoundingBoxForRectangle : public CharacteristicBoundingBox<Position> {
    std::vector<Position> _set_points;
    // besides _xmin and _xmax, we also keep track of the other two corners of the (supposed) rectangle
    Position _cornerUpRight, _cornerLowLeft;
public:
    inline std::vector<Position> getVectorPoints() const {
        return _set_points;
    }

    inline Position getCornerUpRight() const {
        return _cornerUpRight;
    }

    inline Position getCornerLowLeft() const {
        return _cornerLowLeft;
    }

    virtual void addPoint(const Position & x) {
        CharacteristicBoundingBox<Position>::addPoint(x);
        _cornerUpRight(0) = CharacteristicBoundingBox<Position>::getMin()(0);
        _cornerUpRight(1) = CharacteristicBoundingBox<Position>::getMax()(1);
        _cornerLowLeft(0) = CharacteristicBoundingBox<Position>::getMax()(0);
        _cornerLowLeft(1) = CharacteristicBoundingBox<Position>::getMin()(1);
        _set_points.push_back(x);
    }

    // to see if a point x is in the upper-left, upper-right, lower-left, or lower-right region of the (supposed) rectangle
    // output:
    // 0 : upper left
    // 1 : upper right
    // 2 : lower left
    // 3 : lower right
    int getCorner(const Position& x) {
        int res;
        if (x(0) < CharacteristicBoundingBox<Position>::getCenter()(0)) {
            return (x(1) < CharacteristicBoundingBox<Position>::getCenter()(1)? 0 : 1);
        } else {
            return (x(1) < CharacteristicBoundingBox<Position>::getCenter()(1)? 2 : 3);
        }
        return res;
    }

    F32 distanceToBound() {
        F32 sumDistance = 0;
        Position nearestCorner;
        for (typename std::vector<Position>::iterator it = getVectorPoints().begin() ; it != getVectorPoints().end() ; ++ it) {
            switch (getCorner(*it)) {
                case 0 : {
                    nearestCorner = CharacteristicBoundingBox<Position>::getMin();
                    break;
                }
                case 1 : {
                    nearestCorner = _cornerUpRight;
                    break;
                }
                case 2 : {
                    nearestCorner = _cornerLowLeft;
                    break;
                }
                case 3 : {
                    nearestCorner = CharacteristicBoundingBox<Position>::getMax();
                    break;
                }
                default : {
                    std::cerr << "Value of getCorner() is not valid" << std::endl;
                    break;
                }
            }
            Position distance(nearestCorner - *it);
            sumDistance += std::min(std::abs(distance(0)), std::abs(distance(1)));
            //sumDistance += std::abs(distance(0));
        }
        return sumDistance;
    }

    F32 getHeight() {
        return CharacteristicBoundingBox<Position>::getSize()(0);
    }

    F32 getWidth() {
        return CharacteristicBoundingBox<Position>::getSize()(1);
    }
};

template<int DIM,typename TypePixel>
class CharacteristicGreyLevel
{
    const MatN<DIM,TypePixel> * _m;
        int _nbr_point;
    typename ArithmeticsTrait<TypePixel,F32>::Result  _mean;
    typename ArithmeticsTrait<TypePixel,F32>::Result  _variance;

public:
    CharacteristicGreyLevel()
        :_m(NULL),_nbr_point(0),_mean(0),_variance(0)
    {}

    void addPoint(const VecN<DIM,I32> & x){
        POP_DbgAssertMessage(_m!=NULL,"Set an sMatrix to the CharacteristicGreyLevelMean");

        _mean= (_mean *_nbr_point +_m->operator ()(x))/(_nbr_point+1);
        _variance = (_mean *_nbr_point + (_m->operator ()(x)-_mean)*(_m->operator ()(x)-_mean) )/(_nbr_point+1);
        _nbr_point++;
    }
    void setMatrix(const MatN<DIM,TypePixel>  *m){
        _m = m;
    }

    typename ArithmeticsTrait<TypePixel,F32>::Result getMean()const{
        return _mean;
    }
    F32 getStandartDeviation()const{
        F32 value = normValue(_variance);
        if(value==0)
            return 0;
        return
             std::sqrt(value);
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
        return pop::normValue(a.getMean()-b.getMean())/(std::min)(a.getStandartDeviation(),b.getStandartDeviation());
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
template<int DIM,typename TypePixel>
struct CharacteristicClusterMix :  CharacteristicMass, CharacteristicBoundingBox<VecN<DIM,I32> >, CharacteristicGreyLevel<DIM ,TypePixel >
{
    void addPoint(const VecN<DIM,I32> & x){
        CharacteristicMass::addPoint(x);
        CharacteristicBoundingBox<VecN<DIM,I32> >::addPoint(x);
        CharacteristicGreyLevel<DIM ,TypePixel>::addPoint(x);
    }
};

template<int DIM, typename TypePixel>
struct CharacteristicClusterMixForRectangle : CharacteristicMass, CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >, CharacteristicGreyLevel<DIM, TypePixel > {
    void addPoint(const VecN<DIM, I32> & x) {
        CharacteristicMass::addPoint(x);
        CharacteristicBoundingBoxForRectangle<VecN<DIM, I32 > >::addPoint(x);
        CharacteristicGreyLevel<DIM, TypePixel>::addPoint(x);
    }

    bool isEmpty() {
        return (CharacteristicMass::getMass() == 0);
    }

    F32 averageDistanceToBound() {
        return CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::distanceToBound()/CharacteristicMass::getMass();
    }

    F32 averageDistanceToBoundOverSize() {
        F32 distance = averageDistanceToBound();
        F32 size = std::min(CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getWidth(), CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getHeight());
        if (size > 0 ) {
            return distance / size;
        } else {
            // an arbitrarily large value
            return 1000.f;
        }
    }

    bool isRectangle() {
        F32 size = std::min(CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getWidth(), CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getHeight());
        if (!isEmpty() && size > 50) {
            std::cout << averageDistanceToBoundOverSize() << std::endl;
        }
        return (!isEmpty() && size > 50 && averageDistanceToBoundOverSize() < 0.2 && (CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getWidth() > CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getHeight()));
    }

    // return a vector of 4 corners of the rectangle
    std::vector<pop::Vec2I32> getVertex() {
        std::vector<pop::Vec2I32> vertex;
        vertex.push_back(CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getMin());
        vertex.push_back(CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getCornerUpRight());
        vertex.push_back(CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getCornerLowLeft());
        vertex.push_back(CharacteristicBoundingBoxForRectangle<VecN<DIM, I32> >::getMax());
        return vertex;
    }
};

template<int DIM,typename TypePixel>
struct CharacteristicClusterMixListPoints :  CharacteristicMass, CharacteristicBoundingBox<VecN<DIM,I32> >, CharacteristicGreyLevel<DIM ,TypePixel >, ListOfPoints<VecN<DIM, I32 > >
{
    void addPoint(const VecN<DIM,I32> & x){
        CharacteristicMass::addPoint(x);
        CharacteristicBoundingBox<VecN<DIM,I32> >::addPoint(x);
        CharacteristicGreyLevel<DIM ,TypePixel>::addPoint(x);
        ListOfPoints<VecN<DIM, I32 > >::addPoint(x);
    }

    Mat2UI8 toMatrix() {
        Mat2UI8 mat(CharacteristicBoundingBox<VecN<DIM, I32 > >::getSize());
        for (typename Vec<VecN<DIM, I32> >::iterator it = ListOfPoints<VecN<DIM, I32> >::getListPoints().begin() ; it != ListOfPoints<VecN<DIM, I32 > >::getListPoints().end() ; ++it) {
            mat(*it - CharacteristicBoundingBox<VecN<DIM, I32 > >::getMin()) = 255;
        }
        return mat;
    }

    VecN<DIM, I32 > getMax() const {
        return CharacteristicBoundingBox<VecN<DIM, I32 > >::getMax();
    }

    VecN<DIM, I32 > getMin() const {
        return CharacteristicBoundingBox<VecN<DIM, I32 > >::getMin();
    }
};

}
#endif // CARACTERISTICCLUSTER_H
