#include"data/notstable/Ransac.h"
#include"algorithm/GeometricalTransformation.h"
namespace pop {
LinearLeastSquareRANSACModel::LinearLeastSquareRANSACModel()
:_error(NumericLimits<F32>::maximumRange()){


}
std::ostream& operator << (std::ostream& out, const LinearLeastSquareRANSACModel::Data & m){
    out<<m.X<<" and Y="<<m.Y<<std::endl;
    return out;
}
LinearLeastSquareRANSACModel::LinearLeastSquareRANSACModel(const Vec<Data >& data)
{
    Mat2F32 X(static_cast<int>(data.size()),static_cast<int>(data[0].X.size()));
    VecF32 Y(static_cast<int>(data.size()));
    for(unsigned int i =0;i<static_cast<unsigned int>(data.size());i++){
        X.setRow(i,data[i].X);
        Y(i)=data[i].Y;
    }
    _beta = LinearAlgebra::linearLeastSquares(X,Y);
    VecF32 Ymodel = X*_beta;
    _error = (Ymodel-Y).norm();

}

F32 LinearLeastSquareRANSACModel::getError(){
    return _error;
}
F32 LinearLeastSquareRANSACModel::getError(const Data & p){
    return absolute(productInner(p.X,_beta)-p.Y);
}
pop::VecF32 LinearLeastSquareRANSACModel::getBeta(){
    return _beta;
}
 unsigned int LinearLeastSquareRANSACModel::getNumberDataFitModel()const{
    return 2;
}
LinearLeastSquareRANSACModel::Data::Data(pop::VecF32 x,pop::F32 y)
    :X(x),Y(y)
{
}
LinearLeastSquareRANSACModel::Data::Data()
{
}
GeometricalTransformationRANSACModel::~GeometricalTransformationRANSACModel()
{
}
GeometricalTransformationRANSACModel::GeometricalTransformationRANSACModel()
    :_error(NumericLimits<F32>::maximumRange()){
}
GeometricalTransformationRANSACModel::Data::Data()
{}

GeometricalTransformationRANSACModel::Data::Data(pop::Vec2F32 src,pop::Vec2F32 dst)
    :_src(src),_dst(dst)
{}

F32 GeometricalTransformationRANSACModel::getError(){
    return _error;
}
F32 GeometricalTransformationRANSACModel::getError(const Data & data){
    Vec2F32 src0= data._src;
    Vec2F32 dst0= data._dst;
    Vec2F32 dst0approx= GeometricalTransformation::transformHomogeneous2D(_geometricaltransformation,src0);
    return (dst0-dst0approx).norm();
}

pop::Mat2x33F32 GeometricalTransformationRANSACModel::getTransformation(){
    return _geometricaltransformation;
}

AffineTransformationRANSACModel::AffineTransformationRANSACModel()

{}

AffineTransformationRANSACModel::AffineTransformationRANSACModel(const Vec<Data >& data)
{
    Vec2F32 src[3];
    Vec2F32 dst[3];

    src[0]=data(0)._src;
    src[1]=data(1)._src;
    src[2]=data(2)._src;

    dst[0]=data(0)._dst;
    dst[1]=data(1)._dst;
    dst[2]=data(2)._dst;
    this->_geometricaltransformation = GeometricalTransformation::affine2D(src,dst,true);
    if(data.size()==getNumberDataFitModel())
        _error = NumericLimits<F32>::maximumRange();
    else{
        _error = 0;
        for(unsigned int i=3;i<data.size();i++){
            Vec2F32 src0= data(i)._src;
            Vec2F32 dst0= data(i)._dst;
            Vec2F32 dst0approx= GeometricalTransformation::transformHomogeneous2D(this->_geometricaltransformation,src0);
            F32 dist = (dst0-dst0approx).norm();

            _error+=dist;
        }
        _error/=(data.size()-getNumberDataFitModel());
    }

}
 unsigned int AffineTransformationRANSACModel::getNumberDataFitModel()const{
    return 3;
}

ProjectionTransformationRANSACModel::ProjectionTransformationRANSACModel(const Vec<Data >& data)
{
    Vec2F32 src[4];
    Vec2F32 dst[4];

    src[0]=data(0)._src;
    src[1]=data(1)._src;
    src[2]=data(2)._src;
    src[3]=data(3)._src;

    dst[0]=data(0)._dst;
    dst[1]=data(1)._dst;
    dst[2]=data(2)._dst;
    dst[3]=data(3)._dst;
    this->_geometricaltransformation = GeometricalTransformation::projective2D(src,dst,true);
    if(data.size()==getNumberDataFitModel())
        _error = NumericLimits<F32>::maximumRange();
    else{
        _error = 0;
        for(unsigned int i=this->getNumberDataFitModel();i<data.size();i++){
            Vec2F32 src0= data(i)._src;
            Vec2F32 dst0= data(i)._dst;
            Vec2F32 dst0approx= GeometricalTransformation::transformHomogeneous2D(this->_geometricaltransformation,src0);
            F32 dist = (dst0-dst0approx).norm();
            _error+=dist;
        }
        _error/=(data.size()-getNumberDataFitModel());
    }

}
ProjectionTransformationRANSACModel::ProjectionTransformationRANSACModel(){}
 unsigned int ProjectionTransformationRANSACModel::getNumberDataFitModel()const{
    return 4;
}
}
