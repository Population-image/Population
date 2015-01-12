#include"data/notstable/Ransac.h"
#include"algorithm/GeometricalTransformation.h"
namespace pop {
LinearLeastSquareRANSACModel::LinearLeastSquareRANSACModel(const Vec<Data >& data)
{
    Mat2F64 X(data.size(),data[0].X.size());
    VecF64 Y(data.size());
    for(unsigned int i =0;i<data.size();i++){
        X.setRow(i,data[i].X);
        Y(i)=data[i].Y;
    }
    _beta = LinearAlgebra::linearLeastSquares(X,Y);
    VecF64 Ymodel = X*_beta;
    _error = (Ymodel-Y).norm();

}

double LinearLeastSquareRANSACModel::getError(){
    return _error;
}
double LinearLeastSquareRANSACModel::getError(const Data & p){
    return absolute(productInner(p.X,_beta)-p.Y);
}
pop::VecF64 LinearLeastSquareRANSACModel::getBeta(){
    return _beta;
}
 unsigned int LinearLeastSquareRANSACModel::getNumberDataFitModel()const{
    return 2;
}
LinearLeastSquareRANSACModel::Data::Data(pop::VecF64 x,pop::F64 y)
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
    :_error(NumericLimits<double>::maximumRange()){
}
GeometricalTransformationRANSACModel::Data::Data()
{}

GeometricalTransformationRANSACModel::Data::Data(pop::Vec2F64 src,pop::Vec2F64 dst)
    :_src(src),_dst(dst)
{}

double GeometricalTransformationRANSACModel::getError(){
    return _error;
}
double GeometricalTransformationRANSACModel::getError(const Data & data){
    Vec2F64 src0= data._src;
    Vec2F64 dst0= data._dst;
    Vec2F64 dst0approx= GeometricalTransformation::transformHomogeneous2D(_geometricaltransformation,src0);
    return (dst0-dst0approx).norm();
}

pop::Mat2x33F64 GeometricalTransformationRANSACModel::getTransformation(){
    return _geometricaltransformation;
}

AffineTransformationRANSACModel::AffineTransformationRANSACModel()

{}

AffineTransformationRANSACModel::AffineTransformationRANSACModel(const Vec<Data >& data)
{
    Vec2F64 src[3];
    Vec2F64 dst[3];

    src[0]=data(0)._src;
    src[1]=data(1)._src;
    src[2]=data(2)._src;

    dst[0]=data(0)._dst;
    dst[1]=data(1)._dst;
    dst[2]=data(2)._dst;
    this->_geometricaltransformation = GeometricalTransformation::affine2D(src,dst,true);
    if(data.size()==getNumberDataFitModel())
        _error = NumericLimits<double>::maximumRange();
    else{
        _error = 0;
        for(unsigned int i=3;i<data.size();i++){
            Vec2F64 src0= data(i)._src;
            Vec2F64 dst0= data(i)._dst;
            Vec2F64 dst0approx= GeometricalTransformation::transformHomogeneous2D(this->_geometricaltransformation,src0);
            double dist = (dst0-dst0approx).norm();

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
    Vec2F64 src[4];
    Vec2F64 dst[4];

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
        _error = NumericLimits<double>::maximumRange();
    else{
        _error = 0;
        for(unsigned int i=this->getNumberDataFitModel();i<data.size();i++){
            Vec2F64 src0= data(i)._src;
            Vec2F64 dst0= data(i)._dst;
            Vec2F64 dst0approx= GeometricalTransformation::transformHomogeneous2D(this->_geometricaltransformation,src0);
            double dist = (dst0-dst0approx).norm();
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
