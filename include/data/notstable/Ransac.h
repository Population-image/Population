#ifndef RANSAC_H
#define RANSAC_H
#include<vector>
#include<limits>
#include"data/vec/VecN.h"
#include"data/mat/MatN.h"
#include"data/distribution/Distribution.h"
#include"algorithm/LinearAlgebra.h"
namespace pop
{
/*!
 * \param data a set of observations
 * \param nbriteration the number of iterations performed by the algorithm
 * \param threshold a threshold value for determining when a data fits a model
 * \param minnbrdatatoassertmodel the number of close data values required to assert that a model fits well to data
 * \param best_model model parameters which best fit the data (or nil if no good model is found)
 * \param  best_consensus_set data points from which this model has been estimated
 *
 *
 * Generic progragramming Implementation of the RANSAC algorithm from wikipedia http://en.wikipedia.org/wiki/RANSAC.
 * The requirements of the template are :
 *  - a associated Data as the type of a observation (Model::Data) ,
 *  - a constructor based of these observations (Model(const Vec<Model::Data >& data))
 *  - a method F32 getError() the error of this model relative to these observations
 *  - a method F32 getError(const Data & p) the error of this model relative to this observation
 *  - a method unsigned int getNumberDataFitModel() the minimum number of data required to fit the model
 *
 * \code
//   four inliners  (x=(1,1), y=5.2), (x=(1,2), y=5.8), (x=(1,3), y=6.8), (x=(1,4), y=8.2), and one outliners (x=(1,5), y=13)
Vec<LinearLeastSquareRANSACModel::Data> data;
VecF32 x(2);F32 y;
x(0)=1;x(1)=1;y=5.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
x(0)=1;x(1)=2;y=5.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
x(0)=1;x(1)=3;y=6.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
x(0)=1;x(1)=4;y=8.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
x(0)=1;x(1)=5;y=13; data.push_back(LinearLeastSquareRANSACModel::Data(x,y));

LinearLeastSquareRANSACModel m;
Vec<LinearLeastSquareRANSACModel::Data> dataconsencus;
ransac(data,2,10,2,1,m,dataconsencus);
std::cout<<m.getBeta()<<std::endl;
std::cout<<m.getError()<<std::endl;
  * \endcode
  */

template<typename Model>
POP_EXPORTS void ransac(Vec<typename Model::Data> & data,unsigned int nbriteration,F32 threshold,unsigned int minnbrdatatoassertmodel,Model &best_model,Vec<typename Model::Data>&best_consensus_set){
    typedef typename Model::Data DataModel;
    Distribution d;
    for(unsigned int iterations=0;iterations<nbriteration;iterations++){
        std::random_shuffle ( data.begin(), data.end() ,d.MTRand());
        Vec<DataModel> consensus_set (data.begin(),data.begin()+best_model.getNumberDataFitModel());
        Model maybe_model(consensus_set);

        for(unsigned int i =best_model.getNumberDataFitModel();i<data.size();i++){
            if(maybe_model.getError(data[i])<threshold){
                consensus_set.push_back(data[i]);
            }
        }
        if(consensus_set.size()>=best_model.getNumberDataFitModel()+minnbrdatatoassertmodel){
            Model this_model(consensus_set);
            F32 this_error =this_model.getError();
            if(this_error<best_model.getError()){
                best_model = this_model;
                best_consensus_set = consensus_set;
            }
        }
    }
}

template<typename Model>
POP_EXPORTS void ransacMaxDataFitModel(Vec<typename Model::Data> & data,unsigned int nbriteration,F32 threshold,Model &best_model,Vec<typename Model::Data>&best_consensus_set){
    typedef typename Model::Data DataModel;
    unsigned int minnbrdatatoassertmodel =1;
    Distribution d;
    for(unsigned int iterations=0;iterations<nbriteration;iterations++){
        std::random_shuffle ( data.begin(), data.end() , d.MTRand());
        Vec<DataModel> consensus_set (data.begin(),data.begin()+best_model.getNumberDataFitModel());
        Model maybe_model(consensus_set);

        for(unsigned int i =best_model.getNumberDataFitModel();i<data.size();i++){
            if(maybe_model.getError(data[i])<threshold){
                consensus_set.push_back(data[i]);
            }
        }
        if(consensus_set.size()>=best_model.getNumberDataFitModel()+minnbrdatatoassertmodel){

            Model this_model(consensus_set);
            F32 this_error =this_model.getError();
            if(consensus_set.size()>best_model.getNumberDataFitModel()+minnbrdatatoassertmodel){
                best_model = this_model;
                best_consensus_set = consensus_set;
                minnbrdatatoassertmodel = consensus_set.size()-best_model.getNumberDataFitModel();
            }
            else if(this_error<best_model.getError()){
                best_model = this_model;
                best_consensus_set = consensus_set;
            }

        }
    }
}

class POP_EXPORTS LinearLeastSquareRANSACModel
{
private:
    pop::VecF32 _beta;
    F32 _error;
public:
    struct POP_EXPORTS Data {
        Data();
        Data(pop::VecF32 x,pop::F32 y);
        pop::VecF32 X;
        pop::F32 Y;
    };
    LinearLeastSquareRANSACModel(const Vec<Data >& data);
    F32 getError();
    F32 getError(const Data & p);
     unsigned int getNumberDataFitModel()const;
    pop::VecF32 getBeta();
};


class POP_EXPORTS GeometricalTransformationRANSACModel
{
protected:
    pop::Mat2x33F32 _geometricaltransformation;
    F32 _error;
public:
    struct POP_EXPORTS Data
    {
        Data();
        Data(pop::Vec2F32 src,pop::Vec2F32 dst);
        pop::Vec2F32 _src;
        pop::Vec2F32 _dst;
    };
    virtual ~GeometricalTransformationRANSACModel();
    GeometricalTransformationRANSACModel();
    F32 getError();
    F32 getError(const Data & p);
    pop::Mat2x33F32 getTransformation();

    //virtual GeometricalTransformationRANSACModel(const Vec<Data >& data)=0;
    virtual  unsigned int getNumberDataFitModel()const =0 ;
};


class POP_EXPORTS AffineTransformationRANSACModel : public GeometricalTransformationRANSACModel
{

public:
    typedef GeometricalTransformationRANSACModel::Data  Data;
    AffineTransformationRANSACModel();
    AffineTransformationRANSACModel(const Vec<Data >& data);
     unsigned int getNumberDataFitModel()const;
};
class POP_EXPORTS ProjectionTransformationRANSACModel : public GeometricalTransformationRANSACModel
{

public:
    typedef GeometricalTransformationRANSACModel::Data  Data;
    ProjectionTransformationRANSACModel();
    ProjectionTransformationRANSACModel(const Vec<Data >& data);
     unsigned int getNumberDataFitModel()const;
};



}
#endif // RANSAC_H
