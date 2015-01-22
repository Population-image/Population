#ifndef CLASSIFER_H
#define CLASSIFER_H

#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include "data/mat/MatN.h"
#include"data/mat/Mat2x.h"

#include "algorithm/Draw.h"
namespace pop{

/*! \ingroup Data
* \defgroup Classifier Classifier
* @{
*/
// A classifier
//    abstract class
template <typename FeatureType>
class Classifier {
    /*!
        \class pop::Classifier
        \brief  Interface for the labelling classification
        \author Tariel Vincent

        From an object feature, we can identify to which of the labelling it belongs
    */
public:
    typedef FeatureType  Feature;
    /*!
     \brief operator retruning the label of the feature
     \param feature feature object
     */
    virtual int operator()(const  FeatureType& feature) = 0;
};

template <typename FeatureType>
class ClassifierTraining : public Classifier<FeatureType>
{

public:
    typedef FeatureType  Feature;
    virtual void  setWeight(const Vec<F32>v_weight)=0;
//    virtual F32 getError()const = 0;
//    virtual bool isGoodAffectation(unsigned int label)const=0;
//    virtual unsigned int getSizeTrainingSet()const=0;
};
struct CoefficentSort
{
   F32 _coefficient;
   int _label;

};
inline bool operator < (const CoefficentSort & a,const CoefficentSort & b){
    if(a._coefficient<b._coefficient)return true;
    else return false;
}
template<typename FeatureToScalar>
class ClassiferThreshold : public ClassifierTraining<typename FeatureToScalar::Feature>
{
public:

    Vec<CoefficentSort> _v_coefficient_sort;
    Vec<F32> _v_coefficient;
    Vec<int> _v_affectation;
    Vec<F32> _v_weight;
    F32 _threshold;
    bool _sign;
    F32 _error;
    FeatureToScalar  _featuretovalue;

public:
    typedef typename FeatureToScalar::Feature  Feature;

    virtual bool isGoodAffectation(unsigned int label)const{
        return operator ()(_v_coefficient[label]);
    }
    void setTraining(const Vec<F32> v_coefficient,const Vec<int> v_affectation){
        _v_coefficient = v_coefficient;
        _v_affectation = v_affectation;
        _v_coefficient_sort.clear();
        for(unsigned int i=0;i<_v_coefficient.size();i++){
            CoefficentSort coeff;
            coeff._label=i;
            coeff._coefficient = _v_coefficient[i];
            _v_coefficient_sort.push_back(coeff);
        }
        std::sort(_v_coefficient_sort.begin(),_v_coefficient_sort.end());
    }
    void setFeatureToScalar(const FeatureToScalar&featuretovalue){
        _featuretovalue = featuretovalue;
    }

    void setWeight(const Vec<F32>v_weight){
        _v_weight = v_weight;
    }

    F32 getError()const{
        F32 error=0;
        for(unsigned int i=0;i<_v_coefficient.size();i++){
            if(isGoodAffectation(i)==false)
                error +=_v_weight[i];
        }
    }
    void training(){
        F32 error_sign_true=NumericLimits<F32>::maximumRange();
        F32 error_sign_false=NumericLimits<F32>::maximumRange();
        int     index_sign_false = -1;
        int     index_sign_true= -1;
        F32 error_sign_false_current=0;
        F32 error_sign_true_current=0;
        _sign =true;
        for(unsigned int i=0;i<_v_weight.size();i++){
            double weigh = _v_weight[i];
            int affect = _v_affectation[i];
            if(affect!=0){
                error_sign_true_current+=weigh;
            }else{
                error_sign_false_current+=weigh;
            }
        }
        error_sign_true = error_sign_true_current;
        error_sign_false= error_sign_false_current;
        for(unsigned int i=0;i<_v_coefficient_sort.size();i++){
            int index_current        = _v_coefficient_sort[i]._label;
            double weigh_current = _v_weight[index_current];
            int affect_current = _v_affectation[index_current];
            if(affect_current!=0){
                error_sign_true_current-=weigh_current;
                error_sign_false_current+=weigh_current;
            }else{
                error_sign_false_current-=weigh_current;
                error_sign_true_current+=weigh_current;
            }
            if(error_sign_true_current<error_sign_true){
                index_sign_true = i;
                error_sign_true = error_sign_true_current;
            }
            if(error_sign_false_current<error_sign_false){
                index_sign_false = i;
                error_sign_false = error_sign_false_current;
            }
        }
        if(error_sign_true<error_sign_false){
            _sign = true;
            _error= error_sign_true;
            if(index_sign_true==-1)
                 _threshold= NumericLimits<F32>::minimumRange();
            else if(index_sign_true==_v_coefficient_sort.size()-1)
                 _threshold= NumericLimits<F32>::maximumRange();
            else
                _threshold=  (_v_coefficient_sort[index_sign_true]._coefficient+
                        _v_coefficient_sort[index_sign_true+1]._coefficient)/2.;
        }else{
            _sign = false;
            _error= error_sign_false;
            if(index_sign_false==-1)
                 _threshold= NumericLimits<F32>::minimumRange();
            else if(index_sign_false==_v_coefficient_sort.size()-1)
                 _threshold= NumericLimits<F32>::maximumRange();
            else
                _threshold=  (_v_coefficient_sort[index_sign_false]._coefficient+
                        _v_coefficient_sort[index_sign_false+1]._coefficient)/2.;
        }
    }

    int operator()(const Feature& value){
        return operator ()(_featuretovalue.operator ()(value));
    }
    int operator()(F32 value)const{
        bool hit=(value<=_threshold);
        if(_sign==true)
            return hit;
        else
            return !hit;
    }
};

template<typename ClassifierTraining>
class ClassiferAdaBoost:public Classifier<typename ClassifierTraining::Feature>
{

public:


    ClassiferAdaBoost(F32 threshold=0.4);
    void training( std::vector<ClassifierTraining>& classifier, int T);
    int operator()(const typename ClassifierTraining::Feature &feature);
private:

    std::vector<ClassifierTraining> _weak_classifier;
    std::vector<F32> _weigh_alpha;
    F32 _threshold;

};

template<typename ClassifierTraining>
ClassiferAdaBoost<ClassifierTraining>::ClassiferAdaBoost(F32 threshold)
    :_threshold(threshold)
{

}

template<typename ClassifierTraining>
void ClassiferAdaBoost<ClassifierTraining>::training( std::vector<ClassifierTraining>& classifier, int T){


    Vec<F32> elected(classifier.size(),0);
    Vec<F32> weigh(classifier[0].getSizeTrainingSet(),1./classifier[0].getSizeTrainingSet());
    for(int t=0;t<T;t++){
        std::cout<<"training "<<t<<std::endl;
        //normalisze the weight

        F32 sum=std::accumulate(weigh.begin(),weigh.end(),0);
        weigh=weigh*(1/sum);

        F32 error=NumericLimits<F32>::maximumRange();
        unsigned int label_elected;
        // for each classifier
        for (unsigned int i = 0; i < classifier.size();i++){
            if(elected[i]==0){
                classifier[i].setWeight(weigh);
                F32 error_temp = classifier[i].getError();
                if(error_temp<error){
                    label_elected = i;
                    error = error_temp;
                }
            }
        }
        if (error >= 0.5)
            break;
        elected[label_elected] =std::log((1.0 - error)/error)/2.;
        _weak_classifier.push_back(classifier[label_elected]);
        _weigh_alpha.push_back(elected[label_elected] );
        for (unsigned int i=0; i < weigh.size(); i++){
            if(classifier[label_elected].isGoodAffectation(i)==true)
                weigh[i] *=std::exp(-elected[label_elected]);
            else{
                weigh[i] *=std::exp(elected[label_elected]);
            }
        }


    }
    F32 __sum=0;
    for (unsigned int i = 0; i < _weigh_alpha.size();i++){
        __sum+=_weigh_alpha[i];
    }
    for (unsigned int i = 0; i < _weigh_alpha.size();i++){
        _weigh_alpha[i]/=__sum;
    }

}
template<typename ClassifierTraining>
int ClassiferAdaBoost<ClassifierTraining>::operator()(const typename ClassifierTraining::Feature &feature) {
    F32 val=0;
    for (unsigned int i=0;i < _weak_classifier.size();  i++){
        if(_weak_classifier[i](feature)==1)
            val+=_weigh_alpha[i];
        else
            val-=_weigh_alpha[i];
    }
    if (val>_threshold)
        return 1;  // label +1
    else
        return 0; // label 0
}


/*!
@}
*/

}

#endif // CLASSIFER_H
