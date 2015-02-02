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


template<typename FeatureToScalar>
class ClassiferThresholdWeak
{
private:
    struct CoefficentSort
    {
       F32 _coefficient;
       int _label;
       inline bool operator < (const CoefficentSort & b)const{
           if(this->_coefficient<b._coefficient)return true;
           else return false;
       }
    };
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

    bool isGoodAffectation(unsigned int label)const{
        if(label>=_v_affectation.size()||label>=_v_coefficient.size()){
            std::cout<<"error"<<std::endl;
        }
        if(scalar2Class(_v_coefficient[label])==_v_affectation(label))
            return true;
        else
            return false;
    }
    void setTraining(const Vec<F32> v_coefficient,const Vec<bool> v_affectation){
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
    const FeatureToScalar& getFeatureToScalar()const{
        return _featuretovalue;
    }

    void setWeight(const Vec<F32>v_weight){
        _v_weight = v_weight;
    }

    F32 getError()const{
        return _error;
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

    bool operator()(const Feature& value){
        return scalar2Class(_featuretovalue.operator ()(value));
    }
    bool scalar2Class(F32 value)const{
        bool hit=(value<=_threshold);
        if(_sign==true)
            return hit;
        else
            return !hit;
    }
    unsigned int getSizeTrainingSet()const{
        return _v_affectation.size();
    }
};

template<typename ClassifierTraining>
class ClassiferAdaBoost
{

public:
   typename ClassifierTraining::Feature value;

    ClassiferAdaBoost(F32 threshold=0);
    void training( Vec<ClassifierTraining>& classifier, int T);
    bool operator()(const typename ClassifierTraining::Feature &feature);
private:

    Vec<ClassifierTraining> _weak_classifier;
    Vec<F32> _weigh_alpha;
    F32 _threshold;

};

template<typename ClassifierTraining>
ClassiferAdaBoost<ClassifierTraining>::ClassiferAdaBoost(F32 threshold)
    :_threshold(threshold)
{

}

template<typename ClassifierTraining>
void ClassiferAdaBoost<ClassifierTraining>::training( Vec<ClassifierTraining>& classifier, int T){


    Vec<F32> elected(classifier.size(),0);
    Vec<F32> weigh(classifier[0].getSizeTrainingSet(),1.f/classifier[0].getSizeTrainingSet());
    for(int t=0;t<T;t++){
        std::cout<<"training "<<t<<std::endl;
        //normalisze the weight

        F32 sum=std::accumulate(weigh.begin(),weigh.end(),0.f);
        weigh=weigh*(1/sum);

        F32 error=NumericLimits<F32>::maximumRange();
        unsigned int label_elected;
        // for each classifier
        for (unsigned int i = 0; i < classifier.size();i++){
            if(elected[i]==0){
                classifier[i].setWeight(weigh);
                classifier[i].training();
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
bool ClassiferAdaBoost<ClassifierTraining>::operator()(const typename ClassifierTraining::Feature &feature) {
    F32 val=0;
    for (unsigned int i=0;i < _weak_classifier.size();  i++){
        if(_weak_classifier[i](feature)==true)
            val+=_weigh_alpha[i];
        else
            val-=_weigh_alpha[i];
    }
    if (val>_threshold)
        return true;  // label +1
    else
        return false; // label 0
}
}

#endif // CLASSIFER_H
