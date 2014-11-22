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
    virtual void setWeight(const std::vector<double>v_weight)=0;
    virtual double getError()const = 0;
    virtual bool isGoodAffectation(unsigned int label)const=0;
    virtual unsigned int getSizeTrainingSet()const=0;
};

template<typename FeatureToScalar>
class ClassiferThresholdFromTrainingSet : public ClassifierTraining<typename FeatureToScalar::Feature>
{
    //sign + (sign=1) means if h(v)=1 for v in [threshold_min,threshold_max], 0 otherwise
    //sign - (sign=0) means if h(v)=1 for v notin [threshold_min,threshold_max], 0 otherwise

private:


    std::vector<double> _v_coefficient;
    std::vector<double> _v_coefficient_sort;
    std::vector<int> _v_affectation;
    std::vector<double> _v_weight;
    double _threshold_min;
    double _threshold_max;
    bool _sign;
    double _error;
    FeatureToScalar  _featuretovalue;

public:
    typedef typename FeatureToScalar::Feature  Feature;
    ClassiferThresholdFromTrainingSet();
    ClassiferThresholdFromTrainingSet(const ClassiferThresholdFromTrainingSet &classifier);


    virtual bool isGoodAffectation(unsigned int label)const;
    void setTraining(const std::vector<double>v_coefficient);
    void setAffectation(const std::vector<int>v_affectation);
    void setFeatureToScalar(const FeatureToScalar&featuretovalue);
    unsigned int getSizeTrainingSet()const;
    void setWeight(const std::vector<double>v_weight);
    double getError()const;
    void training();
    int operator()(const Feature& value);
    void display();
    void save(std::ostream& out)const;
    void load(std::istream& in);
};

template<typename ClassifierTraining>
class ClassiferAdaBoost:public Classifier<typename ClassifierTraining::Feature>
{

public:


    ClassiferAdaBoost(double threshold=0.4);
    void training( std::vector<ClassifierTraining>& classifier, int T);
    int operator()(const typename ClassifierTraining::Feature &feature);
    void save(std::ostream& out)const;
    void load(std::istream& in);
private:

    std::vector<ClassifierTraining> _weak_classifier;
    std::vector<double> _weigh_alpha;
    double _threshold;

};
template<typename FeatureToScalar>
ClassiferThresholdFromTrainingSet<FeatureToScalar>::ClassiferThresholdFromTrainingSet()
{
}
template<typename FeatureToScalar>
void ClassiferThresholdFromTrainingSet<FeatureToScalar>::setTraining(const std::vector<double>v_coefficient)
{
    _v_coefficient = v_coefficient;
    _v_coefficient_sort = v_coefficient;
    std::sort(_v_coefficient_sort.begin(),_v_coefficient_sort.end());
}
template<typename FeatureToScalar>
void ClassiferThresholdFromTrainingSet<FeatureToScalar>::setAffectation(const std::vector<int> v_affectation)
{
    _v_affectation = v_affectation;
}
template<typename FeatureToScalar>
void ClassiferThresholdFromTrainingSet<FeatureToScalar>::setWeight(const std::vector<double>v_weight)
{
    _v_weight = v_weight;
}
template<typename FeatureToScalar>
double ClassiferThresholdFromTrainingSet<FeatureToScalar>::getError()const{
    double error = 0;
    for(int k=0;k<(int)_v_coefficient.size();k++){
        double coeff_current=_v_coefficient[k];
        if(coeff_current<_threshold_min||coeff_current>_threshold_max){
            if(   (_v_affectation[k]==1&&_sign==true) || (_v_affectation[k]==0&&_sign==false)  )
                error+=_v_weight[k];
        }
        else{
            if(   (_v_affectation[k]==0&&_sign==true) || (_v_affectation[k]==1&&_sign==false)  )
                error+=_v_weight[k];
        }

    }

    return error;
}

template<typename FeatureToScalar>
int ClassiferThresholdFromTrainingSet<FeatureToScalar>::operator()(const typename FeatureToScalar::Feature& feature){
    double value = _featuretovalue(feature);
    if(value>_threshold_min&&value<_threshold_max){
        if(_sign==true)
            return 1;
        else
            return 0;
    }else{
        if(_sign==true)
            return 0;
        else
            return 1;
    }
}

template<typename FeatureToScalar>
bool ClassiferThresholdFromTrainingSet<FeatureToScalar>::isGoodAffectation(unsigned int label) const{
    double value = _v_coefficient[label];
    if(value>_threshold_min&&value<_threshold_max){
        if( (_sign==true&&_v_affectation[label]==1) || (_sign==false&&_v_affectation[label]==0) )
            return 1;
        else
            return 0;
    }else{
        if( (_sign==true&&_v_affectation[label]==0) || (_sign==false&&_v_affectation[label]==1) )
            return 1;
        else
            return 0;
    }
}

template<typename FeatureToScalar>
ClassiferThresholdFromTrainingSet<FeatureToScalar>::ClassiferThresholdFromTrainingSet(const ClassiferThresholdFromTrainingSet &classifier)
    :_v_coefficient(classifier._v_coefficient),
      _v_coefficient_sort(classifier._v_coefficient_sort),
      _v_affectation(classifier._v_affectation),
      _v_weight(classifier._v_weight),
      _threshold_min(classifier._threshold_min),
      _threshold_max(classifier._threshold_max),
      _sign(classifier._sign),
      _error(classifier._error),
      _featuretovalue(classifier._featuretovalue)
{

}
template<typename FeatureToScalar>
void ClassiferThresholdFromTrainingSet<FeatureToScalar>::display(){
    Mat2RGBUI8 img(512,50);
    double min = _v_coefficient_sort[0];
    double max = _v_coefficient_sort[_v_coefficient_sort.size()-1];
    for(unsigned int i =0;i<_v_coefficient.size();i++){
        Mat2RGBUI8::E x;
        x(1)=25;
        x(0)= (_v_coefficient[i]-min)*512./(max-min);
        RGBUI8 r;
        if( isGoodAffectation(i)==true)
            r.b()=255;
        else
            r.r()=255;
        Draw::circle(img,x,5,r,1);
    }
    if(_threshold_min>min){
        Mat2RGBUI8::E x;
        x(1)=25;
        x(0)=(_threshold_min-min)*512./(max-min);
        Draw::circle(img,x,5,RGBUI8(255),1);
    }
    if(_threshold_max<max){
        Mat2RGBUI8::E x;
        x(1)=25;
        x(0)=(_threshold_max-min)*512./(max-min);
        Draw::circle(img,x,5,RGBUI8(255),1);
    }


    img.display();
}
template<typename FeatureToScalar>
unsigned int ClassiferThresholdFromTrainingSet<FeatureToScalar>::getSizeTrainingSet()const{
    return _v_coefficient.size();
}
template<typename FeatureToScalar>
void ClassiferThresholdFromTrainingSet<FeatureToScalar>::setFeatureToScalar(const FeatureToScalar &featuretovalue)
{
    _featuretovalue = featuretovalue;
}
template<typename FeatureToScalar>
void ClassiferThresholdFromTrainingSet<FeatureToScalar>::training()
{
    _error = NumericLimits<double>::maximumRange();
    for(int i=0;i<(int)_v_coefficient_sort.size();i++){
        for(int j=i+1;j<(int)_v_coefficient_sort.size();j++){
            double error_temp=0;
            double coeff_min = _v_coefficient_sort[i];
            double coeff_max = _v_coefficient_sort[j];

            for(int k=0;k<(int)_v_coefficient.size();k++){
                double coeff_current=_v_coefficient[k];
                if(coeff_current<coeff_min||coeff_current>coeff_max){
                    if(_v_affectation[k]==1)
                        error_temp+=_v_weight[k];
                }
                else{
                    if(_v_affectation[k]==0)
                        error_temp+=_v_weight[k];
                }

            }
            if(error_temp<_error)
            {
                _error=error_temp;
                _sign = true;
                if(i==0)
                    _threshold_min=-NumericLimits<double>::maximumRange();
                else
                    _threshold_min= (_v_coefficient_sort[i]+_v_coefficient_sort[i-1])/2;
                if(j==(int)_v_coefficient.size()-1)
                    _threshold_max=NumericLimits<double>::maximumRange();
                else
                    _threshold_max= (_v_coefficient_sort[j]+_v_coefficient_sort[j+1])/2;
            }

        }
    }
    for(int i=0;i<(int)_v_coefficient_sort.size();i++){
        for(int j=i+1;j<(int)_v_coefficient_sort.size();j++){
            double error_temp=0;
            double coeff_min = _v_coefficient_sort[i];
            double coeff_max = _v_coefficient_sort[j];

            for(int k=0;k<(int)_v_coefficient.size();k++){
                double coeff_current=_v_coefficient[k];
                if(coeff_current<coeff_min||coeff_current>coeff_max){
                    if(_v_affectation[k]==0)
                        error_temp+=_v_weight[k];
                }
                else{
                    if(_v_affectation[k]==1)
                        error_temp+=_v_weight[k];
                }
            }
            if(error_temp<_error)
            {
                _error=error_temp;
                _sign = false;
                if(i==0)
                    _threshold_min=-NumericLimits<double>::maximumRange();
                else
                    _threshold_min= (_v_coefficient_sort[i]+_v_coefficient_sort[i-1])/2;
                if(j==(int)_v_coefficient.size()-1)
                    _threshold_max=NumericLimits<double>::maximumRange();
                else
                    _threshold_max= (_v_coefficient_sort[j]+_v_coefficient_sort[j+1])/2;
            }

        }
    }
}





template<typename ClassifierTraining>
ClassiferAdaBoost<ClassifierTraining>::ClassiferAdaBoost(double threshold)
    :_threshold(threshold)
{

}

template<typename ClassifierTraining>
void ClassiferAdaBoost<ClassifierTraining>::training( std::vector<ClassifierTraining>& classifier, int T){


    std::vector<double> elected(classifier.size(),0);
    std::vector<double> weigh(classifier[0].getSizeTrainingSet(),1./classifier[0].getSizeTrainingSet());
    for(int t=0;t<T;t++){
        std::cout<<"training "<<t<<std::endl;
        //normalisze the weight
        std::vector<double>::iterator it=weigh.begin();
        double sum=0;
        for(;it!=weigh.end();++it){
            sum+=*it;
        }
        for(it=weigh.begin();it!=weigh.end();++it){
            *it/=sum;
        }

        double error=NumericLimits<double>::maximumRange();
        unsigned int label_elected;
        // for each classifier
        for (unsigned int i = 0; i < classifier.size();i++){
            if(elected[i]==0){
                classifier[i].setWeight(weigh);
                double error_temp = classifier[i].getError();
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
    double __sum=0;
    for (unsigned int i = 0; i < _weigh_alpha.size();i++){
        __sum+=_weigh_alpha[i];
    }
    for (unsigned int i = 0; i < _weigh_alpha.size();i++){
        _weigh_alpha[i]/=__sum;
    }

}
template<typename ClassifierTraining>
int ClassiferAdaBoost<ClassifierTraining>::operator()(const typename ClassifierTraining::Feature &feature) {
    double val=0;
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
template<typename ClassifierTraining>
void ClassiferAdaBoost<ClassifierTraining>::save(std::ostream& out)const{
    out<<this->_threshold<<"<ADA>";
    out<<this->_weak_classifier<<"<ADA>";
    out<<this->_weigh_alpha<<"<ADA>";
}
template<typename ClassifierTraining>
void ClassiferAdaBoost<ClassifierTraining>::load(std::istream& in){
    std::string str;
    str = pop::BasicUtility::getline(in,"<ADA>");
    pop::BasicUtility::String2Any(str,_threshold);
    str = pop::BasicUtility::getline(in,"<ADA>");
    pop::BasicUtility::String2Any(str,_weak_classifier);
    str = pop::BasicUtility::getline(in,"<ADA>");
    pop::BasicUtility::String2Any(str,_weigh_alpha);
}


template<typename FeatureToScalar>
void ClassiferThresholdFromTrainingSet<FeatureToScalar>::save(std::ostream& out)const{
    out<<this->_threshold_min<<"<Train>";
    out<<this->_threshold_max<<"<Train>";
    out<<this->_sign<<"<Train>";
    out<<this->_featuretovalue<<"<Train>";
}
template<typename FeatureToScalar>
void ClassiferThresholdFromTrainingSet<FeatureToScalar>::load(std::istream& in){
    std::string str;
    str = pop::BasicUtility::getline(in,"<Train>");
    pop::BasicUtility::String2Any(str,_threshold_min);
    str = pop::BasicUtility::getline(in,"<Train>");
    pop::BasicUtility::String2Any(str,_threshold_max);
    str = pop::BasicUtility::getline(in,"<Train>");
    pop::BasicUtility::String2Any(str,_sign);
    str = pop::BasicUtility::getline(in,"<Train>");
    pop::BasicUtility::String2Any(str,_featuretovalue);
}


/*!
@}
*/
template<typename FeatureToScalar>
std::istream& operator >> (std::istream& in,  pop::ClassiferThresholdFromTrainingSet<FeatureToScalar>& classifier){
    classifier.load(in);
    return in;
}

template<typename FeatureToScalar>
std::ostream& operator << (std::ostream& out, const pop::ClassiferThresholdFromTrainingSet<FeatureToScalar>& classifier){
    classifier.save(out);
    return out;
}
template<typename ClassifierTraining>
std::istream& operator >> (std::istream& in,  pop:: ClassiferAdaBoost<ClassifierTraining>& classifier){
    classifier.load(in);
    return in;
}

template<typename ClassifierTraining>
std::ostream& operator << (std::ostream& out, const pop:: ClassiferAdaBoost<ClassifierTraining>& classifier){
    classifier.save(out);
    return out;
}
}

#endif // CLASSIFER_H
