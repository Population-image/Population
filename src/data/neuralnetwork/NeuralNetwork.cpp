#include "data/neuralnetwork/NeuralNetwork.h"
#include "data/distribution/DistributionAnalytic.h"
#include "data/mat/MatN.h"
#include "data/mat/MatNInOut.h"
#include "data/mat/MatNDisplay.h"
#include "PopulationConfig.h"
#include "algorithm/Arithmetic.h"
namespace pop {


//Vec<Vec<Mat2UI8> > TrainingNeuralNetwork::loadMNIST( std::string datapath,  std::string labelpath){
//    Vec<Vec<Mat2UI8> > dataset(10);
//    std::ifstream datas(datapath.c_str(),std::ios::binary);
//    std::ifstream labels(labelpath.c_str(),std::ios::binary);

//    if (!datas.is_open() || !labels.is_open()){
//        std::cerr<<"binary files could not be loaded" << std::endl;
//        return dataset;
//    }

//    int magic_number=0; int number_of_images=0;int r; int c;
//    int n_rows=0; int n_cols=0; unsigned char temp=0;

//    // parse data header
//    datas.read((char*)&magic_number,sizeof(magic_number));
//    magic_number=_reverseInt(magic_number);
//    datas.read((char*)&number_of_images,sizeof(number_of_images));
//    number_of_images=_reverseInt(number_of_images);
//    datas.read((char*)&n_rows,sizeof(n_rows));
//    n_rows=_reverseInt(n_rows);
//    datas.read((char*)&n_cols,sizeof(n_cols));
//    n_cols=_reverseInt(n_cols);

//    // parse label header - ignore
//    int dummy;
//    labels.read((char*)&dummy,sizeof(dummy));
//    labels.read((char*)&dummy,sizeof(dummy));

//    for(int i=0;i<number_of_images;++i){
//        pop::Mat2UI8 img(n_rows,n_cols);

//        for(r=0;r<n_rows;++r){
//            for(c=0;c<n_cols;++c){
//                datas.read((char*)&temp,sizeof(temp));
//                img(r,c) = temp;
//            }
//        }
//        labels.read((char*)&temp,sizeof(temp));
//        dataset[(int)temp].push_back(img);
//    }
//    return dataset;
//}

//Vec<pop::Mat2UI8> TrainingNeuralNetwork::geometricalTransformationDataBaseMatrix( Vec<pop::Mat2UI8>  number_training,
//                                                                                  unsigned int number,
//                                                                                  F32 sigma_elastic_distortion_min,
//                                                                                  F32 sigma_elastic_distortion_max,
//                                                                                  F32 alpha_elastic_distortion_min,
//                                                                                  F32 alpha_elastic_distortion_max,
//                                                                                  F32 beta_angle_degree_rotation,
//                                                                                  F32 beta_angle_degree_shear,
//                                                                                  F32 gamma_x_scale,
//                                                                                  F32 gamma_y_scale){

//    DistributionUniformReal dAngle(-beta_angle_degree_rotation*pop::PI/180,beta_angle_degree_rotation*pop::PI/180);
//    DistributionUniformReal dShear(-beta_angle_degree_shear*pop::PI/180,beta_angle_degree_shear*pop::PI/180);

//    DistributionUniformReal d_deviation_length(sigma_elastic_distortion_min,sigma_elastic_distortion_max);
//    DistributionUniformReal d_correlation_lenght(alpha_elastic_distortion_min,alpha_elastic_distortion_max);

//    DistributionUniformReal d_scale_x(1-gamma_x_scale/100,1+gamma_x_scale/100);
//    DistributionUniformReal d_scale_y(1-gamma_y_scale/100,1+gamma_y_scale/100);


//    Vec<pop::Mat2UI8> v_out_i;
//    for(unsigned int j=0;j<number_training.size();j++){

//        Mat2UI8 binary = number_training(j);
//        v_out_i.push_back(binary);
//        Draw::addBorder(binary,2, UI8(0));


//        Mat2UI8 binary_scale =  binary;
//        for(unsigned int k=0;k<number;k++){
//            F32 deviation_length_random = d_deviation_length.randomVariable();
//            F32 correlation_lenght_random =d_correlation_lenght.randomVariable();
//            Mat2UI8 m= GeometricalTransformation::elasticDeformation(binary_scale,deviation_length_random,correlation_lenght_random);
//            F32 angle = dAngle.randomVariable();
//            F32 shear = dShear.randomVariable();

//            F32 alphax=d_scale_x.randomVariable();
//            F32 alphay=d_scale_y.randomVariable();

//            Vec2F32 v(alphax,alphay);
//            //                std::cout<<"scale "<<v<<std::endl;
//            //                std::cout<<"angle "<<angle<<std::endl;
//            //                std::cout<<"shear "<<shear<<std::endl;
//            Mat2x33F32 maffine  = GeometricalTransformation::translation2DHomogeneousCoordinate(m.getDomain()/2);//go back to the buttom left corner (origin)
//            maffine *=  GeometricalTransformation::scale2DHomogeneousCoordinate(v);
//            maffine *=  GeometricalTransformation::shear2DHomogeneousCoordinate(shear,0);
//            maffine *=  GeometricalTransformation::rotation2DHomogeneousCoordinate(angle);//rotate
//            maffine *=  GeometricalTransformation::translation2DHomogeneousCoordinate(-m.getDomain()/2);
//            m = GeometricalTransformation::transformHomogeneous2D(maffine, m);
//            //                F32 sum2=0;
//            //                ForEachDomain2D(x,m){
//            //                    sum2+=m(x);
//            //                }
//            //                std::cout<<sum2/sum<<std::endl;
//            //             m.display();
//            v_out_i.push_back(m);
//        }
//    }
//    return v_out_i;
//}
//void TrainingNeuralNetwork::convertMatrixToInputValueNeuron(Vec<VecF32> &v_neuron_in, Vec<VecF32> &v_neuron_out,const Vec<Vec<pop::Mat2UI8> >& number_training,Vec2I32 domain ,NNLayerMatrix::CenteringMethod method,NNLayerMatrix::NormalizationValue normalization_value){



//    for(unsigned int i=0;i<number_training.size();i++){
//        for(unsigned int j=0;j<number_training(i).size();j++){
//            Mat2UI8 binary = number_training(i)(j);

//            VecF32 vin = NNLayerMatrix::inputMatrixToInputNeuron(binary,domain,method,normalization_value);
//            v_neuron_in.push_back(vin);
//            VecF32 v_out(static_cast<int>(number_training.size()),-1);
//            v_out(i)=1;
//            v_neuron_out.push_back(v_out);

//        }
//    }
//}

void NeuralLayer::setLearnableParameter(F32 mu){
    _mu = mu;
}
NeuralLayerLinear::NeuralLayerLinear(unsigned int nbr_neurons)
    :__Y(nbr_neurons),__X(nbr_neurons)
{
}
VecF32& NeuralLayerLinear::X(){return __X;}
const VecF32& NeuralLayerLinear::X()const{return __X;}
VecF32& NeuralLayerLinear::d_E_X(){return _d_E_X;}
void NeuralLayerLinear::setTrainable(bool istrainable){
    if(istrainable==true){
        this->_d_E_Y = this->__X;
        this->_d_E_X = this->__X;
    }else{
        this->_d_E_Y.clear();
        this->_d_E_X.clear();
    }
}
NeuralLayerMatrix::NeuralLayerMatrix(unsigned int sizei,unsigned int sizej,unsigned int nbr_map)
    :NeuralLayerLinear(sizei* sizej*nbr_map)
{
    for(unsigned int i=0;i<nbr_map;i++){
        MatNReference<2,F32> m(Vec2I32(sizei, sizej),this->__Y.data()+sizei*sizej*i);
        _Y_reference.push_back(MatNReference<2,F32>(Vec2I32(sizei, sizej),this->__Y.data()+sizei*sizej*i));
        _X_reference.push_back(MatNReference<2,F32>(Vec2I32(sizei, sizej),this->__X.data()+sizei*sizej*i));

    }
}
const Vec<MatNReference<2,F32> > & NeuralLayerMatrix::X_map()const{return _X_reference;}
Vec<MatNReference<2,F32> >& NeuralLayerMatrix::X_map(){return _X_reference;}



const Vec<MatNReference<2,F32> > & NeuralLayerMatrix::d_E_X_map()const{return _d_E_X_reference;}
Vec<MatNReference<2,F32> >& NeuralLayerMatrix::d_E_X_map(){return _d_E_X_reference;}


void NeuralLayerMatrix::setTrainable(bool istrainable){
    NeuralLayerLinear::setTrainable(istrainable);
    if(istrainable==true){
        for(unsigned int i=0;i<_X_reference.size();i++){
            _d_E_Y_reference.push_back(MatNReference<2,F32>(_X_reference(0).getDomain(),_d_E_Y.data()+_X_reference(0).getDomain().multCoordinate()*i));
            _d_E_X_reference.push_back(MatNReference<2,F32>(_X_reference(0).getDomain(),_d_E_X.data()+_X_reference(0).getDomain().multCoordinate()*i));
        }
    }else{
        this->_d_E_Y_reference.clear();
        this->_d_E_X_reference.clear();
    }
}
NeuralLayerLinearFullyConnected::NeuralLayerLinearFullyConnected(unsigned int nbr_neurons_previous,unsigned int nbr_neurons)
    :NeuralLayerLinear(nbr_neurons),_W(nbr_neurons,nbr_neurons_previous+1),_X_biais(nbr_neurons_previous+1,1)
{
    //normalize tbe number inverse square root of the connection feeding into the nodes)
    DistributionNormal n(0,1.f/std::sqrt(nbr_neurons_previous+1.f));
    for(unsigned int i=0;i<_W.size();i++){
        _W(i)=n.randomVariable();
    }
}
void NeuralLayerLinearFullyConnected::setTrainable(bool istrainable){
    NeuralLayerLinear::setTrainable(istrainable);
    if(istrainable==true){
        this->_d_E_W = this->_W;
    }else{
        this->_d_E_W.clear();
    }
}

void NeuralLayerLinearFullyConnected::forwardCPU(const NeuralLayer& layer_previous){
    std::copy(layer_previous.X().begin(),layer_previous.X().end(),this->_X_biais.begin());
    this->__Y = this->_W * this->_X_biais;
    for(unsigned int i=0;i<__Y.size();i++){
        this->__X(i) = NeuronSigmoid::activation(this->__Y(i));
    }

}
void NeuralLayerLinearFullyConnected::backwardCPU(NeuralLayer& layer_previous){

    VecF32& d_E_X_previous= layer_previous.d_E_X();
    for(unsigned int i=0;i<this->__Y.size();i++){
        //        if(NeuronSigmoid::derivedActivation(this->__X(i))>1.3){
        //            std::cerr<<"error "<<std::endl;
        //        }

        this->_d_E_Y(i) = this->_d_E_X(i)*NeuronSigmoid::derivedActivation(this->__X(i));
    }

    //TODO ADD THE ERROR
    for(unsigned int i=0;i<this->_W.sizeI();i++){
        for(unsigned int j=0;j<this->_W.sizeJ();j++){
            this->_d_E_W(i,j)=this->_X_biais(j)*this->_d_E_Y(i);
        }
    }
    for(unsigned int j=0;j<d_E_X_previous.size();j++){
        d_E_X_previous(j)=0;
        for(unsigned int i=0;i<this->_W.sizeI();i++){
            d_E_X_previous(j)+=this->_d_E_Y(i)*this->_W(i,j);
        }
    }
}
void NeuralLayerLinearFullyConnected::learn(F32 lambda_regulation){
    for(unsigned int i=0;i<this->_W.sizeI();i++){
        for(unsigned int j=0;j<this->_W.sizeJ();j++){
            this->_W(i,j)= lambda_regulation*this->_W(i,j) -  this->_mu*this->_d_E_W(i,j);
        }
    }
}
NeuralLayer * NeuralLayerLinearFullyConnected::clone(){
    return new   NeuralLayerLinearFullyConnected(*this);
}

NeuralLayerMatrixMaxPool::NeuralLayerMatrixMaxPool(unsigned int sub_scaling_factor,unsigned int sizei_map_previous,unsigned int sizej_map_previous,unsigned int nbr_map_previous)
    :NeuralLayerMatrix(std::floor (  sizei_map_previous/(1.f*sub_scaling_factor)),std::floor ( sizej_map_previous/(1.f*sub_scaling_factor)),nbr_map_previous),
      _sub_resolution_factor (sub_scaling_factor),
      _istrainable(false)
{

}

void NeuralLayerMatrixMaxPool::setTrainable(bool istrainable){
    NeuralLayerMatrix::setTrainable(istrainable);
    _istrainable = istrainable;
}

void NeuralLayerMatrixMaxPool::forwardCPU(const NeuralLayer& layer_previous){
    if(const NeuralLayerMatrix * neural_matrix = dynamic_cast<const NeuralLayerMatrix *>(&layer_previous)){
        if(_istrainable==false){
            for(unsigned index_map=0;index_map<this->X_map().size();index_map++){
                MatNReference<2,F32> & map_layer = this->X_map()(index_map);
                const MatNReference<2,F32> & map_layer_previous = neural_matrix->X_map()(index_map);
                for(unsigned int i=0;i<map_layer.sizeI();i++){
                    for(unsigned int j=0;j<map_layer.sizeJ();j++){
                        F32 value =-2;
                        for(unsigned i_r=0;i_r<_sub_resolution_factor;i_r++){
                            for(unsigned j_r=0;j_r<_sub_resolution_factor;j_r++){
                                value = std::max(value,map_layer_previous(i*_sub_resolution_factor+i_r,j*_sub_resolution_factor+j_r));
                            }
                        }
                        map_layer(i,j)=value;
                    }
                }
            }
        }else{
            for(unsigned index_map=0;index_map<this->X_map().size();index_map++){
                MatNReference<2,F32> & map_layer = this->X_map()(index_map);
                const MatNReference<2,F32> & map_layer_previous = neural_matrix->X_map()(index_map);
                for(unsigned int i=0;i<map_layer.sizeI();i++){
                    for(unsigned int j=0;j<map_layer.sizeJ();j++){
                        F32 value =-2;
                        for(unsigned i_r=0;i_r<_sub_resolution_factor;i_r++){
                            for(unsigned j_r=0;j_r<_sub_resolution_factor;j_r++){
                                if(value<map_layer_previous(i*_sub_resolution_factor+i_r,j*_sub_resolution_factor+j_r)){
                                    value = map_layer_previous(i*_sub_resolution_factor+i_r,j*_sub_resolution_factor+j_r);
                                    this->_Y_reference(index_map)(i,j)=i_r*_sub_resolution_factor+j_r;
                                }
                            }
                        }
                        map_layer(i,j)=value;
                    }
                }
            }
        }
    }
}

void NeuralLayerMatrixMaxPool::backwardCPU(NeuralLayer& layer_previous){
    if( NeuralLayerMatrix * neural_matrix = dynamic_cast< NeuralLayerMatrix *>(&layer_previous)){
        for(unsigned index_map=0;index_map<this->d_E_X_map().size();index_map++){
            const MatNReference<2,F32> & map_layer = this->d_E_X_map()(index_map);
            MatNReference<2,F32> & map_layer_previous = neural_matrix->d_E_X_map()(index_map);
            map_layer_previous.fill(0);
            for(unsigned int i=0;i<map_layer.sizeI();i++){
                for(unsigned int j=0;j<map_layer.sizeJ();j++){
                    int index =  this->_Y_reference(index_map)(i,j);
                    int i_r,j_r;
                    pop::Arithmetic::euclideanDivision(index,(int)_sub_resolution_factor,i_r,j_r);
                    map_layer_previous(i*_sub_resolution_factor+i_r,j*_sub_resolution_factor+j_r)= map_layer(i,j);

                }
            }
        }
    }
}

void NeuralLayerMatrixMaxPool::learn(F32 ){

}
NeuralLayer * NeuralLayerMatrixMaxPool::clone(){
    return new   NeuralLayerMatrixMaxPool(*this);
}



NeuralLayerMatrixConvolutionSubScaling::NeuralLayerMatrixConvolutionSubScaling(unsigned int nbr_map,unsigned int sub_scaling_factor,unsigned int radius_kernel,unsigned int sizei_map_previous,unsigned int sizej_map_previous,unsigned int nbr_map_previous)
    :NeuralLayerMatrix(std::floor (  (sizei_map_previous-1-2*radius_kernel)/(1.*sub_scaling_factor))+1,std::floor (  (sizej_map_previous-1-2*radius_kernel)/(1.*sub_scaling_factor))+1,nbr_map),
      _W_kernels(nbr_map*nbr_map_previous,Mat2F32(radius_kernel*2+1,radius_kernel*2+1)),
      _W_biais(nbr_map*nbr_map_previous),
      _sub_resolution_factor (sub_scaling_factor),
      _radius_kernel (radius_kernel)
{
    //std::cout<<(sizei_map_previous-1-2*radius_kernel)/(1.*sub_scaling_factor)+1<<std::endl;
    //normalize tbe number inverse square root of the connection feeding into the nodes)
    DistributionNormal n(0,0.001*1.f/((radius_kernel*2+1)*std::sqrt(nbr_map_previous*1.)));
    for(unsigned int i = 0;i<_W_kernels.size();i++){
        for(unsigned int j = 0;j<_W_kernels(i).size();j++){
            _W_kernels(i)(j)=n.randomVariable();
        }
        _W_biais(i)=n.randomVariable();
    }
}
void NeuralLayerMatrixConvolutionSubScaling::setTrainable(bool istrainable){
    NeuralLayerMatrix::setTrainable(istrainable);
    if(istrainable==true){
        _d_E_W_kernels = _W_kernels;
        _d_E_W_biais   = _W_biais;
    }else{
        _d_E_W_kernels.clear();
        _d_E_W_biais.clear();
    }
    for(unsigned int i=0;i<this->_d_E_W_kernels.size();i++){
        for(unsigned int j=0;j<this->_d_E_W_kernels(i).size();j++){
            this->_d_E_W_kernels(i)(j)=0;
        }
    }
    for(unsigned int i=0;i<this->_d_E_W_biais.size();i++){
        this->_d_E_W_biais(i)=0;
    }
}
void NeuralLayerMatrixConvolutionSubScaling::forwardCPU(const NeuralLayer& layer_previous){

    if(const NeuralLayerMatrix * neural_matrix = dynamic_cast<const NeuralLayerMatrix *>(&layer_previous)){

#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
        for( int index_map=0;index_map<static_cast<int>(this->_Y_reference.size());index_map++){
            MatNReference<2,F32> &map_out =  this->_Y_reference[index_map];
            int index_start_kernel = index_map*neural_matrix->X_map().size();
            for(unsigned int i_map_next=0,i_map_previous=_radius_kernel;i_map_next<map_out.sizeI();i_map_next++,i_map_previous+=_sub_resolution_factor){
                for(unsigned int j_map_next=0,j_map_previous=_radius_kernel;j_map_next<map_out.sizeJ();j_map_next++,j_map_previous+=_sub_resolution_factor){
                    F32 sum=0;
                    //convolution
                    for(unsigned int index_map_previous=0;index_map_previous<neural_matrix->X_map().size();index_map_previous++){
                        sum+=_W_biais[ index_map_previous + index_start_kernel];
                        for(unsigned int i=0,index_kernel_ij=0,index_map = (i_map_previous-_radius_kernel)*neural_matrix->X_map()(0).sizeJ()+(j_map_previous-_radius_kernel);i<_W_kernels(0).sizeI();i++,index_map+=(neural_matrix->X_map()(0).sizeJ()-_W_kernels(0).sizeJ())){
                            for(unsigned int j=0;j<_W_kernels(0).sizeJ();j++,index_map++,index_kernel_ij++){
                                sum+=_W_kernels(index_map_previous + index_start_kernel)(index_kernel_ij)*neural_matrix->X_map()(index_map_previous)(index_map);
                            }
                        }
                    }
                    map_out(i_map_next,j_map_next)=sum;
                }
            }
        }

    }
    for(unsigned int i=0;i<__Y.size();i++){
        this->__X(i) = NeuronSigmoid::activation(this->__Y(i));
    }

}
void NeuralLayerMatrixConvolutionSubScaling::backwardCPU(NeuralLayer& layer_previous){
    for(unsigned int i=0;i<this->__Y.size();i++){
        this->_d_E_Y(i) = this->_d_E_X(i)*NeuronSigmoid::derivedActivation(this->__X(i));
    }




    if( NeuralLayerMatrix * neural_matrix = dynamic_cast< NeuralLayerMatrix *>(&layer_previous)){
        for(unsigned int i=0;i<neural_matrix->d_E_X_map().size();i++){
            for(unsigned int j=0;j<neural_matrix->d_E_X_map()(i).size();j++){
                neural_matrix->d_E_X_map()(i)(j)=0;
            }
        }
        for(unsigned int index_map=0;index_map<this->_d_E_Y_reference.size();index_map++){
            MatNReference<2,F32> &map_error_out =  this->_d_E_Y_reference[index_map];

            int index_start_kernel = index_map*neural_matrix->X_map().size();
            for(unsigned int i_map_next=0,i_map_previous=_radius_kernel;i_map_next<map_error_out.sizeI();i_map_next++,i_map_previous+=_sub_resolution_factor){
                for(unsigned int j_map_next=0,j_map_previous=_radius_kernel;j_map_next<map_error_out.sizeJ();j_map_next++,j_map_previous+=_sub_resolution_factor){
                    F32 d_error_y_value = map_error_out(i_map_next,j_map_next);

                    //convolution
                    for(unsigned int index_map_previous=0;index_map_previous<neural_matrix->X_map().size();index_map_previous++){
                        this->_d_E_W_biais(index_map_previous + index_start_kernel)+=d_error_y_value;
                        for(unsigned int i=0,index_kernel_ij=0,index_map = (i_map_previous-_radius_kernel)*neural_matrix->X_map()(0).sizeJ()+(j_map_previous-_radius_kernel);i<_W_kernels(0).sizeI();i++,index_map+=(neural_matrix->X_map()(0).sizeJ()-_W_kernels(0).sizeJ())){
                            for(unsigned int j=0;j<_W_kernels(0).sizeJ();j++,index_map++,index_kernel_ij++){

                                this->_d_E_W_kernels(index_map_previous + index_start_kernel)(index_kernel_ij)+=neural_matrix->X_map()(index_map_previous)(index_map)*d_error_y_value;
                                neural_matrix->d_E_X_map()(index_map_previous)(index_map)+=  this->_W_kernels(index_map_previous + index_start_kernel)(index_kernel_ij)  *d_error_y_value;
                            }
                        }
                    }
                }
            }
        }

    }
}
void NeuralLayerMatrixConvolutionSubScaling::learn(F32 lambda_regulation){
    for(unsigned int i=0;i<this->_d_E_W_kernels.size();i++){
        for(unsigned int j=0;j<this->_d_E_W_kernels(i).size();j++){
            this->_W_kernels(i)(j)=lambda_regulation*this->_W_kernels(i)(j)-_mu*this->_d_E_W_kernels(i)(j);
        }
    }
    for(unsigned int i=0;i<this->_d_E_W_biais.size();i++){
        this->_W_biais(i)=lambda_regulation*this->_W_biais(i)-_mu*this->_d_E_W_biais(i);
    }

    for(unsigned int i=0;i<this->_d_E_W_kernels.size();i++){
        for(unsigned int j=0;j<this->_d_E_W_kernels(i).size();j++){
            this->_d_E_W_kernels(i)(j)=0;
        }
    }
    for(unsigned int i=0;i<this->_d_E_W_biais.size();i++){
        this->_d_E_W_biais(i)=0;
    }

}
NeuralLayer * NeuralLayerMatrixConvolutionSubScaling::clone(){
    NeuralLayerMatrixConvolutionSubScaling * layer = new NeuralLayerMatrixConvolutionSubScaling(*this);
    layer->_Y_reference.clear();
    layer->_X_reference.clear();
    for(unsigned int i=0;i<this->X_map().size();i++){
        layer->_Y_reference.push_back(MatNReference<2,F32>(this->X_map()(0).getDomain(),layer->__Y.data()+this->X_map()(0).getDomain().multCoordinate()*i));
        layer->_X_reference.push_back(MatNReference<2,F32>(this->X_map()(0).getDomain(),layer->__X.data()+this->X_map()(0).getDomain().multCoordinate()*i));
    }
    return layer;
}





NeuralLayerLinearInput::NeuralLayerLinearInput(unsigned int nbr_neurons)
    :NeuralLayerLinear(nbr_neurons){}
void NeuralLayerLinearInput::forwardCPU(const NeuralLayer& ) {}
void NeuralLayerLinearInput::backwardCPU(NeuralLayer& ) {}
void NeuralLayerLinearInput::learn(F32 ){}
void NeuralLayerLinearInput::setTrainable(bool istrainable){NeuralLayerLinear::setTrainable(istrainable);}
NeuralLayer * NeuralLayerLinearInput::clone(){
    return new NeuralLayerLinearInput(*this);
}
NeuralLayerMatrixInput::NeuralLayerMatrixInput(unsigned int sizei,unsigned int sizej,unsigned int nbr_map)
    :NeuralLayerMatrix(sizei,  sizej,  nbr_map){}
void NeuralLayerMatrixInput::forwardCPU(const NeuralLayer& ) {}
void NeuralLayerMatrixInput::backwardCPU(NeuralLayer& ) {}
void NeuralLayerMatrixInput::learn(F32 ){}
void NeuralLayerMatrixInput::setTrainable(bool istrainable){NeuralLayerMatrix::setTrainable(istrainable);}
NeuralLayer * NeuralLayerMatrixInput::clone(){
    NeuralLayerMatrixInput * layer = new NeuralLayerMatrixInput(this->X_map()(0).sizeI(),this->X_map()(0).sizeJ(),this->X_map().size());
    return layer;
}

NeuralNet::NeuralNet()
{}

NeuralNet::NeuralNet(const NeuralNet & neural){

    this->_label2string = neural._label2string;

    this->clear();
    for(unsigned int i=0;i<neural._v_layer.size();i++){
        this->_v_layer.push_back(neural._v_layer(i)->clone());
    }
}

NeuralNet & NeuralNet::operator =(const NeuralNet & neural){
    this->_label2string = neural._label2string;
    this->clear();
    for(unsigned int i=0;i<neural._v_layer.size();i++){
        this->_v_layer.push_back(neural._v_layer(i)->clone());
    }
    return *this;
}

NeuralNet::~NeuralNet(){
    clear();
}

void NeuralNet::addLayerLinearInput(unsigned int nbr_neurons){
    this->_v_layer.push_back(new NeuralLayerLinearInput(nbr_neurons));
}
void NeuralNet::addLayerMatrixInput(unsigned int size_i,unsigned int size_j,unsigned int nbr_map){
    this->_v_layer.push_back(new NeuralLayerMatrixInput(size_i,size_j,nbr_map));
}
void NeuralNet::addLayerLinearFullyConnected(unsigned int nbr_neurons){
    if(_v_layer.size()==0){
        this->_v_layer.push_back(new NeuralLayerLinearFullyConnected(0,nbr_neurons));
    }else{
        this->_v_layer.push_back(new NeuralLayerLinearFullyConnected((*(_v_layer.rbegin()))-> X().size(),nbr_neurons));
    }
}
void NeuralNet::addLayerMatrixConvolutionSubScaling(unsigned int nbr_map,unsigned int sub_scaling_factor,unsigned int radius_kernel){
    if(NeuralLayerMatrix * neural_matrix = dynamic_cast<NeuralLayerMatrix *>(*(_v_layer.rbegin()))){
        this->_v_layer.push_back(new NeuralLayerMatrixConvolutionSubScaling( nbr_map, sub_scaling_factor,  radius_kernel,neural_matrix->X_map()(0).sizeI(),neural_matrix->X_map()(0).sizeJ(),neural_matrix->X_map().size()));
    }
}
void NeuralNet::addLayerMatrixMaxPool(unsigned int sub_scaling_factor){
    if(NeuralLayerMatrix * neural_matrix = dynamic_cast<NeuralLayerMatrix *>(*(_v_layer.rbegin()))){
        this->_v_layer.push_back(new NeuralLayerMatrixMaxPool( sub_scaling_factor,  neural_matrix->X_map()(0).sizeI(),neural_matrix->X_map()(0).sizeJ(),neural_matrix->X_map().size()));
    }
}


void NeuralNet::setLearnableParameter(F32 mu){
    for(unsigned int i=0;i<_v_layer.size();i++){
        _v_layer(i)->setLearnableParameter(mu);
    }
}

void NeuralNet::setTrainable(bool istrainable){
    for(unsigned int i=0;i<_v_layer.size();i++){
        _v_layer(i)->setTrainable(istrainable);
    }
}
void NeuralNet::learn(F32 lambda_regulation){
    for(unsigned int i=0;i<_v_layer.size();i++){
        _v_layer(i)->learn(lambda_regulation);
    }
}
void NeuralNet::forwardCPU(const VecF32& X_in, VecF32& X_out){
    std::copy(X_in.begin(),X_in.end(), (*(_v_layer.begin()))->X().begin());
    for(unsigned int i=1;i<_v_layer.size();i++){
        _v_layer(i)->forwardCPU(*_v_layer(i-1));
    }
    if(X_out.size()!=(*(_v_layer.rbegin()))->X().size()){
        X_out.resize((*(_v_layer.rbegin()))->X().size());
    }
    std::copy((*(_v_layer.rbegin()))->X().begin(),(*(_v_layer.rbegin()))->X().end(),X_out.begin());
}

void NeuralNet::backwardCPU(const VecF32& X_expected){

    //first output layer
    NeuralLayer* layer_last = _v_layer[_v_layer.size()-1];
    for(unsigned int j=0;j<X_expected.size();j++){
        layer_last->d_E_X()(j) = ( layer_last->X()(j)-X_expected(j));
    }

    for( int index_layer=_v_layer.size()-1;index_layer>0;index_layer--){
        NeuralLayer* layer = _v_layer[index_layer];
        NeuralLayer* layer_previous = _v_layer[index_layer-1];
        layer->backwardCPU(* layer_previous);
    }
}
NeuralLayer::~NeuralLayer(){

}

void NeuralNet::clear(){
    for(unsigned int i=0;i<_v_layer.size();i++){
        delete _v_layer[i];
    }
    _v_layer.clear();
}
void NeuralNet::load(const char * file)
{
    XMLDocument doc;
    doc.load(file);
    load(doc);
}
void NeuralNet::loadByteArray(const char *  file)
{
    XMLDocument doc;
    doc.loadFromByteArray(file);
    load(doc);
}

void NeuralNet::load(XMLDocument &doc)
{
    //to circumvent our locales problem in String2Float()
    bool use_optimized_string2float = true;
    std::string sf = "0.1234";
    F32 f1, f2;
    pop::BasicUtility::String2Any(sf, f1);
    pop::BasicUtility::String2Float(sf, f2);
    if (f1 != f2) {
        use_optimized_string2float = false;
    }

    this->clear();
    XMLNode node1 = doc.getChild("label2String");
    std::string type1 = node1.getAttribute("id");
    BasicUtility::String2Any(type1,_label2string);
    XMLNode node = doc.getChild("layers");
    int i=0;
    for (XMLNode tool = node.firstChild(); tool; tool = tool.nextSibling(),++i){
        std::string type = tool.getAttribute("type");
        if(type=="NNLayer::INPUTMATRIX"){
            Vec2I32 domain;
            int nbr_map;
            BasicUtility::String2Any(tool.getAttribute("size"),domain);
            BasicUtility::String2Any(tool.getAttribute("nbr_map"),nbr_map);
            if(tool.hasAttribute("method")&&tool.hasAttribute("normalization")){
                int method;
                BasicUtility::String2Any(tool.getAttribute("method"),method);
                _method = static_cast<NNLayerMatrix::CenteringMethod>(method) ;
                int method_norm;
                BasicUtility::String2Any(tool.getAttribute("normalization"),method_norm);
                _normalization_value= static_cast<NNLayerMatrix::NormalizationValue>(method_norm) ;
            }
            this->addLayerMatrixInput(domain(0),domain(1),nbr_map);
        }else if(type=="NNLayer::INPUTLINEAR"){
            int domain;
            BasicUtility::String2Any(tool.getAttribute("size"),domain);
            this->addLayerLinearInput(domain);
        }
        else if(type=="NNLayer::MATRIXCONVOLUTIONNAL"){

            std::string str = tool.getAttribute("nbr_map");
            int nbr_map;
            BasicUtility::String2Any(str,nbr_map);

            str = tool.getAttribute("sizekernel");
            int sizekernel;
            BasicUtility::String2Any(str,sizekernel);

            str = tool.getAttribute("subsampling");
            int subsampling;
            BasicUtility::String2Any(str,subsampling);

            this->addLayerMatrixConvolutionSubScaling(nbr_map,subsampling,sizekernel);

            std::string str_biais = tool.getAttribute("weight_biais");
            std::string str_kernel = tool.getAttribute("weight_kernel");
            if(NeuralLayerMatrixConvolutionSubScaling * neural_matrix = dynamic_cast<NeuralLayerMatrixConvolutionSubScaling *>(*(_v_layer.rbegin()))){
                std::istringstream stream_biais(str_biais);
                for(unsigned int index_weight=0;index_weight<neural_matrix->_W_biais.size();index_weight++){
                    F32 weight ;
                    str = pop::BasicUtility::getline( stream_biais, ";" );
                    (use_optimized_string2float ? pop::BasicUtility::String2Float(str, weight) : pop::BasicUtility::String2Any(str, weight));
                    neural_matrix->_W_biais[index_weight]=weight;
                }
                std::istringstream stream_kernel(str_kernel);
                for(unsigned int index_weight=0;index_weight<neural_matrix->_W_kernels.size();index_weight++){
                    for(unsigned int index_weight_j=0;index_weight_j<neural_matrix->_W_kernels(index_weight).size();index_weight_j++){
                        F32 weight ;
                        str = pop::BasicUtility::getline( stream_kernel, ";" );
                        (use_optimized_string2float ? pop::BasicUtility::String2Float(str, weight) : pop::BasicUtility::String2Any(str, weight));
                        neural_matrix->_W_kernels(index_weight)(index_weight_j)=weight;
                    }
                }
            }
        }else if(type=="NNLayer::FULLYCONNECTED"){
            std::string str = tool.getAttribute("size");
            int size;
            BasicUtility::String2Any(str,size);
            this->addLayerLinearFullyConnected(size);
            str = tool.getAttribute("weight");
            if(NeuralLayerLinearFullyConnected * neural_linear = dynamic_cast<NeuralLayerLinearFullyConnected *>(*(_v_layer.rbegin()))){
                std::istringstream stream(str);
                for(unsigned int index_weight=0;index_weight<neural_linear->_W.size();index_weight++){
                    F32 weight ;
                    str = pop::BasicUtility::getline( stream, ";" );
                    (use_optimized_string2float ? pop::BasicUtility::String2Float(str, weight) : pop::BasicUtility::String2Any(str, weight));
                    neural_linear->_W[index_weight] = weight;
                }
            }
        }
        else if(type=="NNLayer::MAXPOOL"){
            std::string str = tool.getAttribute("sub_scaling");
            int sub_resolution;
            BasicUtility::String2Any(str,sub_resolution);
            this->addLayerMatrixMaxPool(sub_resolution);

        }
    }
}
void NeuralNet::save(const char * file)const
{
    XMLDocument doc;
    XMLNode node1 = doc.addChild("label2String");
    node1.addAttribute("id",BasicUtility::Any2String(_label2string));
    XMLNode node = doc.addChild("layers");
    for(unsigned int i=0;i<this->_v_layer.size();i++){
        NeuralLayer * layer = this->_v_layer[i];
        if(const NeuralLayerMatrixInput *layer_matrix = dynamic_cast<const NeuralLayerMatrixInput *>(layer)){
            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::INPUTMATRIX");
            nodechild.addAttribute("size",BasicUtility::Any2String(layer_matrix->X_map()(0).getDomain()));
            nodechild.addAttribute("nbr_map",BasicUtility::Any2String(layer_matrix->X_map().size()));
            nodechild.addAttribute("method",BasicUtility::Any2String(_method));
            nodechild.addAttribute("normalization",BasicUtility::Any2String(_normalization_value));
        }
        else if(const NeuralLayerLinearInput *layer_linear = dynamic_cast<const NeuralLayerLinearInput *>(layer)){

            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::INPUTLINEAR");
            nodechild.addAttribute("size",BasicUtility::Any2String(layer_linear->X().size()));
            nodechild.addAttribute("method",BasicUtility::Any2String((_method)));
            nodechild.addAttribute("normalization",BasicUtility::Any2String(_normalization_value));
        }else if(const NeuralLayerMatrixConvolutionSubScaling *layer_conv = dynamic_cast<const NeuralLayerMatrixConvolutionSubScaling *>(layer)){
            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::MATRIXCONVOLUTIONNAL");
            nodechild.addAttribute("nbr_map",BasicUtility::Any2String(layer_conv->X_map().size()));
            nodechild.addAttribute("sizekernel",BasicUtility::Any2String(layer_conv->_radius_kernel));
            nodechild.addAttribute("subsampling",BasicUtility::Any2String(layer_conv->_sub_resolution_factor));

            std::string weight_str;
            for(unsigned int index_w=0;index_w<layer_conv->_W_biais.size();index_w++){
                weight_str+=BasicUtility::Any2String(layer_conv->_W_biais[index_w])+";";
            }
            nodechild.addAttribute("weight_biais",weight_str);
            weight_str.clear();
            for(unsigned int index_w=0;index_w<layer_conv->_W_kernels.size();index_w++){
                for(unsigned int index_weight_j=0;index_weight_j<layer_conv->_W_kernels(index_w).size();index_weight_j++){
                    weight_str+=BasicUtility::Any2String(layer_conv->_W_kernels(index_w)(index_weight_j))+";";
                }
            }
            nodechild.addAttribute("weight_kernel",weight_str);

        }
        else if(const NeuralLayerLinearFullyConnected *layer_fully= dynamic_cast<const NeuralLayerLinearFullyConnected *>(layer))
        {
            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::FULLYCONNECTED");
            nodechild.addAttribute("size",BasicUtility::Any2String(layer_fully->X().size()));

            std::string weight_str;
            for(unsigned int index_w=0;index_w<layer_fully->_W.size();index_w++){
                weight_str+=BasicUtility::Any2String(layer_fully->_W[index_w])+";";
            }
            nodechild.addAttribute("weight",weight_str);
        }else if(const NeuralLayerMatrixMaxPool *layer_max_pool= dynamic_cast<const NeuralLayerMatrixMaxPool *>(layer)){
            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::MAXPOOL");
            nodechild.addAttribute("sub_scaling",BasicUtility::Any2String(layer_max_pool->_sub_resolution_factor));
        }
    }
    doc.save(file);
}
const Vec<std::string>& NeuralNet::label2String()const{
    return _label2string;
}
Vec<std::string>& NeuralNet::label2String(){
    return _label2string;
}
const Vec<NeuralLayer*>& NeuralNet::layers()const{
    return _v_layer;
}
Vec<NeuralLayer*>& NeuralNet::layers(){
    return _v_layer;
}
VecF32 NeuralNet::inputMatrixToInputNeuron(const MatN<2,UI8>  & matrix){
    if(NeuralLayerMatrix* layer_matrix = dynamic_cast<NeuralLayerMatrix *>(this->_v_layer(0))){
        Vec2I32 domain = layer_matrix->X_map()(0).getDomain();
        return NNLayerMatrix::inputMatrixToInputNeuron(matrix,domain,this->_method,this->_normalization_value);
    }else{
        std::cerr<<"No matrixlayer  for neural network"<<std::endl;
        return VecF32();
    }
}
std::pair<Vec2I32,int> NeuralNet::getDomainMatrixInput()const{
    if(NeuralLayerMatrix* layer_matrix = dynamic_cast<NeuralLayerMatrix *>(*(this->_v_layer.begin()))){
        return std::make_pair(layer_matrix->X_map()(0).getDomain(),layer_matrix->X_map().size());
    }else{
        std::cerr<<"No matrixlayer  for neural network"<<std::endl;
        return std::make_pair(Vec2I32(0),0);
    }
}
std::pair<Vec2I32,int> NeuralNet::getDomainMatrixOutput()const{
    if(NeuralLayerMatrix* layer_matrix = dynamic_cast<NeuralLayerMatrix *>(*(this->_v_layer.rbegin()))){
        return std::make_pair(layer_matrix->X_map()(0).getDomain(),layer_matrix->X_map().size());
    }else{
        std::cerr<<"No matrixlayer  for neural network"<<std::endl;
        return std::make_pair(Vec2I32(0),0);
    }
}
MatNReference<2,F32>& NeuralNet::getMatrixOutput(int map_index)const{
    if(NeuralLayerMatrix* layer_matrix = dynamic_cast<NeuralLayerMatrix *>(*(this->_v_layer.rbegin()))){
        return layer_matrix->X_map()(map_index);
    }else{
        std::cerr<<"No matrix layer  for neural network"<<std::endl;
        return layer_matrix->X_map()(0);
    }
}

//NormalizationMatrixInputMass::NormalizationMatrixInputMass(Vec2I32 domain,NormalizationMatrixInput::NormalizationValue normalization=NormalizationMatrixInput::MinusOneToOne){

//}

//virtual ~NormalizationMatrixInputMass();
//virtual VecF32 inputMatrixToInputNeuron(const Mat2UI8  & matrix);
//virtual NormalizationMatrixInput * clone();
//NormalizationMatrixInput::NormalizationMatrixInput(Vec2I32 domain)
//    :_domain(domain){
//}
//NormalizationMatrixInput::~NormalizationMatrixInput(){
//}
}
