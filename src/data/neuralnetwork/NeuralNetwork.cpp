#include "data/neuralnetwork/NeuralNetwork.h"
#include "data/distribution/DistributionAnalytic.h"
#include "data/mat/MatN.h"
#include "data/mat/MatNInOut.h"
#include "data/mat/MatNDisplay.h"
#include "PopulationConfig.h"
#include "algorithm/Arithmetic.h"
namespace pop {

Mat2UI8 MNISTNeuralNetLeCun5::elasticDeformation(const Mat2UI8 &m, F32 sigma,F32 alpha){
    return GeometricalTransformation::elasticDeformation(m,sigma,alpha);
}


Mat2UI8 MNISTNeuralNetLeCun5::affineDeformation(const Mat2UI8 &m, F32 max_rotation_angle_random,F32 max_shear_angle_random,F32 max_scale_vertical_random,F32 max_scale_horizontal_random){


    DistributionUniformReal d_rot(-max_rotation_angle_random*pop::PI/180,max_rotation_angle_random*pop::PI/180);
    DistributionUniformReal d_shear(-max_shear_angle_random*pop::PI/180,max_shear_angle_random*pop::PI/180);


    DistributionUniformReal d_scale_vert(1-max_scale_vertical_random/100.f,1+max_scale_vertical_random/100.f);
    DistributionUniformReal d_scale_hor (1-max_scale_horizontal_random/100.f,1+max_scale_horizontal_random/100.f);


    F32 angle = d_rot.randomVariable();
    F32 shear = d_shear.randomVariable();

    Vec2F32 scale(d_scale_hor.randomVariable(),d_scale_vert.randomVariable());

    Mat2UI8 m_affine(m);
    Mat2x33F32 maffine  = GeometricalTransformation::translation2DHomogeneousCoordinate(m_affine.getDomain()/2);//go back to the buttom left corner (origin)
    maffine *=  GeometricalTransformation::scale2DHomogeneousCoordinate(scale);
    maffine *=  GeometricalTransformation::shear2DHomogeneousCoordinate(shear,0);
    maffine *=  GeometricalTransformation::rotation2DHomogeneousCoordinate(angle);//rotate
    maffine *=  GeometricalTransformation::translation2DHomogeneousCoordinate(-m_affine.getDomain()/2);
    return GeometricalTransformation::transformHomogeneous2D(maffine, m_affine);
}


int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}



Vec<Vec<Mat2UI8> > MNISTNeuralNetLeCun5::loadMNIST( std::string datapath,  std::string labelpath){
    Vec<Vec<Mat2UI8> > dataset(10);
    std::ifstream datas(datapath.c_str(),std::ios::binary);
    std::ifstream labels(labelpath.c_str(),std::ios::binary);

    if (!datas.is_open() || !labels.is_open()){
        std::cerr<<"binary files could not be loaded" << std::endl;
        return dataset;
    }

    int magic_number=0; int number_of_images=0;int r; int c;
    int n_rows=0; int n_cols=0; unsigned char temp=0;

    // parse data header
    datas.read((char*)&magic_number,sizeof(magic_number));
    magic_number=reverseInt(magic_number);
    datas.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images=reverseInt(number_of_images);
    datas.read((char*)&n_rows,sizeof(n_rows));
    n_rows=reverseInt(n_rows);
    datas.read((char*)&n_cols,sizeof(n_cols));
    n_cols=reverseInt(n_cols);

    // parse label header - ignore
    int dummy;
    labels.read((char*)&dummy,sizeof(dummy));
    labels.read((char*)&dummy,sizeof(dummy));

    for(int i=0;i<number_of_images;++i){
        pop::Mat2UI8 img(n_rows,n_cols);

        for(r=0;r<n_rows;++r){
            for(c=0;c<n_cols;++c){
                datas.read((char*)&temp,sizeof(temp));
                img(r,c) = temp;
            }
        }
        labels.read((char*)&temp,sizeof(temp));
        dataset[(int)temp].push_back(img);
    }
    return dataset;
}

NeuralNet MNISTNeuralNetLeCun5::createNet(std::string train_datapath,  std::string train_labelpath, std::string test_datapath,  std::string test_labelpath,UI32 nbr_epoch, UI32 lecun_or_simard,UI32 nbr_deformation,bool iselastic)

{
    //create the neural set
    NeuralNet net;
    if(lecun_or_simard==0){
        std::cout<<"LECUN"<<std::endl;
        net.addLayerMatrixInput(32,32,1);
        net.addLayerMatrixConvolutionSubScaling(6,1,2);
        net.addLayerMatrixMaxPool(2);
        net.addLayerMatrixConvolutionSubScaling(16,1,2);
        net.addLayerMatrixMaxPool(2);
        net.addLayerLinearFullyConnected(120);
        net.addLayerLinearFullyConnected(84);
    }else if(lecun_or_simard==1){
        std::cout<<"SIMARD"<<std::endl;
        net.addLayerMatrixInput(29,29,1);
        net.addLayerMatrixConvolutionSubScaling(6,2,2);
        net.addLayerMatrixConvolutionSubScaling(50,2,2);
        net.addLayerLinearFullyConnected(120);
        net.addLayerLinearFullyConnected(84);
    }
    net.addLayerLinearFullyConnected(10);

    Vec<std::string> label_digit;
    for(int i=0;i<10;i++)
        label_digit.push_back(BasicUtility::Any2String(i));
    net.label2String() = label_digit;


    //create the training set
    Vec<Vec<Mat2UI8> > number_training =  loadMNIST(train_datapath,train_labelpath);
    Vec<Vec<Mat2UI8> > number_test =  loadMNIST(test_datapath,test_labelpath);

    Vec<VecF32> vtraining_in;
    Vec<VecF32> vtraining_out;

    std::cout<<"Nbr deformation "<<nbr_deformation <<std::endl;
    if(iselastic==true)
        std::cout<<"Elastic deformation"<<std::endl;
    for(UI32 i=0;i<number_training.size();i++){
        for(UI32 j=0;j<number_training(i).size();j++){
            Mat2UI8 binary = number_training(i)(j);

            for(unsigned int k=0;k<nbr_deformation;k++){

                Mat2UI8 m_n = binary;
                pop::Draw::addBorder(m_n,3,0);
                if(iselastic==true)
                m_n = pop::MNISTNeuralNetLeCun5::elasticDeformation(m_n,3,2);
                if(i==1||i==7){
                   m_n = pop::MNISTNeuralNetLeCun5::affineDeformation(m_n,15,15,25,30);
                }else{
                   m_n = pop::MNISTNeuralNetLeCun5::affineDeformation(m_n,30,30,25,30);
                }
                VecF32 vin = net.inputMatrixToInputNeuron(m_n);
                vtraining_in.push_back(vin);
                VecF32 v_out(static_cast<int>(number_training.size()),-1);
                v_out(i)=1;
                vtraining_out.push_back(v_out);
            }
            VecF32 vin = net.inputMatrixToInputNeuron(binary);
            vtraining_in.push_back(vin);
            VecF32 v_out(static_cast<int>(number_training.size()),-1);
            v_out(i)=1;
            vtraining_out.push_back(v_out);

        }
    }

    Vec<VecF32> vtest_in;
    Vec<VecF32> vtest_out;
    for(unsigned int i=0;i<number_test.size();i++){
        for(unsigned int j=0;j<number_test(i).size();j++){
            Mat2UI8 binary = number_test(i)(j);
            VecF32 vin = net.inputMatrixToInputNeuron(binary);
            vtest_in.push_back(vin);
            VecF32 v_out(static_cast<int>(number_test.size()),-1);
            v_out(i)=1;
            vtest_out.push_back(v_out);
        }
    }

    //use the backprogation algorithm with first order method
    F32 eta =0.01f;
    net.setTrainable(true);
    net.setLearnableParameter(eta);

    //random vector to shuffle the trraining set
    std::vector<int> v_global_rand(vtraining_in.size());
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i;

    std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;
    for(unsigned int i=0;i<nbr_epoch;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        int error_training=0,error_test=0;
        for(unsigned int j=0;j<v_global_rand.size();j++){
            VecF32 vout;
            net.forwardCPU(vtraining_in(v_global_rand[j]),vout);
            net.backwardCPU(vtraining_out(v_global_rand[j]));
            net.learn();
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
            if(label1!=label2)
                error_training++;
        }

        for(unsigned int j=0;j<vtest_in.size();j++){
            VecF32 vout;
            net.forwardCPU(vtest_in(j),vout);
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(vtest_out(j).begin(),std::max_element(vtest_out(j).begin(),vtest_out(j).end()));
            if(label1!=label2)
                error_test++;
        }
        std::cout<<i<<"\t"<<error_training*1./vtraining_in.size()<<"\t"<<error_test*1.0/vtest_in.size() <<"\t"<<eta<<std::endl;
        eta *=0.9f;
        eta = (std::max)(eta,0.001f);
        net.setLearnableParameter(eta);
    }
    return net;
}

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
void NeuralLayerLinearFullyConnected::learn(){
    for(unsigned int i=0;i<this->_W.sizeI();i++){
        for(unsigned int j=0;j<this->_W.sizeJ();j++){
            this->_W(i,j)= this->_W(i,j) -  this->_mu*this->_d_E_W(i,j);
        }
    }
}
NeuralLayer * NeuralLayerLinearFullyConnected::clone(){
    return new   NeuralLayerLinearFullyConnected(*this);
}

NeuralLayerMatrixMaxPool::NeuralLayerMatrixMaxPool(unsigned int sub_scaling_factor,unsigned int sizei_map_previous,unsigned int sizej_map_previous,unsigned int nbr_map_previous)
    :NeuralLayerMatrix(static_cast<unsigned int>(std::floor (  sizei_map_previous/(1.f*sub_scaling_factor))),
                       static_cast<unsigned int>(std::floor ( sizej_map_previous/(1.f*sub_scaling_factor))),
                       nbr_map_previous),
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
                                value = (std::max)(value,map_layer_previous(i*_sub_resolution_factor+i_r,j*_sub_resolution_factor+j_r));
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
                                    this->_Y_reference(index_map)(i,j)=static_cast<F32>(i_r*_sub_resolution_factor+j_r);
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
                    int index = static_cast<int>( this->_Y_reference(index_map)(i,j));
                    int i_r,j_r;
                    pop::Arithmetic::euclideanDivision(index,(int)_sub_resolution_factor,i_r,j_r);
                    map_layer_previous(i*_sub_resolution_factor+i_r,j*_sub_resolution_factor+j_r)= map_layer(i,j);

                }
            }
        }
    }
}

void NeuralLayerMatrixMaxPool::learn( ){

}
NeuralLayer * NeuralLayerMatrixMaxPool::clone(){
    return new   NeuralLayerMatrixMaxPool(*this);
}



NeuralLayerMatrixConvolutionSubScaling::NeuralLayerMatrixConvolutionSubScaling(unsigned int nbr_map,unsigned int sub_scaling_factor,unsigned int radius_kernel,unsigned int sizei_map_previous,unsigned int sizej_map_previous,unsigned int nbr_map_previous)
    :NeuralLayerMatrix(static_cast<unsigned int>(std::floor (  (sizei_map_previous-1-2*radius_kernel)/(1.*sub_scaling_factor))+1),
                       static_cast<unsigned int>(std::floor (  (sizej_map_previous-1-2*radius_kernel)/(1.*sub_scaling_factor))+1)
                       ,nbr_map),
      _W_kernels(nbr_map*nbr_map_previous,Mat2F32(radius_kernel*2+1,radius_kernel*2+1)),
      _W_biais(nbr_map*nbr_map_previous),
      _sub_resolution_factor (sub_scaling_factor),
      _radius_kernel (radius_kernel)
{
    //std::cout<<(sizei_map_previous-1-2*radius_kernel)/(1.*sub_scaling_factor)+1<<std::endl;
    //normalize tbe number inverse square root of the connection feeding into the nodes)
    DistributionNormal n(0,1.f/((radius_kernel*2+1)*std::sqrt(nbr_map_previous*1.f)));
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
void NeuralLayerMatrixConvolutionSubScaling::learn(){
    for(unsigned int i=0;i<this->_d_E_W_kernels.size();i++){
        for(unsigned int j=0;j<this->_d_E_W_kernels(i).size();j++){
            this->_W_kernels(i)(j)=this->_W_kernels(i)(j)-_mu*this->_d_E_W_kernels(i)(j);
        }
    }
    for(unsigned int i=0;i<this->_d_E_W_biais.size();i++){
        this->_W_biais(i)=this->_W_biais(i)-_mu*this->_d_E_W_biais(i);
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
void NeuralLayerLinearInput::learn( ){}
void NeuralLayerLinearInput::setTrainable(bool istrainable){NeuralLayerLinear::setTrainable(istrainable);}
NeuralLayer * NeuralLayerLinearInput::clone(){
    return new NeuralLayerLinearInput(*this);
}
NeuralLayerMatrixInput::NeuralLayerMatrixInput(unsigned int sizei,unsigned int sizej,unsigned int nbr_map)
    :NeuralLayerMatrix(sizei,  sizej,  nbr_map){}
void NeuralLayerMatrixInput::forwardCPU(const NeuralLayer& ) {}
void NeuralLayerMatrixInput::backwardCPU(NeuralLayer& ) {}
void NeuralLayerMatrixInput::learn( ){}
void NeuralLayerMatrixInput::setTrainable(bool istrainable){NeuralLayerMatrix::setTrainable(istrainable);}
NeuralLayer * NeuralLayerMatrixInput::clone(){
    NeuralLayerMatrixInput * layer = new NeuralLayerMatrixInput(this->X_map()(0).sizeI(),this->X_map()(0).sizeJ(),this->X_map().size());
    return layer;
}

NeuralNet::NeuralNet()
    :_normalizationmatrixinput(new NormalizationMatrixInputMass())
{}

NeuralNet::NeuralNet(const NeuralNet & neural){

    this->_label2string = neural._label2string;

    this->clear();
    for(unsigned int i=0;i<neural._v_layer.size();i++){
        this->_v_layer.push_back(neural._v_layer(i)->clone());
    }
    _normalizationmatrixinput = neural._normalizationmatrixinput->clone();
}

NeuralNet & NeuralNet::operator =(const NeuralNet & neural){
    this->_label2string = neural._label2string;
    this->clear();
    for(unsigned int i=0;i<neural._v_layer.size();i++){
        this->_v_layer.push_back(neural._v_layer(i)->clone());
    }
    if(_normalizationmatrixinput!=NULL)
        delete _normalizationmatrixinput;
    _normalizationmatrixinput = neural._normalizationmatrixinput->clone();
    return *this;
}

NeuralNet::~NeuralNet(){
    if(_normalizationmatrixinput!=NULL)
        delete _normalizationmatrixinput;
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
void NeuralNet::learn(){
    for(unsigned int i=0;i<_v_layer.size();i++){
        _v_layer(i)->learn();
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
                int method_norm;
                BasicUtility::String2Any(tool.getAttribute("normalization"),method_norm);
                if(method==0){
                    NormalizationMatrixInputMass *mass = new NormalizationMatrixInputMass(static_cast<NormalizationMatrixInput::NormalizationValue>(method_norm));
                    this->setNormalizationMatrixInput(mass);
                }else{
                    NormalizationMatrixInputCentering *centering= new NormalizationMatrixInputCentering(static_cast<NormalizationMatrixInput::NormalizationValue>(method_norm));
                    this->setNormalizationMatrixInput(centering);
                }
            }else{
                NormalizationMatrixInput * norm = NormalizationMatrixInput::load(tool);
                this->setNormalizationMatrixInput(norm);
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

            this->_normalizationmatrixinput->save(nodechild);
            //            nodechild.addAttribute("method",BasicUtility::Any2String(_method));
            //            nodechild.addAttribute("normalization",BasicUtility::Any2String(_normalization_value));
        }
        else if(const NeuralLayerLinearInput *layer_linear = dynamic_cast<const NeuralLayerLinearInput *>(layer)){
            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::INPUTLINEAR");
            nodechild.addAttribute("size",BasicUtility::Any2String(layer_linear->X().size()));
            this->_normalizationmatrixinput->save(nodechild);
            //            nodechild.addAttribute("method",BasicUtility::Any2String((_method)));
            //            nodechild.addAttribute("normalization",BasicUtility::Any2String(_normalization_value));
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
void NeuralNet::setNormalizationMatrixInput(NormalizationMatrixInput * input){
    if(_normalizationmatrixinput!=NULL)
        delete _normalizationmatrixinput;
    _normalizationmatrixinput = input;
}

NormalizationMatrixInput::~NormalizationMatrixInput(){

}

void NormalizationMatrixInput::save(XMLNode & node)const{
    if(const NormalizationMatrixInputMass * mass= dynamic_cast<const NormalizationMatrixInputMass *>(this)){
        node.addAttribute("type_norm_matrix","MASS");
        node.addAttribute("normalization",BasicUtility::Any2String(mass->_normalization_value));
    }  else if(const NormalizationMatrixInputCentering * centering= dynamic_cast<const NormalizationMatrixInputCentering *>(this)){
        node.addAttribute("type_norm_matrix","CENTERING");
        node.addAttribute("normalization",BasicUtility::Any2String(centering->_normalization_value));
    }
}
NormalizationMatrixInput* NormalizationMatrixInput::load(const XMLNode & node){
    std::string type = node.getAttribute("type_norm_matrix");
    int method_norm;

    BasicUtility::String2Any(node.getAttribute("normalization"),method_norm);
    NormalizationMatrixInput::NormalizationValue method_norm_enum= static_cast<NormalizationMatrixInput::NormalizationValue>(method_norm) ;
    if(type=="MASS"){
        return new NormalizationMatrixInputMass(method_norm_enum);
    }else{
        return new NormalizationMatrixInputCentering(method_norm_enum);
    }
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
        return this->_normalizationmatrixinput->inputMatrixToInputNeuron(matrix,layer_matrix->_X_reference(0).getDomain());
    }else{
        std::cerr<<"No matrixlayer  for neural network"<<std::endl;
        return VecF32();
    }
}
//std::pair<Vec2I32,int> NeuralNet::getDomainMatrixInput()const{
//    if(NeuralLayerMatrix* layer_matrix = dynamic_cast<NeuralLayerMatrix *>(*(this->_v_layer.begin()))){
//        return std::make_pair(layer_matrix->X_map()(0).getDomain(),layer_matrix->X_map().size());
//    }else{
//        std::cerr<<"No matrixlayer  for neural network"<<std::endl;
//        return std::make_pair(Vec2I32(0),0);
//    }
//}
//std::pair<Vec2I32,int> NeuralNet::getDomainMatrixOutput()const{
//    if(NeuralLayerMatrix* layer_matrix = dynamic_cast<NeuralLayerMatrix *>(*(this->_v_layer.rbegin()))){
//        return std::make_pair(layer_matrix->X_map()(0).getDomain(),layer_matrix->X_map().size());
//    }else{
//        std::cerr<<"No matrixlayer  for neural network"<<std::endl;
//        return std::make_pair(Vec2I32(0),0);
//    }
//}
//MatNReference<2,F32>& NeuralNet::getMatrixOutput(int map_index)const{
//    if(NeuralLayerMatrix* layer_matrix = dynamic_cast<NeuralLayerMatrix *>(*(this->_v_layer.rbegin()))){
//        return layer_matrix->X_map()(map_index);
//    }else{
//        std::cerr<<"No matrix layer  for neural network"<<std::endl;
//        return layer_matrix->X_map()(0);
//    }
//}

NormalizationMatrixInputMass::NormalizationMatrixInputMass(NormalizationMatrixInput::NormalizationValue normalization)
    :_normalization_value(normalization)
{

}

VecF32 NormalizationMatrixInputMass::inputMatrixToInputNeuron(const Mat2UI8  & img,Vec2I32 domain){
    //center of gravity
    //Mat2UI8 img = Processing::threshold(img2,125);

    pop::Vec2I32 xmin(NumericLimits<int>::maximumRange(),NumericLimits<int>::maximumRange()),xmax(0,0);

    pop::Vec2F32 center_gravity(0,0);
    F32 weight_sum=0;
    ForEachDomain2D(x,img){
        center_gravity += static_cast<F32>(img(x))*pop::Vec2F32(x);
        weight_sum +=img(x);
        if(img(x)!=0){
            xmin=minimum(xmin,x);
            xmax=maximum(xmax,x);
        }
    }
    center_gravity = center_gravity/weight_sum;

    F32 max_i= (std::max)(xmax(0)-center_gravity(0),center_gravity(0)-xmin(0))*2;
    F32 max_j= (std::max)(xmax(1)-center_gravity(1),center_gravity(1)-xmin(1))*2;

    F32 homo = (std::max)(max_i/domain(0),max_j/domain(1));

    //    ForEachDomain2D(xx,mr){

    //         mrf(xx+trans)=mr(xx);
    F32 maxi=pop::NumericLimits<F32>::minimumRange();
    F32 mini=pop::NumericLimits<F32>::maximumRange();

    Mat2F32 mrf(domain);
    ForEachDomain2D(xx,mrf){
        pop::Vec2F32 xxx(xx);
        xxx = (xxx-Vec2F32(domain)/2.)*homo + center_gravity;
        mrf(xx)=img.interpolationBilinear(xxx);
        maxi=(std::max)(maxi,mrf(xx));
        mini=(std::min)(mini,mrf(xx));
    }
    if(maxi-mini==0){
        throw(std::string("[ERROR] in conversion matrix to neuron values"));
    }
    if(_normalization_value==0){
        F32 diff = (maxi-mini)/2.f;
        ForEachDomain2D(xxx,mrf){
            mrf(xxx) = (mrf(xxx)-mini)/diff-1;
        }
    }else{
        F32 diff = (maxi-mini);
        ForEachDomain2D(xxx,mrf){
            mrf(xxx) = (mrf(xxx)-mini)/diff;
        }
    }
    //        mrf.display();
    return VecF32(mrf);
}
NormalizationMatrixInputMass *NormalizationMatrixInputMass::clone(){
    return new NormalizationMatrixInputMass(_normalization_value);
}

NormalizationMatrixInputCentering::NormalizationMatrixInputCentering(NormalizationMatrixInput::NormalizationValue normalization)
    :_normalization_value(normalization)
{

}

VecF32 NormalizationMatrixInputCentering::inputMatrixToInputNeuron(const Mat2UI8  & img,Vec2I32 domain){
    //center of gravity

    //cropping
    pop::Vec2I32 xmin(NumericLimits<int>::maximumRange(),NumericLimits<int>::maximumRange()),xmax(0,0);
    for(unsigned int i=0;i<img.size();i++){

    }

    ForEachDomain2D(x,img){
        if(img(x)!=0){
            xmin=minimum(xmin,x);
            xmax=maximum(xmax,x);
        }
    }
    Mat2F32 m = img(xmin,xmax+1);
    //downsampling the input matrix
    int index;
    F32 scale_factor;
    if(F32(domain(0))/m.getDomain()(0)<F32(domain(1))/m.getDomain()(1)){
        index = 0;
        scale_factor = F32(domain(0))/m.getDomain()(0);
    }
    else{
        index = 1;
        scale_factor = F32(domain(1))/m.getDomain()(1);
    }

    Mat2F32 mr = GeometricalTransformation::scale(m,Vec2F32(scale_factor,scale_factor),MATN_INTERPOLATION_BILINEAR);
    Mat2F32 mrf(domain);
    Vec2I32 trans(0,0);
    if(index==0){
        trans(0)=0;
        trans(1)=(domain(1)-mr.getDomain()(1))/2;
    }else{
        trans(0)=(domain(0)-mr.getDomain()(0))/2;
        trans(1)=0;
    }


    F32 maxi=pop::NumericLimits<F32>::minimumRange();
    F32 mini=pop::NumericLimits<F32>::maximumRange();

    ForEachDomain2D(xx,mr){
        maxi=(std::max)(maxi,mr(xx));
        mini=(std::min)(mini,mr(xx));
        mrf(xx+trans)=mr(xx);
    }
    if(maxi-mini==0){
        throw(std::string("[ERROR] in conversion matrix to neuron values"));
    }
    if(_normalization_value==0){
        F32 diff = (maxi-mini)/2.f;
        ForEachDomain2D(xxx,mrf){
            mrf(xxx) = (mrf(xxx)-mini)/diff-1;
        }
    }else{
        F32 diff = (maxi-mini);
        ForEachDomain2D(xxx,mrf){
            mrf(xxx) = (mrf(xxx)-mini)/diff;
        }
    }
    return VecF32(mrf);
}
NormalizationMatrixInputCentering *NormalizationMatrixInputCentering::clone(){
    return new NormalizationMatrixInputCentering(_normalization_value);
}
}
