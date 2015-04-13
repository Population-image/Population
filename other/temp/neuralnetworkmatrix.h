#ifndef NEURALNETWORKMATRIX_H
#define NEURALNETWORKMATRIX_H

#include"Population.h"
using namespace pop;//Population namespace


enum TypeLayer{
    LAYER_INPUT,
    LAYER_INPUT_MATRIX,
    LAYER_FULLY_CONNECTED,
    LAYER_CONVOLUTIONNAL
};
struct Layer
{
    F32 sigmoid(F32 x){ return 1.7159*tanh(0.66666667*x);}
    F32 derived_sigmoid(F32 S){ return 0.666667f/1.7159f*(1.7159f+(S))*(1.7159f-(S));}  // derivative of the sigmoid as a function of the sigmoid's output

    Vec<F32> _X;
    Vec<F32> _Y;
    Vec<F32> _W;

    int _nbr_map;
    int _sizei_map;
    int _sizej_map;


    int _nbr_kernel;
    int _sizei_kernel;
    int _sizej_kernel;
    int _sub_resolution_factor;

    Vec<F32> _d_E_X;
    Vec<F32> _d_E_Y;
    Vec<F32> _d_E_W;
    TypeLayer _type;

    void fullyConnected(const Layer & layer_previous){
        const F32 * ptr_X_init= layer_previous._X.data();
        const F32 * ptr_X_end = layer_previous._X.data()+layer_previous._X.size();
        const F32 * ptr_X_incr;

        const F32 * ptr_W_incr = this->_W.data();

        F32 * ptr_Y_init= this->_Y.data();
        F32 * ptr_Y_end = this->_Y.data()+this->_Y.size();
        F32 * ptr_Y_incr;

        for(ptr_Y_incr=ptr_Y_init;ptr_Y_incr!=ptr_Y_end;ptr_Y_incr++){
            F32 sum =0;
            ptr_X_incr = ptr_X_init;
            for(ptr_X_incr=ptr_X_init;ptr_X_incr!=ptr_X_end;ptr_X_incr++,ptr_W_incr++){
                sum+=(*ptr_X_incr)*(*ptr_W_incr);
            }
            *ptr_Y_incr=sum;
        }
    }


    void convolutional_single_point(F32* Y,int index_i,int index_j,int index_map, const Layer & layer_previous){
        int rayon_kernel=(_sizei_kernel-1)*0.5;
        int index_i_previous = index_i*_sub_resolution_factor+ rayon_kernel;
        int index_j_previous = index_j*_sub_resolution_factor+ rayon_kernel;

        const F32 * ptr_X_previous_start = layer_previous._X.data() +   index_j_previous + index_i_previous*layer_previous._sizej_map;//start point on the previous layer
        int X_sift    = rayon_kernel*(1+layer_previous._sizej_map);//start on the corner


        int map_size    = layer_previous._sizei_map*layer_previous._sizej_map;

        const F32 * ptr_W_incr = this->_W.data()+(_sizei_kernel*_sizej_kernel+1)*layer_previous._nbr_map*index_map ;//start weight

        for(unsigned int index_map_previous =0;index_map_previous<layer_previous._nbr_map;index_map_previous++){
            const F32 * ptr_X_previous = ptr_X_previous_start + map_size*index_map_previous-X_sift;
            for(unsigned int index_i_W =0;index_i_W<this->_sizei_kernel;index_i_W++){
                for(unsigned int index_j_W =0;index_j_W<this->_sizej_kernel;index_j_W++,ptr_X_previous++,ptr_W_incr++){
                    *Y+=*ptr_X_previous * *ptr_W_incr;
                }
                ptr_X_previous+=(layer_previous._sizej_map-this->_sizej_kernel);
            }
            *Y+=*ptr_W_incr;//biais weight;
            ptr_W_incr++;
        }
    }

    void convolutional(const Layer & layer_previous){
        F32 * ptr_Y_incr=this->_Y.data();
        //iter on the output point for each map
        //start for each map
        for(unsigned int index_map =0;index_map<this->_nbr_map;index_map++){
            for(unsigned int index_i =0;index_i<this->_sizei_map;index_i++){
                for(unsigned int index_j =0;index_j<this->_sizej_map;index_j++,ptr_Y_incr++){
                    *ptr_Y_incr=0;
                    convolutional_single_point(ptr_Y_incr,index_i,index_j,index_map,layer_previous);
                }
            }

        }
    }
    void error_W_Convolutional_single_point(const F32* d_E_Y,int index_i,int index_j,int index_map, const Layer & layer_previous){

        int index_i_previous = index_i*_sub_resolution_factor+ (_sizei_kernel-1)*0.5;
        int index_j_previous = index_j*_sub_resolution_factor+ (_sizej_kernel-1)*0.5;

        const F32 * ptr_X_previous_start = layer_previous._X.data() +   index_j_previous + index_i_previous*layer_previous._sizej_map;//start point on the previous layer
        int X_sift    = (_sizei_kernel-1)*0.5*(1+layer_previous._sizej_map);//start on the corner


        int map_size    = layer_previous._sizei_map*layer_previous._sizej_map;

        F32 * ptr_d_E_W_incr = this->_d_E_W.data()+(_sizei_kernel*_sizej_kernel+1)*layer_previous._nbr_map*index_map ;//start weight

        for(unsigned int index_map_previous =0;index_map_previous<layer_previous._nbr_map;index_map_previous++){
            const F32 * ptr_X_previous = ptr_X_previous_start + map_size*index_map_previous-X_sift;

            for(unsigned int index_i_W =0;index_i_W<this->_sizei_kernel;index_i_W++){
                for(unsigned int index_j_W =0;index_j_W<this->_sizej_kernel;index_j_W++,ptr_X_previous++,ptr_d_E_W_incr++){
                    //difference line with convolutional_single_point
                    *ptr_d_E_W_incr += *d_E_Y *   *ptr_X_previous ;
                }
                ptr_X_previous+=(layer_previous._sizej_map-this->_sizej_kernel);
            }
            //difference line with convolutional_single_point
            *ptr_d_E_W_incr += *d_E_Y  ;
            ptr_d_E_W_incr++;

        }
    }

    void error_X_Convolutional_single_point(const F32* d_E_Y,int index_i,int index_j,int index_map,  Layer & layer_previous){

        int index_i_previous = index_i*_sub_resolution_factor+ (_sizei_kernel-1)*0.5;
        int index_j_previous = index_j*_sub_resolution_factor+ (_sizej_kernel-1)*0.5;

        F32 * ptr_d_E_X_previous_start = layer_previous._d_E_X.data() +   index_j_previous + index_i_previous*layer_previous._sizej_map;//start point on the previous layer
        int X_sift    = (_sizei_kernel-1)*0.5*(1+layer_previous._sizej_map);//start on the corner


        int map_size    = layer_previous._sizei_map*layer_previous._sizej_map;

        const F32 * ptr_W_incr = this->_W.data()+(_sizei_kernel*_sizej_kernel+1)*layer_previous._nbr_map*index_map ;//start weight

        for(unsigned int index_map_previous =0;index_map_previous<layer_previous._nbr_map;index_map_previous++){
            F32 * ptr_d_E_X_previous = ptr_d_E_X_previous_start + map_size*index_map_previous-X_sift;

            for(unsigned int index_i_W =0;index_i_W<this->_sizei_kernel;index_i_W++){
                for(unsigned int index_j_W =0;index_j_W<this->_sizej_kernel;index_j_W++,ptr_d_E_X_previous++,ptr_W_incr++){
                    //difference line with convolutional_single_point
                    *ptr_d_E_X_previous += *d_E_Y *  * ptr_W_incr   ;
                }
                ptr_d_E_X_previous+=(layer_previous._sizej_map-this->_sizej_kernel);
            }
            //difference line with convolutional_single_point
            ptr_W_incr++;

        }
    }


    void error_W_Convolutional(const Layer & layer_previous){

        //difference line with convolutional
        for(unsigned int i=0;i<this->_d_E_W.size();i++){
            this->_d_E_W[i]=0;
        }
        //difference line with convolutional
        F32 * ptr_d_E_Y_incr=this->_d_E_Y.data();
        //iter on the output point for each map
        //start for each map
        for(unsigned int index_map =0;index_map<this->_nbr_map;index_map++){
            for(unsigned int index_i =0;index_i<this->_sizei_map;index_i++){
                for(unsigned int index_j =0;index_j<this->_sizej_map;index_j++,ptr_d_E_Y_incr++){
                    //difference line with convolutional
                    error_W_Convolutional_single_point(ptr_d_E_Y_incr,index_i,index_j,index_map,layer_previous);
                }
            }
        }
    }
    void error_X_Convolutional(Layer & layer_previous){

        //difference line with convolutional
        for(unsigned int i=0;i<layer_previous._d_E_X.size();i++){
            layer_previous._d_E_X[i]=0;
        }
        //difference line with convolutional
        F32 * ptr_d_E_Y_incr=this->_d_E_Y.data();
        //iter on the output point for each map
        //start for each map
        for(unsigned int index_map =0;index_map<this->_nbr_map;index_map++){
            for(unsigned int index_i =0;index_i<this->_sizei_map;index_i++){
                for(unsigned int index_j =0;index_j<this->_sizej_map;index_j++,ptr_d_E_Y_incr++){
                    //difference line with convolutional
                    error_X_Convolutional_single_point(ptr_d_E_Y_incr,index_i,index_j,index_map,layer_previous);
                }
            }
        }
    }

    void f(){
        const F32 * ptr_Y_init= this->_Y.data();
        const F32 * ptr_Y_end = this->_Y.data()+this->_Y.size();
        const F32 * ptr_Y_incr;
        F32 * ptr_X_incr      = this->_X.data();
        for(ptr_Y_incr=ptr_Y_init;ptr_Y_incr!=ptr_Y_end;ptr_Y_incr++,ptr_X_incr++){
            *ptr_X_incr = sigmoid(*ptr_Y_incr);
        }
    }

    void error_f(){
        F32 * ptr_d_E_Y_init= this->_d_E_Y.data();
        F32 * ptr_d_E_Y_end = this->_d_E_Y.data()+this->_d_E_Y.size();
        F32 * ptr_d_E_Y_incr;
        const F32 * ptr_d_E_X_incr      = this->_d_E_X.data();
        const F32 * ptr_X_incr      = this->_X.data();
        for(ptr_d_E_Y_incr=ptr_d_E_Y_init;ptr_d_E_Y_incr!=ptr_d_E_Y_end;ptr_d_E_Y_incr++,ptr_d_E_X_incr++,ptr_X_incr++){
            *ptr_d_E_Y_incr = *ptr_d_E_X_incr* derived_sigmoid(*ptr_X_incr);
        }
    }
    void error_W_Fully_Connected(const Layer & layer_previous){
        const F32 * ptr_X_init= layer_previous._X.data();
        const F32 * ptr_X_end = layer_previous._X.data()+layer_previous._X.size();
        const F32 * ptr_X_incr;

        F32 * ptr_W_incr = this->_W.data();
        F32 * ptr_d_E_W_incr = this->_d_E_W.data();

        const F32 * ptr_d_E_Y_init= this->_d_E_Y.data();
        const F32 * ptr_d_E_Y_end = this->_d_E_Y.data()+this->_d_E_Y.size();
        const F32 * ptr_d_E_Y_incr;

        for(ptr_d_E_Y_incr=ptr_d_E_Y_init;ptr_d_E_Y_incr!=ptr_d_E_Y_end;ptr_d_E_Y_incr++){
            ptr_X_incr = ptr_X_init;
            for(ptr_X_incr=ptr_X_init;ptr_X_incr!=ptr_X_end;ptr_X_incr++,ptr_W_incr++,ptr_d_E_W_incr++){
                *ptr_d_E_W_incr  = (*ptr_d_E_Y_incr)*(*ptr_X_incr);
            }
        }
    }



    void error_X_Fully_Connected( Layer & layer_previous){
        F32 * ptr_d_E_X_init= layer_previous._d_E_X.data();
        F32 * ptr_d_E_X_end = layer_previous._d_E_X.data()+layer_previous._d_E_X.size()-1;//the last one is the biais
        F32 * ptr_d_E_X_incr;

        const F32 * ptr_W_incr = this->_W.data();
        const F32 * ptr_W_incr_j = this->_W.data();
        int size_j = this->_sizej_kernel;

        const F32 * ptr_d_E_Y_init= this->_d_E_Y.data();
        const F32 * ptr_d_E_Y_end = this->_d_E_Y.data()+this->_d_E_Y.size();
        const F32 * ptr_d_E_Y_incr;

        for(ptr_d_E_X_incr=ptr_d_E_X_init;ptr_d_E_X_incr!=ptr_d_E_X_end;ptr_d_E_X_incr++,ptr_W_incr_j++){
            *ptr_d_E_X_incr=0;
            ptr_W_incr=ptr_W_incr_j;
            for(ptr_d_E_Y_incr=ptr_d_E_Y_init;ptr_d_E_Y_incr!=ptr_d_E_Y_end;ptr_d_E_Y_incr++,ptr_W_incr+=size_j){
                *ptr_d_E_X_incr+=(*ptr_d_E_Y_incr)*(*ptr_W_incr);
            }
        }
    }

};


class NeuralNetworkFullyConnected2
{
public:
    Vec<Layer> _v_layer;
    F32 _eta;

    void setEta(F32 eta){
        for(unsigned int i = 0;i<_v_layer.size();i++){
            _eta=eta;
        }
    }
    void addInputNeuron(int size_neuron){
        Layer layer;
        layer._type = LAYER_INPUT;
        layer._X.resize(size_neuron+1,1);//add the neuron with constant value 1
        layer._d_E_X.resize(size_neuron+1);//add the neuron with constant value 1
        _v_layer.push_back(layer);
    }

    void addInputNeuronMatrix(Vec2I32 size, int nbr_map=1){
        int size_neuron = size.multCoordinate()*nbr_map;
        Layer layer;
        layer._type = LAYER_INPUT_MATRIX;
        layer._X.resize(size_neuron+1,1);//add the neuron with constant value 1
        layer._d_E_X.resize(size_neuron+1);//add the neuron with constant value 1
        layer._nbr_map =nbr_map;
        layer._sizei_map = size(0);
        layer._sizej_map = size(1);
        _v_layer.push_back(layer);
    }

    void addConvolutionnal(int radius_kernel=1, int nbr_map=1, int sub_resolution_factor=2){
        Layer layer;

        int sizei_map_previous = this->_v_layer.rbegin()->_sizei_map;
        int sizej_map_previous = this->_v_layer.rbegin()->_sizej_map;

        layer._sizei_map =std::floor (  (sizei_map_previous-1-2*radius_kernel)/(1.*sub_resolution_factor))+1;
        layer._sizej_map =std::floor (  (sizej_map_previous-1-2*radius_kernel)/(1.*sub_resolution_factor))+1;
        layer._sub_resolution_factor = sub_resolution_factor;
        std::cout<<layer._sizei_map <<std::endl;
        std::cout<<layer._sizej_map<<std::endl;
        layer._nbr_map   = nbr_map;

        int size_neuron = layer._sizei_map*layer._sizej_map*layer._nbr_map;

        layer._type = LAYER_CONVOLUTIONNAL;
        layer._X.resize(size_neuron+1,1);//add the neuron with constant value 1
        layer._d_E_X.resize(size_neuron+1);//add the neuron with constant value 1

        layer._Y.resize(size_neuron);
        layer._d_E_Y.resize(size_neuron);


        layer._nbr_kernel    =nbr_map*this->_v_layer.rbegin()->_nbr_map;
        layer._sizei_kernel = radius_kernel*2+1;
        layer._sizej_kernel = radius_kernel*2+1;

        int nbr_weigh = layer._nbr_kernel*(layer._sizei_kernel*layer._sizej_kernel+1);//kernel size :  weight= layer._radius_kernel*2, heigh= layer._radius_kernel*2, and one biais

        layer._W.resize(nbr_weigh);
        layer._d_E_W.resize(nbr_weigh);
        int erase=0;
        DistributionNormal n(0,1./std::sqrt(layer._sizei_kernel*layer._sizej_kernel));
        for(unsigned int i = 0;i<layer._W.size();i++){
            layer._W[i]=n.randomVariable();
            //layer._W[i]=erase;
            //erase++;
        }
        _v_layer.push_back(layer);
    }


    void addFullyConnected(int size_neuron){
        Layer layer;
        layer._type = LAYER_FULLY_CONNECTED;
        layer._X.resize(size_neuron+1,1);//add the neuron with constant value 1
        layer._d_E_X.resize(size_neuron+1);//add the neuron with constant value 1
        layer._Y.resize(size_neuron);
        layer._d_E_Y.resize(size_neuron);

        layer._nbr_map=0;
        layer._nbr_kernel=1;
        layer._sizej_kernel = _v_layer(_v_layer.size()-1)._X.size();
        layer._sizei_kernel = size_neuron;
        layer._W.resize(layer._sizei_kernel*layer._sizej_kernel);
        layer._d_E_W.resize(layer._sizei_kernel*layer._sizej_kernel);
        DistributionNormal n(0,1./std::sqrt(layer._sizej_kernel));
        for(unsigned int i = 0;i<layer._W.size();i++){
            layer._W[i]=n.randomVariable();
        }
        _v_layer.push_back(layer);
    }

    void propagateFront(int index_layer){
        Layer & layer_previous= _v_layer[index_layer-1];
        Layer & layer         = _v_layer[index_layer];
        if(layer._type==LAYER_FULLY_CONNECTED){
            layer.fullyConnected(layer_previous);
        }else if(layer._type==LAYER_CONVOLUTIONNAL){
            layer.convolutional(layer_previous);
        }
        layer.f();
    }
    void propagateFront(const pop::Vec<Mat2F32> & v_map , pop::VecF32 &out){
        for(unsigned int index_map=0;index_map<v_map.size();index_map++){
            int sift_map = index_map*v_map(0).size();
            std::copy(v_map(index_map).begin(),v_map(index_map).end(),_v_layer(0)._X.begin()+sift_map);
        }
        for(unsigned int layer_index=1;layer_index<_v_layer.size();layer_index++){
            propagateFront(layer_index);
        }
        std::copy(_v_layer.rbegin()->_X.begin(),_v_layer.rbegin()->_X.begin()+out.size(),out.begin());
    }


    void propagateFront(const pop::VecF32& in , pop::VecF32 &out){
        std::copy(in.begin(),in.end(),_v_layer(0)._X.begin());
        for(unsigned int layer_index=1;layer_index<_v_layer.size();layer_index++){
            propagateFront(layer_index);
        }
        if(out.size()<_v_layer.rbegin()->_X.size()-1)//1 for the shift layer
            out.resize(_v_layer.rbegin()->_X.size()-1);
        std::copy(_v_layer.rbegin()->_X.begin(),_v_layer.rbegin()->_X.begin()+out.size(),out.begin());
    }
    void propagateBackFirstDerivate(int index_layer){
        Layer & layer_previous= _v_layer[index_layer-1];
        Layer & layer         = _v_layer[index_layer];
        layer.error_f();
        if(layer._type==LAYER_FULLY_CONNECTED){
            layer.error_X_Fully_Connected(layer_previous);
            layer.error_W_Fully_Connected(layer_previous);

        }else if(layer._type==LAYER_CONVOLUTIONNAL){
            layer.error_X_Convolutional(layer_previous);
            layer.error_W_Convolutional(layer_previous);

        }
    }

    void propagateBackFirstDerivate(const pop::VecF32& desired_output){


        //first output layer
        Layer & layer_out = _v_layer[_v_layer.size()-1];
        for(unsigned int j=0;j<desired_output.size();j++){
            layer_out._d_E_X(j) = ( layer_out._X(j)-desired_output[j]);
        }

        for( int index_layer=_v_layer.size()-1;index_layer>0;index_layer--){
            propagateBackFirstDerivate(index_layer);
        }

    }
    void learningFirstDerivate(){
        for( int index_layer=1;index_layer<_v_layer.size();index_layer++){
            for(unsigned int indew_weight =0;indew_weight<_v_layer(index_layer)._W.size();indew_weight++)
                _v_layer(index_layer)._W(indew_weight) = _v_layer(index_layer)._W(indew_weight) - _eta*_v_layer(index_layer)._d_E_W(indew_weight);
        }
    }
};

void neuralNetworkForRecognitionForHandwrittenDigits()
{

    Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST("/home/vincent/train-images.idx3-ubyte","/home/vincent/train-labels.idx1-ubyte");

    Vec<Vec<Mat2UI8> > number_test =  TrainingNeuralNetwork::loadMNIST("/home/vincent/t10k-images.idx3-ubyte","/home/vincent/t10k-labels.idx1-ubyte");
    Vec2I32 domain(29,29);
    NeuralNetworkFullyConnected2 neural;
    neural.addInputNeuronMatrix(domain,1);
    neural.addConvolutionnal(2,20,2);
    neural.addConvolutionnal(2,75,2);
    neural.addFullyConnected(100);
    neural.addFullyConnected(10);



    Vec<VecF32> trainingins;
    Vec<VecF32> trainingouts;

    for(unsigned int i=0;i<number_training.size();i++){
        for(unsigned int j=0;j<number_training(i).size();j++){
            Mat2UI8 binary = number_training(i)(j);
            VecF32 vin = NNLayerMatrix::inputMatrixToInputNeuron(binary,domain);
            trainingins.push_back(vin);
            VecF32 v_out(static_cast<int>(number_training.size()),-1);
            v_out(i)=1;
            trainingouts.push_back(v_out);

        }
    }
    Vec<VecF32> testins;
    Vec<VecF32> testouts;
    for(unsigned int i=0;i<number_test.size();i++){
        for(unsigned int j=0;j<number_test(i).size();j++){
            Mat2UI8 binary = number_test(i)(j);
            VecF32 vin = NNLayerMatrix::inputMatrixToInputNeuron(binary,domain);
            testins.push_back(vin);
            VecF32 v_out(static_cast<int>(number_test.size()),-1);
            v_out(i)=1;
            testouts.push_back(v_out);
        }
    }
    F32 eta = 0.001f;
    int nbr_epoch =20;
    neural.setEta(eta);
    std::vector<int> v_global_rand(trainingins.size());
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i;
    std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;

    for(unsigned int i=0;i<nbr_epoch;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        int error_training=0,error_test=0;
        for(unsigned int j=0;j<v_global_rand.size();j++){
            VecF32 vout;
            neural.propagateFront(trainingins(v_global_rand[j]),vout);
//            std::cout<<trainingouts(v_global_rand[j])<<std::endl;
//            std::cout<<vout<<std::endl;
            neural.propagateBackFirstDerivate(trainingouts(v_global_rand[j]));
            neural.learningFirstDerivate();
            neural.propagateFront(trainingins(v_global_rand[j]),vout);
//            std::cout<<vout<<std::endl;

            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(trainingouts(v_global_rand[j]).begin(),std::max_element(trainingouts(v_global_rand[j]).begin(),trainingouts(v_global_rand[j]).end()));
            if(label1!=label2)
                error_training++;
            if(j*10%v_global_rand.size()==0)
                std::cout<<j*1.f/v_global_rand.size()<<std::endl;
        }
        for(unsigned int j=0;j<testins.size();j++){
            VecF32 vout;
            neural.propagateFront(testins(j),vout);
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(testouts(j).begin(),std::max_element(testouts(j).begin(),testouts(j).end()));
            if(label1!=label2)
                error_test++;
        }
        std::cout<<i<<"\t"<<error_training*1./trainingins.size()<<"\t"<<error_test*1.0/testins.size() <<"\t"<<eta<<std::endl;
        eta *=0.9f;
        neural.setEta(eta);
    }
}
void neuralnetwortest2(){
    {
        neuralNetworkForRecognitionForHandwrittenDigits();
        NeuralNetworkFeedForward n;
        Vec2I32 domain(29,29);
        n.addInputLayerMatrix(domain(0),domain(1));
        n.addLayerConvolutionalPlusSubScaling(6,5,2,1);
        n.addLayerConvolutionalPlusSubScaling(50,5,2,1);
        n.addLayerFullyConnected(100,1);
        n.addLayerFullyConnected(10,1);

        Mat2F32 m(29,29);

                    VecF32 b_in = n.inputMatrixToInputNeuron(m);
        VecF32 b_out(10);
        int time1 =time(NULL);
        for(unsigned int i=0;i<10000;i++){

            n.propagateFront(b_in,b_out);
        }
        int time2 =time(NULL);
        std::cout<<time2-time1<<std::endl;

        NeuralNetworkFullyConnected2 neural;
        neural.addInputNeuronMatrix(domain,1);
        neural.addConvolutionnal(1,6,2);
        neural.addConvolutionnal(1,50,2);
        neural.addFullyConnected(100);
        neural.addFullyConnected(10);

         time1 =time(NULL);
        for(unsigned int i=0;i<10000;i++){

            neural.propagateFront(b_in,b_out);
        }
         time2 =time(NULL);
        std::cout<<time2-time1<<std::endl;
        return ;

    }
    int size_input_matrix = 7;
    int nbr_map=1;
    NeuralNetworkFeedForward neural;
    neural.addInputLayerMatrix(size_input_matrix,size_input_matrix);
    neural.addLayerConvolutionalPlusSubScaling(nbr_map,3,2);
    neural.addLayerConvolutionalPlusSubScaling(nbr_map,3,2);
    neural.addLayerFullyConnected(1);
    neural.setLearningRate(0.01);

    NeuralNetworkFullyConnected2 test;
    test.addInputNeuronMatrix(Vec3I32(size_input_matrix,size_input_matrix),1);
    test.addConvolutionnal(1,nbr_map,2);
    test.addConvolutionnal(1,nbr_map,2);
    test.addFullyConnected(1);
    test.setEta(0.01);

    for(unsigned int i=1;i<=3;i++){
        NNLayer* layer_neural = neural.layers()(i);
        std::cout<<layer_neural->_weights.size()<<std::endl;
        std::cout<<test._v_layer(i)._W.size()<<std::endl;
        for(unsigned int index_weight=0;index_weight<layer_neural->_weights.size();index_weight++){
            if(index_weight==0){
                test._v_layer(i)._W(layer_neural->_weights.size()-1)=layer_neural->_weights(index_weight)->_Wn;
            }else{
                test._v_layer(i)._W(index_weight-1)=layer_neural->_weights(index_weight)->_Wn;
            }

        }

    }

    Mat2F32 m1(size_input_matrix,size_input_matrix);

    DistributionNormal d(0,1);
    for(unsigned int i=0;i<m1.size();i++){
        m1(i)=d.randomVariable();
    }
    //std::cout<<m1<<std::endl;
    Mat2F32 m2(size_input_matrix,size_input_matrix);
    for(unsigned int i=0;i<m2.size();i++){
        m2(i)=d.randomVariable();
    }

    Vec<Mat2F32> v_m(1);
    v_m(0) = m1;
    Vec<F32> v_in(size_input_matrix*size_input_matrix);
    for(unsigned int index_map=0;index_map<v_m.size();index_map++){
        int sift_map = index_map*v_m(0).size();
        std::copy(v_m(index_map).begin(),v_m(index_map).end(),v_in.begin()+sift_map);
    }
    //    v_m(1) = m2;
    while(1==1){


        VecF32 v_out(1);
        test.propagateFront(v_in,v_out);
        VecF32 v_out_temp=v_out;
        neural.propagateFront(v_in,v_out);
        std::cout<<"propagate front "<<v_out_temp(0)<<std::endl;
        std::cout<<"propagate front "<<v_out(0)<<std::endl;





        //exit(0);
        v_out(0)=-1;
        test.propagateBackFirstDerivate(v_out);
        test.learningFirstDerivate();
        v_out(0)=-1;
        neural.propagateBackFirstDerivate(v_out);
        neural.learningFirstDerivate();
        test.propagateFront(v_in,v_out);
        v_out_temp=v_out;
        neural.propagateFront(v_in,v_out);
        std::cout<<"propagate front "<<v_out_temp(0)<<std::endl;
        std::cout<<"propagate front "<<v_out(0)<<std::endl;


        //        std::cout<<"error"<<std::endl;
        //        for( int layer=2;layer>=0;layer--){
        //            std::cout<<"layer "<<layer<<std::endl;
        //            //            for(unsigned int i=0;i<neural.layers()(layer)->_neurons.size();i++)
        //            //                std::cout<<(neural.layers()(layer)->_neurons(i)->_dErr_dXn)<<" ";
        //            //            std::cout<<std::endl;
        //            //            std::cout<<test._v_layer(layer)._d_E_X<<std::endl;

        //            for(unsigned int i=0;i<neural.layers()(layer)->_weights.size();i++)
        //                std::cout<<(neural.layers()(layer)->_weights(i)->_dE_dWn)<<" ";
        //            std::cout<<std::endl;
        //            std::cout<<test._v_layer(layer)._d_E_W<<std::endl;

        //        }
        //        exit(0);

        //        for(unsigned int i=0;i<neural.layers()(1)->_neurons.size();i++)
        //            std::cout<<(neural.layers()(1)->_neurons(i)->_dErr_dXn)<<" ";
        //        std::cout<<std::endl;
        //        std::cout<<test._v_layer(1)._d_E_X<<std::endl;
        //        exit(0);

    }

}




























class NeuralNetworkFullyConnected
{
public:
    double sigmoid(double x){ return 1.7159*tanh(0.66666667*x);}
    double derived_sigmoid(double S){ return 0.666667f/1.7159f*(1.7159f+(S))*(1.7159f-(S));}  // derivative of the sigmoid as a function of the sigmoid's output


    void createNetwork(Vec<unsigned int> v_layer){
        _layers = v_layer;_X.clear();_Y.clear();_W.clear();_d_E_X.clear();_d_E_Y.clear();_d_E_W.clear();

        for(unsigned int i=0;i<v_layer.size();i++){
            int size_layer =v_layer[i];
            if(i!=v_layer.size()-1)
                //               _X.push_back(VecF32(size_layer,1));//no add the neuro
                _X.push_back(VecF32(size_layer+1,1));//add the neuron with constant value 1
            else
                _X.push_back(VecF32(size_layer,1));//except for the last one
            _Y.push_back(VecF32(size_layer));

            if(i!=0){
                int size_layer_previous = _X[i-1].size();
                Mat2F32  R(size_layer  ,size_layer_previous);
                DistributionNormal n(0,1./std::sqrt(size_layer_previous));
                for(unsigned int i = 0;i<R.size();i++){
                    R[i]=n.randomVariable();
                }
                _W.push_back(R);
            }

        }
    }

    void propagateFront(const pop::VecF32& in , pop::VecF32 &out){
        std::copy(in.begin(),in.end(),_X[0].begin());
        for(unsigned int layer_index=0;layer_index<_W.size();layer_index++){
            _Y[layer_index+1] = _W[layer_index] * _X[layer_index];
            for(unsigned int j=0;j<_Y[layer_index+1].size();j++){
                _X[layer_index+1][j] = sigmoid(_Y[layer_index+1][j]);
            }
        }
        if(out.size()!=_X.rbegin()->size())
            out.resize(_X.rbegin()->size());
        std::copy(_X.rbegin()->begin(),_X.rbegin()->begin()+out.size(),out.begin());
    }
    void propagateBackFirstDerivate(const pop::VecF32& desired_output){
        if(_d_E_X.size()==0){
            _d_E_X = _X;
            _d_E_Y = _Y;
            _d_E_W = _W;
        }
        for( int index_layer=_X.size()-1;index_layer>0;index_layer--){
            //X error
            if(index_layer==_X.size()-1){
                for(unsigned int j=0;j<_X[index_layer].size();j++){
                    _d_E_X[index_layer][j] = (_X[index_layer][j]-desired_output[j]);
                }
            }

            for(unsigned int i=0;i<_d_E_Y[index_layer].size();i++){
                _d_E_Y[index_layer][i] = _d_E_X[index_layer][i] * derived_sigmoid(_X[index_layer][i]);
            }
            for(unsigned int j=0;j<_d_E_W[index_layer-1].sizeJ();j++){
                for(unsigned int i=0;i<_d_E_W[index_layer-1].sizeI();i++){
                    _d_E_W[index_layer-1](i,j)= _d_E_Y[index_layer][i]*_X[index_layer-1][j];
                    _W[index_layer-1](i,j)= _W[index_layer-1](i,j) - _eta* _d_E_W[index_layer-1](i,j);
                }
            }
            for(unsigned int j=0;j<_X[index_layer-1].size();j++){
                _d_E_X[index_layer-1][j]=0;
                for(unsigned int i=0;i<_W[index_layer-1].sizeI();i++){
                    _d_E_X[index_layer-1][j]+=_W[index_layer-1](i,j)*_d_E_Y[index_layer][i];
                }
            }

        }

    }
    void save(const char * file){
        std::ofstream  out(file);
        out<<_layers;
        for(unsigned int i=0;i<_W.size();i++){
            MatNInOutPgm::writeRawData(_W(i) ,out);
        }
    }
    void load(const char * file){
        std::ifstream  is(file,std::iostream::binary);
        _layers.clear();
        is>>_layers;
        this->createNetwork(_layers);
        for(unsigned int i=0;i<_W.size();i++){
            MatNInOutPgm::readRaw(_W(i) ,is);
        }
    }
    void printNeuronVector(pop::VecF32 V, std::string label) {
        std::cout << label << "(" << V.size() << ") = [";
        std::cout << V << "]" << std::endl;
    }

    void printWeightMatrix(pop::Mat2F32 M, std::string label) {
        std::cout << label << "(" << M.sizeI() << ", " << M.sizeJ() << ") = [" << std::endl;
        std::cout << M << "]" << std::endl;
    }

    void printNetwork(void) {
        std::cout << "Number of layers: " << _X.size() << std::endl;
        std::cout << "Eta: " << _eta << std::endl;

        for (unsigned int l=0; l<_X.size(); l++) {
            std::cout << "\n-- Layer " << l << ":" << std::endl;

            printNeuronVector(_X[l], "_X");
            printNeuronVector(_Y[l], "_Y");
            if(_d_E_X.size()==0){
                std::cout << "_d_E_X = NULL" << std::endl;
                std::cout << "_d_E_Y = NULL" << std::endl;
            } else {
                printNeuronVector(_d_E_X[l], "_d_E_X");
                printNeuronVector(_d_E_Y[l], "_d_E_Y");
            }
            if (l != 0) {
                printWeightMatrix(_W[l-1], "_W");
                if(_d_E_X.size()==0){
                    std::cout << "_d_E_W = NULL" << std::endl;
                } else {
                    printWeightMatrix(_d_E_W[l-1], "_d_E_W");
                }
            } else {
                std::cout << "_W = NULL" << std::endl;
                std::cout << "_d_E_W = NULL" << std::endl;
            }
        }

        std::cout << "####################" << std::endl;

    }
    double _eta;
    Vec<pop::VecF32>  _X;
    Vec<pop::VecF32>  _Y;
    Vec<pop::Mat2F32> _W;
    Vec<pop::VecF32>  _d_E_X;
    Vec<pop::VecF32>  _d_E_Y;
    Vec<pop::Mat2F32> _d_E_W;
    Vec<unsigned>  _layers;
};




void neuralnetwortest(){
    {
        NeuralNetworkFullyConnected2 test;
        test.addInputNeuronMatrix(Vec3I32(4,4),2);
        test.addConvolutionnal(3,2,1);
        test.addFullyConnected(1);

        Vec<int> v_in(4*4);

    }
    {
        F32 eta = 0.1;
        int size =3;
        NeuralNetworkFullyConnected test;
        test._eta =eta;
        Vec<int> v_layer(size);
        //    v(0)=2;v(1)=1;
        v_layer(0)=2;v_layer(1)=3;v_layer(2)=1;
        test.createNetwork(v_layer);

        NeuralNetworkFullyConnected2 test2;
        test2.addInputNeuron(2);
        test2.addFullyConnected(3);
        test2.addFullyConnected(1);
        test2.setEta(eta);

        for(unsigned int i=1;i<size;i++){
            for(unsigned int j=0;j<test2._v_layer(i)._W.size();j++){
                test2._v_layer(i)._W(j) = test._W(i-1)(j);
            }
        }



        Vec<VecF32> v(4,VecF32(2));
        v(0)(0)=-1;v(0)(1)=-1;
        v(1)(0)= 1;v(1)(1)=-1;
        v(2)(0)=-1;v(2)(1)= 1;
        v(3)(0)= 1;v(3)(1)= 1;

        Vec<VecF32> vout(4,VecF32(1));
        vout(0)(0)=-1;
        vout(1)(0)= 1;
        vout(2)(0)= 1;
        vout(3)(0)=-1;

        Vec<int> v_global_rand(4);
        for(unsigned int i=0;i<v_global_rand.size();i++)
            v_global_rand[i]=i;


        while(1==1){
            std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
            for(unsigned int K=0;K<v_global_rand.size();K++)
            {
                int i = v_global_rand[K];
                VecF32 out;
                std::cout<<i<<std::endl;
                test.propagateFront(v(i),out);
                std::cout<<out<<std::endl;
                test2.propagateFront(v(i),out);
                std::cout<<out<<std::endl;


                test.propagateBackFirstDerivate(vout(i));
                test2.propagateBackFirstDerivate(vout(i));
                test.propagateFront(v(i),out);
                std::cout<<out<<std::endl;
                test2.propagateFront(v(i),out);
                std::cout<<out<<std::endl;


            }
        }
        return;
    }
    //        {
    //            NeuralNetworkFeedForward n2;
    //            n2.addInputLayer(2);
    //            n2.addLayerFullyConnected(3);
    //            n2.addLayerFullyConnected(1);

    //            NeuralNetworkFullConnection3 neural;


    //            neural._eta=0.001;





    NeuralNetworkFullyConnected network;


    Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST( "/home/vincent/Desktop/train-images.idx3-ubyte","/home/vincent/Desktop/train-labels.idx1-ubyte");
    Vec<Vec<Mat2UI8> > number_test =  TrainingNeuralNetwork::loadMNIST("/home/vincent/Desktop/t10k-images.idx3-ubyte","/home/vincent/Desktop/t10k-labels.idx1-ubyte");

    //    number_training.resize(10);
    //    number_test.resize(10);
    //    for(unsigned int i=0;i<number_training.size();i++){
    //        number_training(i).resize(1000);
    //        number_test(i).resize(50);
    //    }



    double size_in=number_training(0)(0).getDomain()(0)*number_training(0)(0).getDomain()(1);
    std::cout<<"size trainings: "<<number_training(0).size()<<std::endl;
    std::string net_struc = "BoundingBox -1,1 400_300_200_100";
    std::cout<<net_struc<<std::endl;
    Vec<unsigned int> v_layer;
    v_layer.push_back(size_in);
    //    v_layer.push_back(400);
    //    v_layer.push_back(300);
    //    v_layer.push_back(200);
    //    v_layer.push_back(100);

    v_layer.push_back(400);
    v_layer.push_back(300);
    v_layer.push_back(200);
    v_layer.push_back(100);
    v_layer.push_back(number_training.size());
    network.createNetwork(v_layer);
    network._eta = 0.001;


    NeuralNetworkFeedForward n2;
    n2.addInputLayer(size_in);
    n2.addLayerFullyConnected(100);
    n2.addLayerFullyConnected(100);
    n2.addLayerFullyConnected(number_training.size());
    n2.setLearningRate(0.001);


    Vec<VecF32> vtraining_in;
    Vec<VecF32> vtraining_out;


    TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,number_training(0)(0).getDomain(),NNLayerMatrix::BoundingBox,NNLayerMatrix::MinusOneToOne);

    Vec<VecF32> vtest_in;
    Vec<VecF32> vtest_out;
    TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,number_training(0)(0).getDomain(),NNLayerMatrix::BoundingBox,NNLayerMatrix::MinusOneToOne);

    number_training.clear();
    number_test.clear();

    std::vector<int> v_global_rand(vtraining_in.size());
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i;

    std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;


    for(unsigned int i=0;i<100;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        int error_training=0,error_test=0;

        for(unsigned int j=0;j<v_global_rand.size();j++){
            VecF32 vout;
            network.propagateFront(vtraining_in(v_global_rand[j]),vout);
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            network.propagateBackFirstDerivate(vtraining_out(v_global_rand[j]));
            int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
            if(label1!=label2){
                error_training++;
            }
        }
        for(unsigned int j=0;j<vtest_in.size();j++){
            VecF32 vout;
            network.propagateFront(vtest_in(j),vout);
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(vtest_out(j).begin(),std::max_element(vtest_out(j).begin(),vtest_out(j).end()));
            if(label1!=label2){
                error_test++;
            }
        }
        network.save((net_struc+"_"+pop::BasicUtility::Any2String(i)+".net").c_str());
        network._eta *=0.9;
        if(network._eta<0.000001)
            network._eta = 0.000001f;
        std::cout<<i<<"\t"<<error_training*1./v_global_rand.size()<<"\t"<<error_test*1./vtest_in.size() <<"\t"<<network._eta <<std::endl;
        //        std::cout<<i<<"\t"<<error_training2*1./v_global_rand.size()<<std::endl;
    }
}
#endif
