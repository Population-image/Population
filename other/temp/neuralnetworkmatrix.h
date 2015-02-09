#ifndef NEURALNETWORKMATRIX_H
#define NEURALNETWORKMATRIX_H

#include"Population.h"
using namespace pop;//Population namespace


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
    //        {
    //            NeuralNetworkFeedForward n2;
    //            n2.addInputLayer(2);
    //            n2.addLayerFullyConnected(3);
    //            n2.addLayerFullyConnected(1);

    //            NeuralNetworkFullConnection3 neural;

    //            std::vector<unsigned int>  v_int;
    //            v_int.push_back(2);//
    //            v_int.push_back(3);
    //            v_int.push_back(1);
    //            neural.createNetwork(v_int);

    //            Vec<VecF32> v(4,VecF32(2));
    //            v(0)(0)=-1;v(0)(1)=-1;
    //            v(1)(0)= 1;v(1)(1)=-1;
    //            v(2)(0)=-1;v(2)(1)= 1;
    //            v(3)(0)= 1;v(3)(1)= 1;

    //            Vec<VecF32> vout(4,VecF32(1));
    //            vout(0)(0)=-1;
    //            vout(1)(0)= 1;
    //            vout(2)(0)= 1;
    //            vout(3)(0)=-1;
    //            neural._eta=0.001;


    //            std::vector<int> v_global_rand(4);
    //            for(unsigned int i=0;i<v_global_rand.size();i++)
    //                v_global_rand[i]=i;

    //            while(1==1){
    //                std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
    //                for(unsigned int K=0;K<v_global_rand.size();K++)
    //                {
    //                    int i = v_global_rand[K];
    //                    VecF32 out;
    //                    std::cout<<i<<std::endl;
    //                    neural.propagateFront(v(i),out);
    //                    //std::cout<<out<<std::endl;

    //                    neural.propagateBackFirstDerivate(v(i),vout(i));
    //                    neural.propagateFront(v(i),out);
    //                    std::cout<<out<<std::endl;


    //                    n2.propagateFront(v(i),out);
    //                    n2.propagateBackFirstDerivate(vout(i));
    //                    n2.learningFirstDerivate();
    //                    n2.propagateFront(v(i),out);
    //                    std::cout<<out<<std::endl;
    //                }
    //            }
    //            return;
    //        }
    NeuralNetworkFullyConnected network;


    Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST( "/home/vincent/Desktop/train-images.idx3-ubyte","/home/vincent/Desktop/train-labels.idx1-ubyte");
    Vec<Vec<Mat2UI8> > number_test =  TrainingNeuralNetwork::loadMNIST("/home/vincent/Desktop/t10k-images.idx3-ubyte","/home/vincent/Desktop/t10k-labels.idx1-ubyte");

    //    number_training.resize(10);
    //    number_test.resize(2);
    //    for(unsigned int i=0;i<number_training.size();i++){
    //        number_training(i).resize(400);
    //        number_test(i).resize(50);
    //    }



    double size_in=number_training(0)(0).getDomain()(0)*number_training(0)(0).getDomain()(1);
    std::cout<<"size trainings: "<<number_training(0).size()<<std::endl;
    std::string net_struc = "400_300_200_100";
    std::cout<<net_struc<<std::endl;
    Vec<unsigned int> v_layer;
    v_layer.push_back(size_in);
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
    n2.setLearningRate(0.01);


    Vec<VecF32> vtraining_in;
    Vec<VecF32> vtraining_out;


    double ratio = 1;
    TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,number_training(0)(0).getDomain(),NNLayerMatrix::Mass,NNLayerMatrix::MinusOneToOne,ratio);

    Vec<VecF32> vtest_in;
    Vec<VecF32> vtest_out;
    TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,number_training(0)(0).getDomain(),NNLayerMatrix::Mass,NNLayerMatrix::MinusOneToOne,1);

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
        if(network._eta<0.00001)
            network._eta = 0.0001f;
        std::cout<<i<<"\t"<<error_training*1./v_global_rand.size()<<"\t"<<error_test*1./vtest_in.size() <<"\t"<<network._eta <<std::endl;
        //        std::cout<<i<<"\t"<<error_training2*1./v_global_rand.size()<<std::endl;
    }
}
#endif
