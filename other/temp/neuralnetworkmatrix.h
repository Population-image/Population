#ifndef NEURALNETWORKMATRIX_H
#define NEURALNETWORKMATRIX_H

#include"Population.h"
using namespace pop;//Population namespace


class NeuralNetworkFullConnection
{
public:
    double sigmoid(double x){ return 1.7159*tanh(0.66666667*x);}
    double derived_sigmoid(double S){ return 0.66666667/1.7159*(1.7159+(S))*(1.7159-(S));}  // derivative of the sigmoid as a function of the sigmoid's output


    void createNetwork(unsigned int input_neurons,unsigned int output_neurons,unsigned int hidden_neurons,unsigned int time){
        _time = time;
        _input_neurons  = input_neurons;
        _output_neurons = output_neurons;
        _hidden_neurons = hidden_neurons;
        _X.resize(_input_neurons+_output_neurons+_hidden_neurons);
        _Y.resize(_input_neurons+_output_neurons+_hidden_neurons);
        _R.resize(_input_neurons+_output_neurons+_hidden_neurons,_input_neurons+_output_neurons+_hidden_neurons);

        DistributionNormal n(0,1./std::sqrt(_input_neurons+_output_neurons+_hidden_neurons));

        for(unsigned int i = 0;i<_R.size();i++){
            _R[i]=n.randomVariable();
        }
    }

    void propagateFront(const pop::VecF64& in , pop::VecF64 &out){
        std::cout<<"R "<<Analysis::standardDeviationValue(_R)*std::sqrt(_R.sizeI())<<std::endl;
        //        std::cout<<"R "<<_R<<std::endl;
        for(unsigned int j=0;j<_X.size();j++){
            if(j<in.size()){
                _X[j]=in[j];
            }else{
                _X[j]=0;
            }
        }
        for(unsigned int i=0;i<_time;i++){
            _Y = _R * _X;
            for(unsigned int j=0;j<_Y.size();j++){
                _X[j] = sigmoid(_Y[j]);
            }
        }
        std::copy(_X.end()-_output_neurons,_X.end(),out.begin());
    }

    void propagateBackFirstDerivate(const pop::VecF64& in ,const pop::VecF64& desired_output){
        if(_X_time.size()!=_time+1){
            _X_time.resize(_time+1,_X);
            _Y_time.resize(_time+1,_Y);
            _X_delta_error.resize(_X.size());
            _Y_delta_error.resize(_Y.size());
            _R_delta_error=_R;
            _R_delta_error=0;
        }

        //propagation
        for(unsigned int j=0;j<_X_time[0].size();j++){
            if(j<in.size()){
                _X_time[0][j]=in[j];
            }else{
                _X_time[0][j]=0;
            }
        }

        for(unsigned int time_index=0;time_index<_time;time_index++){
            _Y_time[time_index] = _R * _X_time[time_index];
            for(unsigned int j=0;j<_Y.size();j++){
                _X_time[time_index+1][j] = sigmoid(_Y_time[time_index][j]);
            }
        }


        //back propagation error
        for( int time_index=_time;time_index>=0;time_index--){
            if(time_index==_time){
                for(unsigned int j=0;j<_X.size();j++){
                    if(j<_X.size()-desired_output.size())
                        _X_delta_error[j] = 0;
                    else
                        _X_delta_error[j] = (_X_time[time_index][j]-desired_output[j-(_X.size()-desired_output.size())]);
                }
            }else{
                for(unsigned int j=0;j<_X.size();j++){
                    _X_delta_error[j]=0;
                    for(unsigned int k=0;k<_X.size();k++){
                        _X_delta_error[j]+=_R(k,j)*_Y_delta_error[k];
                    }
                }
            }
            for(unsigned int j=0;j<_X.size();j++){
                _Y_delta_error[j] = _X_delta_error[j] * derived_sigmoid(_Y_time[time_index][j]);
            }
            for(unsigned int j=0;j<_X.size();j++){
                for(unsigned int k=0;k<_X.size();k++){
                    _R_delta_error(k,j)+=_Y_delta_error[k]*_X_time[time_index][j];
                }
            }
        }

        //error
        for(unsigned int j=0;j<_X.size();j++){
            for(unsigned int k=0;k<_X.size();k++){
                _R(k,j) = _R(k,j) - 0.0001/_time*_R_delta_error(k,j);
            }
        }
    }

    unsigned int _time;
    unsigned int _input_neurons;
    unsigned int _output_neurons;
    unsigned int _hidden_neurons;
    pop::VecF64  _X;
    pop::VecF64  _Y;
    pop::Mat2F64 _R;
    pop::Mat2F64 _R_delta_error;

    Vec<pop::VecF64> _X_time;
    Vec<pop::VecF64> _Y_time;
    pop::VecF64 _X_delta_error;
    pop::VecF64 _Y_delta_error;

};


class NeuralNetworkFullConnection2
{
public:
    double sigmoid(double x){ return 1.7159*tanh(0.66666667*x);}
    //    double derived_sigmoid(double S){ return 1.7159*(1+(0.66666667*S))*(1.7159-(0.66666667*S));}  // derivative of the sigmoid as a function of the sigmoid's output

    double derived_sigmoid(double S){ return 0.66666667/1.7159*(1.7159+(S))*(1.7159-(S));}  // derivative of the sigmoid as a function of the sigmoid's output


    void createNetwork(unsigned int input_neurons,unsigned int output_neurons,unsigned int hidden_neurons,unsigned int time){
        _time = time;
        _input_neurons  = input_neurons;
        _output_neurons = output_neurons;
        _hidden_neurons = hidden_neurons;
        //        int nbr_neurons = _input_neurons+_output_neurons+_hidden_neurons+1;
        int nbr_neurons = std::max(_input_neurons,_output_neurons)+_hidden_neurons;
        _X.resize(nbr_neurons);
        _Y.resize(nbr_neurons);

        DistributionNormal n(0,1./std::sqrt(nbr_neurons));
        for(unsigned int time_index=0;time_index<=_time;time_index++){
            Mat2F64 R(nbr_neurons,nbr_neurons);
            for(unsigned int i = 0;i<R.size();i++){
                R[i]=n.randomVariable();
            }
            _R.push_back(R);
        }
    }
    double max(){
        double maxi=0;
        for(unsigned int time_index=0;time_index<_R.size();time_index++){

            for(unsigned int j=0;j<_X.size();j++){
                for(unsigned int k=0;k<_X.size();k++){
                    maxi = std::max(std::abs(_R[time_index](j,k)),maxi);
                }
            }
        }
        return maxi;
    }

    void propagateFront(const pop::VecF64& in , pop::VecF64 &out){



        //        std::cout<<"R "<<Analysis::standardDeviationValue(_R[1])*std::sqrt(_R[1].sizeI())<<std::endl;
        //        std::cout<<"R "<<_R<<std::endl;
        for(unsigned int j=0;j<_X.size();j++){
            if(j<in.size())
                _X[j]=in[j];
            else
                _X[j]=0;

        }
        for(unsigned int time_index=0;time_index<_time;time_index++){
            _Y = _R[time_index+1] * _X;
            for(unsigned int j=0;j<_Y.size();j++){
                _X[j] = sigmoid(_Y[j]);
            }
        }
        std::copy(_X.begin(),_X.begin()+_output_neurons,out.begin());
    }
    double  error(const pop::VecF64& desired_output){
        double error=0;
        for(unsigned int j=0;j<desired_output.size();j++){
            error += std::abs(_X[j]-desired_output[j]);
        }
        return error;
    }

    void propagateBackFirstDerivate(const pop::VecF64& in ,const pop::VecF64& desired_output){
        if(_X_time.size()!=_time+1){
            _X_time.resize(_time+1,_X);
            _Y_time.resize(_time+1,_Y);
            _X_delta_error.resize(_X.size());
            _Y_delta_error.resize(_Y.size());
            _R_delta_error.resize(_Y.size(),pop::Mat2F64(_R[0].getDomain()));

        }

        //propagation
        for(unsigned int j=0;j<_X_time[0].size();j++){
            if(j<in.size())
                _X_time[0][j]=in[j];
            else
                _X_time[0][j]=0;

        }
        for(unsigned int time_index=0;time_index<_time;time_index++){
            _Y_time[time_index+1] = _R[time_index+1] * _X_time[time_index];
            for(unsigned int j=0;j<_Y.size();j++){
                _X_time[time_index+1][j] = sigmoid(_Y_time[time_index+1][j]);
            }
        }

        //back propagation error
        for( int time_index=_time;time_index>0;time_index--){
            if(time_index==_time){
                for(unsigned int j=0;j<_X.size();j++){
                    if(j<_output_neurons){
                        _X_delta_error[j] = (_X_time[time_index][j]-desired_output[j]);
                        //                        error_before +=std::abs(_X_time[time_index][j]-desired_output[j]);
                    }else
                        _X_delta_error[j] = 0;
                }
            }
            for(unsigned int j=0;j<_X.size();j++){
                _Y_delta_error[j] = _X_delta_error[j] * derived_sigmoid(_X_time[time_index][j]);
            }
            //            _Y_delta_error[in.size()] = 0;
            for(unsigned int j=0;j<_X.size();j++){
                for(unsigned int i=0;i<_X.size();i++){
                    _R_delta_error[time_index](i,j)=_Y_delta_error[i]*_X_time[time_index-1][j];
                }
            }

            for(unsigned int j=0;j<_X.size();j++){
                _X_delta_error[j]=0;
                for(unsigned int i=0;i<_X.size();i++){
                    _X_delta_error[j]+=_R[time_index](i,j)*_Y_delta_error[i];
                }
                //_X_delta_error[j] = _X_delta_error[j]/_X.size();
            }
            for(unsigned int j=0;j<_X.size();j++){
                for(unsigned int k=0;k<_X.size();k++){
                    _R[time_index](k,j) = _R[time_index](k,j) - _eta*_R_delta_error[time_index](k,j);
                }
            }
        }

    }
    double _eta;
    unsigned int _time;

    unsigned int _input_neurons;
    unsigned int _output_neurons;
    unsigned int _hidden_neurons;
    pop::VecF64  _X;
    pop::VecF64  _Y;
    Vec<pop::Mat2F64> _R;
    Vec<pop::Mat2F64> _R_temp;
    Vec<pop::Mat2F64>_R_delta_error;

    Vec<pop::VecF64> _X_time;
    Vec<pop::VecF64> _Y_time;
    pop::VecF64 _X_delta_error;
    pop::VecF64 _Y_delta_error;

};



class NeuralNetworkFullConnection3
{
public:
    double sigmoid(double x){ return 1.7159*tanh(0.66666667*x);}
    //    double derived_sigmoid(double S){ return 1.7159*(1+(0.66666667*S))*(1.7159-(0.66666667*S));}  // derivative of the sigmoid as a function of the sigmoid's output

    double derived_sigmoid(double S){ return 0.66666667/1.7159*(1.7159+(S))*(1.7159-(S));}  // derivative of the sigmoid as a function of the sigmoid's output


    void createNetwork(std::vector<unsigned int> v_layer){
        for(unsigned int i=0;i<v_layer.size();i++){
            int size_layer =v_layer[i];
            if(i!=v_layer.size()-1)
                _v_X.push_back(VecF64(size_layer+1,1));//add the neuron with constant value 1
            else
                _v_X.push_back(VecF64(size_layer,1));//except for the last one
            _v_Y.push_back(VecF64(size_layer));

            if(i!=0){
                int size_layer_previous = _v_X[i-1].size();
                Mat2F64  R(size_layer  ,size_layer_previous);
                DistributionNormal n(0,1./std::sqrt(size_layer_previous));
                for(unsigned int i = 0;i<R.size();i++){
                    R[i]=n.randomVariable();
                }
                _v_R.push_back(R);
            }

        }
    }

    void propagateFront(const pop::VecF64& in , pop::VecF64 &out){
        std::copy(in.begin(),in.end(),_v_X[0].begin());
        for(unsigned int layer_index=0;layer_index<_v_R.size();layer_index++){
            _v_Y[layer_index+1] = _v_R[layer_index] * _v_X[layer_index];
            for(unsigned int j=0;j<_v_Y[layer_index+1].size();j++){
                _v_X[layer_index+1][j] = sigmoid(_v_Y[layer_index+1][j]);
            }
        }
        if(out.size()!=_v_X.rbegin()->size())
            out.resize(_v_X.rbegin()->size());
        std::copy(_v_X.rbegin()->begin(),_v_X.rbegin()->begin()+out.size(),out.begin());
    }
    void propagateBackFirstDerivate(const pop::VecF64& in ,const pop::VecF64& desired_output){
        if(_v_X_error.size()==0){
            _v_X_error = _v_X;
            _v_Y_error = _v_Y;
            _v_R_error = _v_R;
        }

        propagateFront(in,_out);


        for( int index_layer=_v_X.size()-1;index_layer>0;index_layer--){
            //X error
            if(index_layer==_v_X.size()-1){
                for(unsigned int j=0;j<_v_X[index_layer].size();j++){
                    _v_X_error[index_layer][j] = (_v_X[index_layer][j]-desired_output[j]);
                }
            }
            for(unsigned int i=0;i<_v_Y_error[index_layer].size();i++){
                _v_Y_error[index_layer][i] = _v_X_error[index_layer][i] * derived_sigmoid(_v_X[index_layer][i]);
            }
            for(unsigned int j=0;j<_v_X[index_layer-1].size();j++){
                _v_X_error[index_layer-1][j]=0;
                for(unsigned int i=0;i<_v_R[index_layer-1].sizeI();i++){
                    _v_X_error[index_layer-1][j]+=_v_R[index_layer-1](i,j)*_v_Y_error[index_layer-1][i];
                }
                //v_X_error[index_layer-1][j]/=_v_X[index_layer-1].size();
            }

            for(unsigned int j=0;j<_v_R_error[index_layer-1].sizeJ();j++){
                for(unsigned int i=0;i<_v_R_error[index_layer-1].sizeI();i++){
                    _v_R_error[index_layer-1](i,j)= _v_Y_error[index_layer][i]*_v_X[index_layer-1][j];
                    _v_R[index_layer-1](i,j)= _v_R[index_layer-1](i,j) - _eta* _v_R_error[index_layer-1](i,j);
                }
            }
        }

    }
    double _eta;

    Vec<pop::VecF64>  _v_X;
    Vec<pop::VecF64>  _v_Y;
    Vec<pop::Mat2F64> _v_R;
    Vec<pop::VecF64>  _v_X_error;
    Vec<pop::VecF64>  _v_Y_error;
    Vec<pop::Mat2F64> _v_R_error;
    pop::VecF64 _out;

};




void neuralnetwortest(){
    //    {
    //        NeuralNetworkFullConnection3 neural;

    //        std::vector<unsigned int>  v_int;
    //        v_int.push_back(2);
    //        v_int.push_back(6);
    //        v_int.push_back(1);
    //        neural.createNetwork(v_int);

    //        Vec<VecF64> v(4,VecF64(2));
    //        v(0)(0)=-1;v(0)(1)=-1;
    //        v(1)(0)= 1;v(1)(1)=-1;
    //        v(2)(0)=-1;v(2)(1)= 1;
    //        v(3)(0)= 1;v(3)(1)= 1;

    //        Vec<VecF64> vout(4,VecF64(1));
    //        vout(0)(0)=-1;
    //        vout(1)(0)= 1;
    //        vout(2)(0)= 1;
    //        vout(3)(0)=-1;
    //        neural._eta=0.01;

    //        Distribution d;
    //        std::vector<int> v_global_rand(4);
    //        for(unsigned int i=0;i<v_global_rand.size();i++)
    //            v_global_rand[i]=i;

    //        while(1==1){
    //            std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,d.MTRand());
    //            for(unsigned int K=0;K<v_global_rand.size();K++)
    //            {
    //                int i = v_global_rand[K];
    //                VecF64 out;
    //                std::cout<<i<<std::endl;
    //                neural.propagateFront(v(i),out);
    //                std::cout<<out<<std::endl;

    //                neural.propagateBackFirstDerivate(v(i),vout(i));
    //                neural.propagateFront(v(i),out);
    //                std::cout<<out<<std::endl;
    //            }
    //        }
    //        return;
    //    }
    NeuralNetworkFullConnection3 network;


    Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST( "/home/vincent/train-images.idx3-ubyte","/home/vincent/train-labels.idx1-ubyte");
    Vec<Vec<Mat2UI8> > number_test =  TrainingNeuralNetwork::loadMNIST("/home/vincent/t10k-images.idx3-ubyte","/home/vincent/t10k-labels.idx1-ubyte");



    Vec<Vec<Mat2UI8> > number_training_augmented;
    for(unsigned int i=0;i<number_training.size();i++){

        std::cout<<i<<std::endl;
        Vec<Mat2UI8> augmented;
        if(i!=1||i!=7)
            augmented = TrainingNeuralNetwork::geometricalTransformationDataBaseMatrix(number_training(i));
        else
            augmented = TrainingNeuralNetwork::geometricalTransformationDataBaseMatrix(number_training(i),10,5,6,36,38,7.5,7.5);
        number_training_augmented.push_back(augmented);
    }
    number_training = number_training_augmented;
    //    TrainingNeuralNetwork::geometricalTransformationDataBaseMatrix(number_test(9));

    double size_in=number_training(0)(0).getDomain()(0)*number_training(0)(0).getDomain()(1);
    //    double size_hidden =100;
    //    double size_output =number_training.size();
    std::vector<unsigned int> v_layer;
    v_layer.push_back(size_in);
    v_layer.push_back(1000);
    v_layer.push_back(500);
    v_layer.push_back(10);
    network.createNetwork(v_layer);
    network._eta = 0.001;

    Vec<VecF64> vtraining_in;
    Vec<VecF64> vtraining_out;


    double ratio = 1;
    TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,number_training(0)(0).getDomain(),NNLayerMatrix::Mass,NNLayerMatrix::MinusOneToOne,ratio);




    Vec<VecF64> vtest_in;
    Vec<VecF64> vtest_out;
    TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,number_training(0)(0).getDomain(),NNLayerMatrix::Mass,NNLayerMatrix::MinusOneToOne,1);



    number_training.clear();
    number_test.clear();

    std::vector<int> v_global_rand(vtraining_in.size());
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i;

    std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;

    int iter =0;
    int display_error =1000;
    Distribution d;
    for(unsigned int i=0;i<100;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,d.MTRand());
        int error_training=0,error_test=0,error_intermediaire=0;

        for(unsigned int j=0;j<v_global_rand.size();j++){
            VecF64 vout(10);
            //            double error_previous,error_after;
            //            network.propagateFront(vtraining_in(v_global_rand[j]),vout);
            //            error_previous = network.error(vtraining_out(v_global_rand[j]));
            //            std::cout<<vout<<std::endl;
            network.propagateBackFirstDerivate(vtraining_in(v_global_rand[j]),vtraining_out(v_global_rand[j]));
            iter++;
            network.propagateFront(vtraining_in(v_global_rand[j]),vout);

            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
            if(label1!=label2){
                error_training++;
                error_intermediaire++;
            }
            if(iter%display_error==0){
                std::cout<<"fraction "<<1.*j/v_global_rand.size()<<std::endl;
                std::cout<<label2<<"   "<<vout<<std::endl;
                std::cout<<i<<"\t"<<error_intermediaire*1./display_error<<"\t"<<network._eta<<std::endl;
                //number_test=0;
                error_intermediaire=0;
            }
        }
        int k=rand()%vtest_in.size();
        for(unsigned int j=0;j<vtest_in.size();j++){

            VecF64 vout(10);
            network.propagateFront(vtest_in(j),vout);
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(vtest_out(j).begin(),std::max_element(vtest_out(j).begin(),vtest_out(j).end()));
            if(label1!=label2){
                if(j==k){
                    std::cout<<"test "<<label2<<"   "<<vout<<std::endl;
                    std::cout<<i<<"\t"<<error_intermediaire*1./display_error<<"\t"<<network._eta<<std::endl;
                }
                error_test++;
            }
        }
        network._eta *=0.9;
        std::cout<<i<<"\t"<<error_training*1./vtraining_in.size()<<"\t"<<error_test*1./vtest_in.size()<<"\t"<<network._eta<<std::endl;
    }



    //    Vec<VecF64> v_in(4,VecF64(2));//v2(2),v3(2),v4(2);
    //    v_in(0)(0)=-1;v_in(0)(1)=-1;
    //    v_in(1)(0)= 1;v_in(1)(1)=-1;
    //    v_in(2)(0)=-1;v_in(2)(1)= 1;
    //    v_in(3)(0)= 1;v_in(3)(1)= 1;


    //    //    v_in(0)(0)= 0;v_in(0)(1)= 0;
    //    //    v_in(1)(0)= 1;v_in(1)(1)= 0;
    //    //    v_in(2)(0)= 0;v_in(2)(1)= 1;
    //    //    v_in(3)(0)= 1;v_in(3)(1)= 1;

    //    Vec<VecF64> v_out(4,VecF64(1));//v2_out(1),v3_out(1),v4_out(1);
    //    v_out(0)(0)= -1;
    //    v_out(1)(0)= 1;
    //    v_out(2)(0)= 1;
    //    v_out(3)(0)= -1;

    //    //    v_out(0)(0)= 0;
    //    //    v_out(1)(0)= 1;
    //    //    v_out(2)(0)= 1;
    //    //    v_out(3)(0)= 0;

    //    DistributionUniformInt d_random(0,3);

    //    std::vector<int> v_global_rand(4);
    //    for(unsigned int i=0;i<v_global_rand.size();i++)
    //        v_global_rand[i]=i;

    //    Distribution d;



    //    while(1==1){
    //        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,d.MTRand());
    //        double error=0;
    //        for(unsigned int j=0;j<v_global_rand.size();j++){
    //            int i = v_global_rand[j];
    //            std::cout<<v_in(i)<<std::endl;
    //            VecF64 v_out_test(1);
    //            network.propagateFront(v_in(i), v_out_test);
    //            std::cout<<"before neural : "<<v_out_test(0)<<std::endl;

    //            network.propagateBackFirstDerivate(v_in(i),v_out(i));

    //            network.propagateFront(v_in(i), v_out_test);
    //            std::cout<<"after  neural : "<<v_out_test(0)<<std::endl;
    //            error+=std::abs(v_out(i)(0)-v_out_test(0));
    //            std::cout<<"expected: "<<v_out(i)(0)<<std::endl;
    //        }
    //        //        network.init();
    //        std::cout<<"error: "<<error<<std::endl;
    //    }



    //    network.propagateFront(v_in(0), v_out_test);




    //    VecF64 Xin(size_in);
    //    DistributionNormal n(0,1);
    //    std::cout<<n.randomVariable()<<std::endl;
    //    for(unsigned int i = 0;i<Xin.size();i++){
    //        Xin(i)=n.randomVariable();
    //    }


    //    VecF64 Xout(2);
    //    network.propagateFront(Xin,Xout);

    //    std::cout<<Xout<<std::endl;
}
#endif // ANALYSIS_H
