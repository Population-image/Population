#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <vector>
#include "data/vec/Vec.h"
#include "data/mat/MatN.h"
#include "data/notstable/MatNReference.h"
#include"algorithm/GeometricalTransformation.h"
#include"data/utility/XML.h"

namespace pop {

/*! \defgroup NeuralNetwork NeuralNetwork
 *  \ingroup Other
 *  \brief Layer neural network with backpropagation training
 *
 * For an introduction of neural network, you can read this <a href="http://www.dkriesel.com/en/science/neural_networks">book</a> .\n
 * The neural network is a  multi-layer neural network. The code is quite simple and gives nice result in character recognition. It is not deep learning since
 * GPU optimization is available (work in progress).
 *
 *
 * In the following example, we train a neural network to reproduce a XOR gate. The neural network has one hidden fully connected layer with 5 neurons.
 * \code
        NeuralNet net;
        net.addLayerLinearInput(2);//2 scalar input
        net.addLayerLinearFullyConnected(5);// 1 fully connected layer with 3 neurons
        net.addLayerLinearFullyConnected(1);// 1 scalar output
        //create the training set
        // (-1,-1)->-1
        // ( 1,-1)-> 1
        // (-1, 1)-> 1
        // ( 1, 1)->-1
        Vec<VecF32> v_input(4,VecF32(2));//4 vector of two scalar values
        v_input(0)(0)=-1;v_input(0)(1)=-1; // (-1,-1)
        v_input(1)(0)= 1;v_input(1)(1)=-1; // ( 1,-1)
        v_input(2)(0)=-1;v_input(2)(1)= 1; // (-1, 1)
        v_input(3)(0)= 1;v_input(3)(1)= 1; // ( 1, 1)

        Vec<VecF32> v_output_expected(4,VecF32(1));//4 vector of one scalar value
        v_output_expected(0)(0)=-1;// -1
        v_output_expected(1)(0)= 1;//  1
        v_output_expected(2)(0)= 1;//  1
        v_output_expected(3)(0)=-1;// -1


        //use the backprogation algorithm with first order method

        net.setLearnableParameter(0.1);
        net.setTrainable(true);


        //random vector to shuffle the trraining set
        Vec<int> v_global_rand(v_input.size());
        for(unsigned int i=0;i<v_global_rand.size();i++)
            v_global_rand[i]=i;

        std::cout<<"iter_epoch\t error_train\t  learning rate"<<std::endl;
        unsigned int nbr_epoch=100;
        for(unsigned int i=0;i<nbr_epoch;i++){
            std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
            F32 error_training=0;
            for(unsigned int j=0;j<v_global_rand.size();j++){
                VecF32 vout;
                net.forwardCPU(v_input(v_global_rand[j]),vout);
                net.backwardCPU(v_output_expected(v_global_rand[j]));
                net.learn();
                error_training+=std::abs(v_output_expected(v_global_rand[j])(0)-vout(0));
            }
            std::cout<<i<<"\t"<<error_training<<"\t"<<std::endl;
        }
        //test the training
        for(int j=0;j<4;j++){
            VecF32 vout;
            net.forwardCPU(v_input(j),vout);
            std::cout<<vout(0)<<std::endl;// we obtain the expected value -1 , 1 , 1 , -1
        }
 * \endcode
 *
 *  In MNISTNeuralNetLeCun5
*/


class NeuralLayer
{
public:
    virtual ~NeuralLayer();
    /** @brief Using the CPU device, compute the output values . */
    virtual void forwardCPU(const NeuralLayer& layer_previous) = 0;
    /** @brief Using the CPU device, compute the error of the output values of the layer prrevious. */
    virtual void backwardCPU(NeuralLayer& layer_previous) = 0;
    virtual void learn()=0;
    /** @brief get output value */
    virtual const VecF32& X()const=0;
    virtual VecF32& X()=0;
    /** @brief get output value */
    virtual VecF32& d_E_X()=0;
    /** @brief set the layer to be trainable */
    virtual void setTrainable(bool istrainable)=0;
    void setLearnableParameter(F32 mu);
    virtual NeuralLayer * clone()=0;
    //XXX TODO: need a destructor in order to avoid an undefined behaviour
    //virtual ~NeuralLayer()=0;
    F32 _mu;
};

struct NeuronSigmoid
{
    inline F32 activation(F32 y){ return 1.7159f*tanh(0.66666667f*y);}
    inline F32 derivedActivation(F32 x){ return 0.666667f/1.7159f*(1.7159f+(x))*(1.7159f-(x));}  // derivative of the sigmoid as a function of the sigmoid's output

};

struct NeuralLayerLinear : public NeuralLayer
{
    NeuralLayerLinear(unsigned int nbr_neurons);
    VecF32& X();
    const VecF32& X()const;
    VecF32& d_E_X();
    virtual void setTrainable(bool istrainable);

    VecF32 __Y;
    VecF32 __X;
    VecF32 _d_E_Y;
    VecF32 _d_E_X;
};

class NeuralLayerMatrix : public NeuralLayerLinear
{
public:
    NeuralLayerMatrix(unsigned int sizei,unsigned int sizej,unsigned int nbr_map);
    const Vec<MatNReference<2,F32> > & X_map()const;
    Vec<MatNReference<2,F32> >& X_map();
    const Vec<MatNReference<2,F32> > & d_E_X_map()const;
    Vec<MatNReference<2,F32> >& d_E_X_map();
    virtual void setTrainable(bool istrainable);

    Vec<MatNReference<2,F32> > _X_reference;
    Vec<MatNReference<2,F32> > _Y_reference;
    Vec<MatNReference<2,F32> > _d_E_X_reference;
    Vec<MatNReference<2,F32> > _d_E_Y_reference;


};

class NeuralLayerLinearInput : public NeuralLayerLinear
{
public:

    NeuralLayerLinearInput(unsigned int nbr_neurons);
    void forwardCPU(const NeuralLayer& );
    void backwardCPU(NeuralLayer& ) ;
    void learn();
    void setTrainable(bool istrainable);
    virtual NeuralLayer * clone();
};

class NeuralLayerMatrixInput : public NeuralLayerMatrix
{
public:

    NeuralLayerMatrixInput(unsigned int sizei,unsigned int sizej,unsigned int nbr_map);
    void forwardCPU(const NeuralLayer& ) ;
    void backwardCPU(NeuralLayer& ) ;
    void learn();
    void setTrainable(bool istrainable);
    virtual NeuralLayer * clone();
};

class NeuralLayerLinearFullyConnected : public NeuronSigmoid,public NeuralLayerLinear
{
public:
    NeuralLayerLinearFullyConnected(unsigned int nbr_neurons_previous,unsigned int nbr_neurons);
    void setTrainable(bool istrainable);
    virtual void forwardCPU(const NeuralLayer& layer_previous);
    virtual void backwardCPU(NeuralLayer& layer_previous);
    void learn();
    virtual NeuralLayer * clone();
    Mat2F32 _W;
    VecF32 _X_biais;
    Mat2F32 _d_E_W;
};

class NeuralLayerMatrixConvolutionSubScaling : public NeuronSigmoid,public NeuralLayerMatrix
{
public:
    NeuralLayerMatrixConvolutionSubScaling(unsigned int nbr_map,unsigned int sub_scaling_factor,unsigned int radius_kernel,unsigned int sizei_map_previous,unsigned int sizej_map_previous,unsigned int nbr_map_previous);
    void setTrainable(bool istrainable);

    virtual void forwardCPU(const NeuralLayer& layer_previous);
    virtual void backwardCPU(NeuralLayer& layer_previous);
    void learn();
    virtual NeuralLayer * clone();
    Vec<Mat2F32> _W_kernels;
    Vec<F32> _W_biais;
    Vec<Mat2F32> _d_E_W_kernels;
    Vec<F32> _d_E_W_biais;
    unsigned int _sub_resolution_factor;
    unsigned int _radius_kernel;
};
class NeuralLayerMatrixMaxPool : public NeuronSigmoid,public NeuralLayerMatrix
{
public:
    NeuralLayerMatrixMaxPool(unsigned int sub_scaling_factor,unsigned int sizei_map_previous,unsigned int sizej_map_previous,unsigned int nbr_map_previous);
    void setTrainable(bool istrainable);

    virtual void forwardCPU(const NeuralLayer& layer_previous);
    virtual void backwardCPU(NeuralLayer& layer_previous);
    void learn();
    virtual NeuralLayer * clone();
    unsigned int _sub_resolution_factor;
    bool _istrainable;
    Vec<Mat2UI8> _v_map_index_max;
};

class NormalizationMatrixInput
{
public:
    enum NormalizationValue{
        MinusOneToOne=0,
        ZeroToOne=1
    };

    virtual ~NormalizationMatrixInput();

    virtual NormalizationMatrixInput *clone()=0;
    void save(XMLNode & node)const;
    static NormalizationMatrixInput* load(const XMLNode & node);
    virtual VecF32 inputMatrixToInputNeuron(const Mat2UI8  & matrix,Vec2I32 domain)=0;
};
class NormalizationMatrixInputMass : public NormalizationMatrixInput
{
public:
    NormalizationMatrixInputMass(NormalizationMatrixInput::NormalizationValue normalization=NormalizationMatrixInput::MinusOneToOne);
    VecF32 inputMatrixToInputNeuron(const Mat2UI8  & matrix,Vec2I32 domain);
    NormalizationMatrixInputMass *clone();
    NormalizationValue _normalization_value;
};
class NormalizationMatrixInputCentering : public NormalizationMatrixInput
{
public:
    NormalizationMatrixInputCentering(NormalizationMatrixInput::NormalizationValue normalization=NormalizationMatrixInput::MinusOneToOne);
    VecF32 inputMatrixToInputNeuron(const Mat2UI8  & matrix,Vec2I32 domain);
    NormalizationMatrixInputCentering *clone();
    NormalizationValue _normalization_value;
};


class POP_EXPORTS NeuralNet
{
public:
    /*!
     * \class pop::NeuralNet
     * \ingroup NeuralNetwork
     * \brief neural network organised in feedforward topology
     * \author Tariel Vincent
     *
     *  The neurons are grouped in the following layers: One input layer, n-hidden processing layers and one output layer. Each neuron in one layer has only directed connections
     *  to the neurons of the next layer.
     */

    /*!
     * default constructor
     */
    NeuralNet();
    /*!
     * copy constructor
     */
    NeuralNet(const NeuralNet & neural);
    NeuralNet & operator =(const NeuralNet & neural);
    /*!
     * destructor
     */
    virtual ~NeuralNet();
    /*!
     * \brief add linear input layer
     * \param number of input neurons
     *
     */
    void addLayerLinearInput(unsigned int nbr_neurons);
    /*!
     * \brief add matrix input layer (multiple maps)
     * \param size_i number of rows
     * \param size_j number of columns
     * \param nbr_map number of input maps
     *
     *
     * add input layer with a matrix of neurons (the number of neuron is equal to height*width*nbr_map). You must use this input layer if you add convolutional layers after.
     *
     */

    void addLayerMatrixInput(unsigned int size_i,unsigned int size_j,unsigned int nbr_map);
    /*!
     * \brief  add a fully connected layer
     * \param nbr_neurons number of neurons
     *
     */
    void addLayerLinearFullyConnected(unsigned int nbr_neurons);

    /*!
     * \brief  add a convolutionnal layer
     * \param nbr_map number of maps (matrices)
     * \param radius_kernel radius of the convolutionnal kernel (1=3*3 kernel size
     * \param sub_scale_sampling sub scaling factor for Simard network (
     *
     *
     * add a convolutionnal layer with Feature maps and a sub scaling
     *
     */
    void addLayerMatrixConvolutionSubScaling(unsigned int nbr_map,unsigned int sub_scaling_factor=1,unsigned int radius_kernel=1);
    /*!
     * \brief add max pool layer
     * \param sub_scale_sampling sub scaling factor
     *
     */
    void addLayerMatrixMaxPool(unsigned int sub_scaling_factor=2);

    /*!
     * \brief set learnable paramater for the Newton's method
     * \param mu sub mu parameter
     *
     */
    void setLearnableParameter(F32 mu);
    /*!
     * \brief set trainable at true to create the data-structures (error) associated to the learning process
     * \param mu sub mu parameter
     *
     */
    void setTrainable(bool istrainable);

    /*!
     * \brief propagate front (feed-froward neural network)
     * \param  X_in input values
     * \param  X_out output values
     *
     * The outputs of the neurons of the input layer are updated with the input values \sa X_in,
     * then the propagation and activation of the neurons of layer-by-layer until the output layer. We set the
     * the output values \sa X_out with the outputs of the neurons of the output layer.
     *
     */
    void forwardCPU(const VecF32& X_in,VecF32 & X_out);
    /*!
     * \brief
     * \param  X_expected desired output value
     *
     *  In supervised learning algorithm, we want to find a function that best maps a set of inputs to its correct output.
     *  As explained by LeCun, in neural network, to find this function,
     *  we iterate a training procedure based of the back propagation of the error function. First we propagate one input generating a given output
     * \code
     * n.forwardCPU(vin,vout);
     * \endcode
     * Then, in this method, we compare this given output with a desired output to define a mean square error for this output function following by the back propagation of this error
     *  function layer-by-layer until the input layer.
     * \sa learn
     *
     *
     */
    void backwardCPU(const VecF32 &X_expected);
    /*!
     * \brief learn after the accumumation of the error for the weight
     *
     */
    void learn();

//    std::pair<Vec2I32,int> getDomainMatrixInput()const;
//    std::pair<Vec2I32,int> getDomainMatrixOutput()const;
//    MatNReference<2,F32>&  getMatrixOutput(int map_index)const;

    /*!
     * \brief set the normalization algorithm for the generation of the input values from a matrix
     *
     */
    void setNormalizationMatrixInput(NormalizationMatrixInput * input);

    /*!
     * \brief get the input values for a matrix
     *
     */
    VecF32 inputMatrixToInputNeuron(const Mat2UI8  & matrix);

    /*!
     * \brief clear the network
     *
     */
    void clear();

    /*!
    * \brief load xml file
    * \param file input file
    *
    * The loader attempts to read the neural network in the given file.
    */
    void load(const char * file);
    /*!
    * \brief load byte arrray
    * \param file input file
    *
    */
    void loadByteArray(const char *  file);

    void load(XMLDocument &doc);
    /*!
    * \brief save xml file
    * \param file output file
    *
    */
    void save(const char * file)const;

    /*!
    * \brief access information related to each output neuron (for instance "A","B","C","D",... for Latin script)
    * \return vector of strings
    *
    */
    const Vec<std::string>& label2String()const;
    /*!
    * \brief access information related to each output neuron (for instance "A","B","C","D",... for Latin script)
    * \return vector of strings
    *
    */
    Vec<std::string>& label2String();
    const Vec<NeuralLayer*>& layers()const;
    Vec<NeuralLayer*>& layers();

private:
    Vec<std::string> _label2string;
    Vec<NeuralLayer*> _v_layer;
    NormalizationMatrixInput * _normalizationmatrixinput;
};


struct MNISTNeuralNetLeCun5{
    static Mat2UI8 elasticDeformation(const Mat2UI8 &m, F32 sigma,F32 alpha);
    static Mat2UI8 affineDeformation(const Mat2UI8 &m, F32 max_rotation_angle_random,F32 max_shear_angle_random,F32 max_scale_vertical_random,F32 max_scale_horizontal_random);
    static NeuralNet createNet(std::string train_datapath,  std::string train_labelpath, std::string test_datapath,  std::string test_labelpath, UI32 nbr_epoch=10, UI32 lecun_or_simard=0, UI32 nbr_deformation=3, bool iselastic=false);
    static Vec<Vec<Mat2UI8> > loadMNIST( std::string datapath,  std::string labelpath);

};


}

#endif // NEURALNETWORK_H
