#ifndef NEURALNETWORKBLAS_H
#define NEURALNETWORKBLAS_H

#include"data/neuralnetwork/NeuralNetwork.h"

namespace popblas {

typedef pop::MatN<2, F32> BMat;

class NeuralLayer : public pop::NeuralLayer
{
public:
    virtual ~NeuralLayer();
    /** @brief get output value */
    virtual const BMat& X()const=0;
    virtual BMat& X()=0;
    /** @brief get the error output value */
    virtual BMat& d_E_X()=0;

    /** @brief old version */
    /** @brief get output value */
    inline const pop::VecF32& X()const {
        std::cerr << __FILE__ << "::" << __LINE__ << "calling old version X() of NeuralLayer" << std::endl;
        return pop::VecF32();
    }
    inline pop::VecF32& X() {
        std::cerr << __FILE__ << "::" << __LINE__ << "calling old version X() of NeuralLayer" << std::endl;
        return pop::VecF32();
    }

    /** @brief get the error output value */
    inline pop::VecF32& d_E_X() {
        std::cerr << __FILE__ << "::" << __LINE__ << "calling old version d_E_X() of NeuralLayer" << std::endl;
        return pop::VecF32();
    }
};

struct Softmax {
    void softmax(BMat& x);
};

struct NeuralLayerLinear : public NeuralLayer
{
    NeuralLayerLinear(unsigned int nbr_neurons);
    NeuralLayerLinear(const NeuralLayerLinear & net);
    NeuralLayerLinear&  operator=(const NeuralLayerLinear & net);
    BMat& X();
    const BMat& X()const;
    BMat& d_E_X();
    virtual void setTrainable(bool istrainable);
    virtual void print();
    BMat __Y;
    BMat __X;
    BMat _d_E_Y;
    BMat _d_E_X;
};

class NeuralLayerMatrix : public NeuralLayerLinear
{
public:
    NeuralLayerMatrix(unsigned int sizei,unsigned int sizej,unsigned int nbr_map);
    NeuralLayerMatrix(const NeuralLayerMatrix & net);
    NeuralLayerMatrix&  operator=(const NeuralLayerMatrix & net);
    const pop::Vec<BMat > & X_map()const;
    pop::Vec<BMat >& X_map();
    const pop::Vec<BMat > & d_E_X_map()const;
    pop::Vec<BMat >& d_E_X_map();
    virtual void setTrainable(bool istrainable);
    virtual void print();
    Vec<BMat > _X_reference;
    Vec<BMat > _Y_reference;
    Vec<BMat > _d_E_X_reference;
    Vec<BMat > _d_E_Y_reference;
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
    virtual void print();
    virtual void save(XMLNode& nodechild);
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
    virtual void print();
    virtual void save(XMLNode& nodechild);
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
    virtual void print();
    virtual void save(XMLNode& nodechild);
};

class NeuralLayerLinearFullyConnected : public pop::NeuronSigmoid,public NeuralLayerLinear
{
public:
    NeuralLayerLinearFullyConnected(unsigned int nbr_neurons_previous,unsigned int nbr_neurons);
    void setTrainable(bool istrainable);
    virtual void forwardCPU(const NeuralLayer& layer_previous);
    virtual void backwardCPU(NeuralLayer& layer_previous);
    virtual void learn();
    virtual NeuralLayer * clone();
    virtual void print();
    virtual void save(XMLNode& nodechild);
    BMat _W;
    BMat _X_biais;
    BMat _d_E_W;
};

class NeuralLayerLinearFullyConnectedSoftmax : public NeuralLayerLinearFullyConnected
{
public:
    NeuralLayerLinearFullyConnectedSoftmax(unsigned int nbr_neurons_previous,unsigned int nbr_neurons);
    virtual void forwardCPU(const NeuralLayer &layer_previous);
    virtual void backwardCPU(NeuralLayer& layer_previous);
    virtual void print();
    virtual void save(XMLNode& nodechild);
    Softmax _sm;
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
    virtual void print();
    virtual void save(XMLNode& nodechild);
    pop::Vec<BMat> _W_kernels;
    BMat _W_biais;
    pop::Vec<BMat> _d_E_W_kernels;
    BMat _d_E_W_biais;
    unsigned int _sub_resolution_factor;
    unsigned int _radius_kernel;
};

class NeuralLayerMatrixMaxPool : public NeuronSigmoid,public NeuralLayerMatrix
{
public:
    NeuralLayerMatrixMaxPool(unsigned int sub_scaling_factor,unsigned int sizei_map_previous,unsigned int sizej_map_previous,unsigned int nbr_map_previous);
    void setTrainable(bool istrainable);
    virtual void print();
    virtual void forwardCPU(const NeuralLayer& layer_previous);
    virtual void backwardCPU(NeuralLayer& layer_previous);
    void learn();
    virtual NeuralLayer * clone();
    virtual void save(XMLNode& nodechild);
    unsigned int _sub_resolution_factor;
    bool _istrainable;
    pop::Vec<pop::Mat2UI8> _v_map_index_max;
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
    virtual BMat inputMatrixToInputNeuron(const pop::Mat2UI8  & matrix, pop::Vec2I32 domain)=0;
    virtual void print()=0;
};

class NormalizationMatrixInputMass : public NormalizationMatrixInput
{
public:
    NormalizationMatrixInputMass(NormalizationMatrixInput::NormalizationValue normalization=NormalizationMatrixInput::MinusOneToOne);
    BMat inputMatrixToInputNeuron(const pop::Mat2UI8  & matrix, pop::Vec2I32 domain);
    NormalizationMatrixInputMass *clone();
    virtual void print();
    NormalizationValue _normalization_value;
};

class NormalizationMatrixInputCentering : public NormalizationMatrixInput
{
public:
    NormalizationMatrixInputCentering(NormalizationMatrixInput::NormalizationValue normalization=NormalizationMatrixInput::MinusOneToOne);
    BMat inputMatrixToInputNeuron(const pop::Mat2UI8  & matrix, pop::Vec2I32 domain);
    NormalizationMatrixInputCentering *clone();
    NormalizationValue _normalization_value;
    virtual void print();
};

class POP_EXPORTS NeuralNet
{
public:
    /*!
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
     * \brief  add an output fully connected layer with softmax
     * \param nbr_neurons number of neurons
     *
     */
    void addLayerLinearFullyConnectedSoftmax(unsigned int nbr_neurons);

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
    void setLearnableParameter(pop::F32 mu);
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
    void forwardCPU(const BMat& X_in, BMat& X_out);
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
    void backwardCPU(const BMat& X_expected);
    /*!
     * \brief learn after the accumumation of the error for the weight
     *
     */
    void learn();
    /*!
     * \brief set the normalization algorithm for the generation of the input values from a matrix
     *
     */
    void setNormalizationMatrixInput(NormalizationMatrixInput * input);

    /*!
     * \brief get the input values from a matrix
     *
     */
    VecF32 inputMatrixToInputNeuron(const pop::Mat2UI8  & matrix);

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
    * \brief save xml file
    * \param xml output file
    *
    */
    void save(XMLDocument& doc) const;

    /*!
    * \brief access information related to each output neuron (for instance "A","B","C","D",... for Latin script)
    * \return vector of strings
    *
    */
    const pop::Vec<std::string>& label2String()const;
    /*!
    * \brief access information related to each output neuron (for instance "A","B","C","D",... for Latin script)
    * \return vector of strings
    *
    */
    Vec<std::string>& label2String();
    const Vec<NeuralLayer*>& layers()const;
    Vec<NeuralLayer*>& layers();
    /*!
    * \brief print the network structure on the standart output
    */
    virtual void print();

private:
    pop::Vec<std::string> _label2string;
    pop::Vec<NeuralLayer*> _v_layer;
    NormalizationMatrixInput * _normalizationmatrixinput;
};

}

#endif // NEURALNETWORKBLAS_H

