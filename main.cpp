#include <iostream>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace
{
template <typename T> T vectorProduct( const std::vector<T>& v )
{
    return std::accumulate( v.begin( ), v.end( ), 1, std::multiplies<T>( ) );
}
} // namespace

/*
void printInfo( const Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator )
{
    size_t numInputNodes = session.GetInputCount( );
    size_t numOutputNodes = session.GetOutputCount( );
    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
    const char* inputName = session.GetInputName( 0, allocator );
    std::cout << "Input Name: " << inputName << std::endl;
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo( 0 );
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo( );
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType( );
    std::cout << "Input Type: " << inputType << std::endl;
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape( );
    std::cout << "Input Dimensions: ( ";
    for ( const auto& i : inputDims ) {
        std::cout << i << " ";
    }
    std::cout << ")" << std::endl;

    std::vector<const char*> outputNames;
    for ( size_t idx = 0; idx < numOutputNodes; ++idx ) {
        const char* outputName = session.GetOutputName( idx, allocator );
        std::cout << "Output Name: " << outputName << std::endl;
        outputNames.push_back( outputName );
    }
}
*/
namespace
{

std::vector<float> loadImage( const std::string& filename, cv::Size size )
{
    cv::Mat image = cv::imread( filename );
    if ( image.empty( ) ) {
        std::cout << "No image found.";
    }

    // convert from BGR to RGB
    cv::cvtColor( image, image, cv::COLOR_BGR2RGB );

    // resize
    cv::resize( image, image, size );

    // reshape to 1D
    image = image.reshape( 1, 1 );

    // uint_8, [0, 255] -> float, [0, 1]
    // Normailze number to between 0 and 1
    // Convert to vector<float> from cv::Mat.
    std::vector<float> vec;
    image.convertTo( vec, CV_32FC1, 1. / 255 );
    return vec;
}

struct ModelParameters {
    size_t numInputNodes;
    std::vector<const char*> inputNodeNames;
    std::vector<std::vector<int64_t>> inputNodesDimensions;
    size_t numOutputNodes;
    std::vector<const char*> outputNodeNames;
    std::vector<std::vector<int64_t>> outputNodesDimensions;
};

ModelParameters getModelParameters( const Ort::Session& session, bool printInfo = false )
{
    ModelParameters mp;

    Ort::AllocatorWithDefaultOptions allocator; // TODO: Free memory
    mp.numInputNodes = session.GetInputCount( );
    for ( size_t idx = 0; idx < mp.numInputNodes; ++idx ) {
        mp.inputNodeNames.push_back( session.GetInputName( idx, allocator ) );

        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo( idx );
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo( );
        mp.inputNodesDimensions.emplace_back( inputTensorInfo.GetShape( ) );
    }

    mp.numOutputNodes = session.GetOutputCount( );
    for ( size_t idx = 0; idx < mp.numOutputNodes; ++idx ) {
        mp.outputNodeNames.push_back( session.GetOutputName( idx, allocator ) );

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo( idx );
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo( );
        mp.outputNodesDimensions.emplace_back( outputTensorInfo.GetShape( ) );
    }

    if ( !printInfo )
        return mp;

    std::cout << "Number of Input Nodes: " << mp.numInputNodes << std::endl;
    for ( size_t idx = 0; idx < mp.numInputNodes; ++idx ) {
        std::cout << "Input Name " << idx << ": " << mp.inputNodeNames[ idx ] << std::endl;
        std::cout << "Input " << idx << " Dimensions: ( ";
        for ( const auto& i : mp.inputNodesDimensions[ idx ] ) {
            std::cout << i << " ";
        }
        std::cout << ")" << std::endl;
    }

    std::cout << "Number of Output Nodes: " << mp.numOutputNodes << std::endl;
    for ( size_t idx = 0; idx < mp.numOutputNodes; ++idx ) {
        std::cout << "Output Name " << idx << ": " << mp.outputNodeNames[ idx ] << std::endl;
        std::cout << "Output " << idx << " Dimensions: ( ";
        for ( const auto& i : mp.outputNodesDimensions[ idx ] ) {
            std::cout << i << " ";
        }
        std::cout << ")" << std::endl;
    }

    return mp;
}

} // namespace

int main( )
{
    ///////////// Create session
    constexpr bool useCuda = true;
    const std::string instanceName = "SparseInst";

    const std::string imageFilepath = "C:\\Users\\mattias.backstrom\\models\\outframe.jpg";

    Ort::Env env( OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str( ) );
    Ort::Session session( env, L"C:\\Users\\mattias.backstrom\\models\\sparse_inst_r50_giam_aug_2b7d68.onnx", Ort::SessionOptions{ nullptr } );

    ModelParameters mp = getModelParameters( session, true );

    ///////////// Prepare input
    // const cv::Mat imageBGR = cv::imread( imageFilepath );
    std::vector<float> imageData = loadImage( imageFilepath, cv::Size( 640, 640 ) );
    const size_t inputTensorSize = 640 * 640 * 3;
    std::array<float, inputTensorSize>* pInput = new std::array<float, inputTensorSize>;
    /////////////

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtDeviceAllocator, OrtMemTypeDefault );
    const std::array<int64_t, 4> inputShape = { 1, 640, 640, 3 };

    auto inputTensor = Ort::Value::CreateTensor<float>( memoryInfo, pInput->data( ), pInput->size( ), inputShape.data( ), inputShape.size( ) );
    std::vector<Ort::Value> outputTensors = session.Run( Ort::RunOptions{ nullptr }, mp.inputNodeNames.data( ), &inputTensor, mp.numInputNodes, mp.outputNodeNames.data( ), mp.numOutputNodes );
    delete pInput;

    const Ort::Value& masks = outputTensors[ 0 ];
    const Ort::Value& scores = outputTensors[ 1 ];
    const Ort::Value& labels = outputTensors[ 2 ];

    try {
        const float* scoresData = scores.GetTensorData<float>( );
        const int64_t* labelsData = labels.GetTensorData<int64_t>( );
        for ( int idx = 0; idx < 50; ++idx ) {
            std::cout << "Data: " << scoresData[ idx ] << ", " << labelsData[ idx ] << std::endl;
        }
    } catch ( const std::exception& e ) {
        std::cerr << e.what( ) << '\n';
    }

    /*
        bool hasValue;
        for ( const auto& output : outputTensors ) {
            hasValue = output.HasValue( );
            if ( hasValue ) {
                // std::cout << "has value!" << std::endl;
                auto tensorTypeAndShape = output.GetTypeInfo( ).GetTensorTypeAndShapeInfo( );
                std::vector<int64_t> outputSize = tensorTypeAndShape.GetShape( );
                std::cout << "Output Size Dimensions: ( ";
                for ( const auto& i : outputSize ) {
                    std::cout << i << " ";
                }
                std::cout << ")" << std::endl;
            } else {
                std::cout << "Has no value!" << std::endl;
            }
        }
        */
    return 0;
}