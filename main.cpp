#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace
{

std::vector<float> loadImage( const std::string& filename, cv::Size size )
{
    cv::Mat image = cv::imread( filename );      // read image
    cvtColor( image, image, cv::COLOR_BGR2RGB ); // convert from bgr  to rgb
    cv::resize( image, image, size );            // resize to network image size
    image = image.reshape( 1, 1 );               // flatten to 1D
    std::vector<float> imageData;
    image.convertTo( imageData, CV_32FC1, 1. / 255. ); // convert to float and scale
    return imageData;
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

    const std::string imageFilepath = "C:\\Users\\mattias.backstrom\\models\\inframe.jpg";

    Ort::Env env( OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str( ) );
    Ort::Session session( env, L"C:\\Users\\mattias.backstrom\\models\\sparse_inst_r50_giam_aug_2b7d68.onnx", Ort::SessionOptions{ nullptr } );

    ModelParameters mp = getModelParameters( session, true );

    ///////////// Prepare input
    constexpr int numChannels = 3;
    const cv::Size inputImageSize = cv::Size( 640, 640 );
    std::vector<float> inputImageData = loadImage( imageFilepath, inputImageSize );
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtDeviceAllocator, OrtMemTypeDefault );
    const std::array<int64_t, 4> inputShape = { 1, inputImageSize.width, inputImageSize.height, numChannels };
    auto inputTensor = Ort::Value::CreateTensor<float>( memoryInfo, inputImageData.data( ), inputImageData.size( ), inputShape.data( ), inputShape.size( ) );

    ///////////// Run inference and inspect data

    const std::vector<Ort::Value> outputTensors = session.Run( Ort::RunOptions{ nullptr }, mp.inputNodeNames.data( ), &inputTensor, mp.numInputNodes, mp.outputNodeNames.data( ), mp.numOutputNodes );

    const Ort::Value& masks = outputTensors[ 0 ];
    const Ort::Value& scores = outputTensors[ 1 ];
    const Ort::Value& labels = outputTensors[ 2 ];

    const float* scoresData = scores.GetTensorData<float>( );
    const int64_t* labelsData = labels.GetTensorData<int64_t>( );
    for ( int idx = 0; idx < 50; ++idx ) {
        std::cout << "Data: " << scoresData[ idx ] << ", " << labelsData[ idx ] << std::endl;
    }

    return 0;
}