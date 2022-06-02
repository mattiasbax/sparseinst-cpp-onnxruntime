#include <fstream>
#include <iostream>
#include <onnxruntime_c_api.h>
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

std::vector<float> preProcessFrame( const cv::Mat& inputFrame, const cv::Size& networkInputSize )
{
    cv::Mat frame;
    cv::resize( inputFrame, frame, networkInputSize ); // resize to network image size
    cvtColor( frame, frame, cv::COLOR_BGR2RGB );       // convert from bgr  to rgb
    frame = frame.reshape( 1, 1 );                     // flatten to 1D
    std::vector<float> imageData;
    frame.convertTo( imageData, CV_32FC1 ); // convert to float
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

    // TODO: Copy and deallocate names

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

enum class InputMode {
    Image,
    Video
};

} // namespace

int main( )
{
    ///////////// Create session
    constexpr bool useCuda = true;
    const std::string instanceName = "SparseInst";

    Ort::Env env( OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str( ) );
    Ort::SessionOptions sessionOptions;
    // sessionOptions.SetIntraOpNumThreads( 1 );
    // sessionOptions.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_DISABLE_ALL );
    OrtCUDAProviderOptions cudaOptions;
    cudaOptions.device_id = 0;
    sessionOptions.AppendExecutionProvider_CUDA( cudaOptions );
    Ort::Session session( env, L"C:\\Users\\mattias.backstrom\\models\\sparse_inst_r50_giam_aug_2b7d68.onnx", sessionOptions );
    ModelParameters mp = getModelParameters( session );

    ///////////// Prepare input
    constexpr int numChannels = 3;
    const cv::Size inputImageSize = cv::Size( 640, 640 );

    const std::string imageFilepath = "C:\\Users\\mattias.backstrom\\models\\dog.jpg";
    cv::VideoCapture video( "C:\\Users\\mattias.backstrom\\models\\Untitled_0000.mov" );
    InputMode inputMode = InputMode::Image;

    while ( 1 ) {
        cv::Mat inputFrame;
        if ( inputMode == InputMode::Video )
            video >> inputFrame;
        else
            inputFrame = cv::imread( imageFilepath );
        if ( inputFrame.empty( ) )
            break;

        std::vector<float> inputImageData = preProcessFrame( inputFrame, inputImageSize );
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtDeviceAllocator, OrtMemTypeDefault );
        const std::array<int64_t, 4> inputShape = { 1, inputImageSize.width, inputImageSize.height, numChannels };
        const Ort::Value inputTensor = Ort::Value::CreateTensor<float>( memoryInfo, inputImageData.data( ), inputImageData.size( ), inputShape.data( ), inputShape.size( ) );
        std::vector<Ort::Value> outputTensors = session.Run( Ort::RunOptions{ nullptr }, mp.inputNodeNames.data( ), &inputTensor, mp.numInputNodes, mp.outputNodeNames.data( ), mp.numOutputNodes );

        const Ort::Value& masks = outputTensors[ 0 ];
        const Ort::Value& scores = outputTensors[ 1 ];
        const Ort::Value& labels = outputTensors[ 2 ];

        const bool* masksData = masks.GetTensorData<bool>( );
        const float* scoresData = scores.GetTensorData<float>( );
        const int64_t* labelsData = labels.GetTensorData<int64_t>( );

        cv::Mat green( inputFrame.size( ), CV_8UC3, cv::Scalar( 0., 75., 0. ) );

        const int64_t maxNumberOfMasks = scores.GetTensorTypeAndShapeInfo( ).GetShape( ).at( 1 );
        const float confidenceThreshold = 0.6f;
        const std::vector<int> classesToDetect = { 0, 16 };
        std::vector<cv::Mat> filteredMasks;
        for ( int64_t idx = 0; idx < maxNumberOfMasks; ++idx ) {
            if ( ( scoresData[ idx ] ) < confidenceThreshold || std::find( classesToDetect.begin( ), classesToDetect.end( ), labelsData[ idx ] ) == classesToDetect.end( ) )
                continue;
            filteredMasks.push_back( cv::Mat( inputImageSize, CV_8UC1, (uchar*) masksData + inputImageSize.area( ) * idx ) );
            cv::Mat& mask = filteredMasks.back( );

            cv::resize( mask, mask, inputFrame.size( ) );
            cv::add( green, inputFrame, inputFrame, mask );

            std::vector<cv::Mat> contours;
            cv::findContours( mask, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );
            cv::drawContours( inputFrame, contours, -1, cv::Scalar{ 0., 175., 0. }, 1, cv::LINE_AA );
        }
        cv::imshow( "result", inputFrame );
        if ( inputMode != InputMode::Video ) {
            cv::waitKey( 0 );
            cv::imwrite( "C:\\Users\\mattias.backstrom\\data\\result.jpg", inputFrame );
            break;
        }
        cv::waitKey( 1 );
    }
    video.release( );
    cv::destroyAllWindows( );
    return 0;
}