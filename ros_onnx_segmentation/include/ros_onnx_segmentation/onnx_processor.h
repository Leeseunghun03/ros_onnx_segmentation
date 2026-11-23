#pragma once

#include <memory>
#include <string>
#include <vector>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace ros_onnx_segmentation
{

// tensor = raw * scale + pad
struct ImageInfo
{
    cv::Size raw_size;
    double scale_x;
    double scale_y;
    double pad_x;
    double pad_y;
};

class OnnxProcessor : public rclcpp::Node
{
public:
    OnnxProcessor(const std::string &name, const rclcpp::NodeOptions &options);

protected:
    bool init();
    void Start();

    virtual void ProcessOutput(std::vector<Ort::Value> &outputs, const ImageInfo &info,
                               const cv::Mat &image, const std_msgs::msg::Header &header) = 0;

    std::unique_ptr<Ort::Env> _env;
    std::unique_ptr<Ort::Session> _session;

    std::vector<std::string> _inputNames;
    std::vector<std::string> _outputNames;

    int _tensorWidth = 640;
    int _tensorHeight = 640;
    bool _debug = false;

private:
    void ProcessImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    cv::Mat Preprocess(const cv::Mat &image, ImageInfo &info) const;
    void AppendExecutionProvider(Ort::SessionOptions &options, const std::string &provider,
                                 int device_id, std::string trt_cache_dir);

    bool _letterbox = true;
    std::string _imageTopic;
    std::vector<const char *> _inputNamePtrs;
    std::vector<const char *> _outputNamePtrs;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _imageSub;
};

}  // namespace ros_onnx_segmentation
