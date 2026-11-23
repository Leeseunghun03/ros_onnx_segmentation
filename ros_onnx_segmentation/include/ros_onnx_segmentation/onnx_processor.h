#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

// tensor = raw * scale + pad
struct ImageInfo
{
    cv::Size raw_size;
    double scale_x;
    double scale_y;
    double pad_x;
    double pad_y;
};

class OnnxProcessor
{
public:
    virtual ~OnnxProcessor() = default;
    virtual bool init(rclcpp::Node::SharedPtr &node);

protected:
    virtual void ProcessOutput(std::vector<Ort::Value> &outputs, const ImageInfo &info,
                               const cv::Mat &image, const std_msgs::msg::Header &header) = 0;

    rclcpp::Node::SharedPtr _node;
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

    bool _letterbox = true;
    std::vector<const char *> _inputNamePtrs;
    std::vector<const char *> _outputNamePtrs;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _imageSub;
};
