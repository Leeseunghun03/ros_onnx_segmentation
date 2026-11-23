#include <rclcpp/rclcpp.hpp>

#include <ros_onnx_segmentation/yolo_processor.h>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("ros_onnx_segmentation");

    yolo::YoloProcessor processor;
    if (!processor.init(node))
    {
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
