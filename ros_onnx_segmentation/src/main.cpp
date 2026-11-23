#include <rclcpp/rclcpp.hpp>

#include <ros_onnx_segmentation/yolo_processor.h>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    int ret = 0;
    try
    {
        rclcpp::spin(std::make_shared<ros_onnx_segmentation::YoloProcessor>());
    }
    catch (const std::exception &)
    {
        ret = 1;
    }

    rclcpp::shutdown();
    return ret;
}
