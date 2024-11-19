#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <ros_onnx_segmentation/onnx_processor.h>
#include <ros_onnx_segmentation/yolo_processor.h>

#include <algorithm>
#include <numeric>
#include <functional>

namespace yolo
{
    float conf_thresh = 0.3;
    float mask_thresh = 0.5;
    float iou_thresh = 0.5;

    int seg_ch = 32;
    int seg_w = 160, seg_h = 160;

    const std::string kDefaultLabel = "person";
    std::vector<std::string> class_names =
        {
            "outside", "field", "line", "ball", "player"};

    YoloProcessor::YoloProcessor()
    {
        _normalize = false;
    }

    bool YoloProcessor::init(rclcpp::Node::SharedPtr &node)
    {
        OnnxProcessor::init(node);
        _outName = {"output0", "output1"};
        _inName = {"images"};
        _node->get_parameter_or("label", _label, kDefaultLabel);

        srand(time(0));
        for (int i = 0; i < class_names.size(); i++)
        {
            color.push_back(cv::Scalar(rand() % 128 + 128, rand() % 128 + 128, rand() % 128 + 128));
        }

        return true;
    }

    void YoloProcessor::ProcessOutput(const cv::Mat &output0, const cv::Mat &output1, const cv::Mat &raw_image)
    {
        ImageInfo img_info = {
            raw_image.size(),
            {static_cast<double>(_tensorWidth) / static_cast<double>(raw_image.cols),
             static_cast<double>(_tensorHeight) / static_cast<double>(raw_image.rows),
             0.0,
             0.0}};

        std::vector<Obj> seg_result;
        std::vector<int> class_ids;
        std::vector<float> accus;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> masks;

        std::map<int, int> class_count;

        int data_width = class_names.size() + 4 + 32;
        int rows = output0.rows;
        float *pdata = (float *)output0.data;

        for (int r = 0; r < rows; ++r)
        {
            cv::Mat scores(1, class_names.size(), CV_32FC1, pdata + 4);

            cv::Point class_id;
            double max_conf;
            cv::minMaxLoc(scores, 0, &max_conf, 0, &class_id);

            if (max_conf > conf_thresh)
            {
                masks.push_back(std::vector<float>(pdata + 4 + class_names.size(), pdata + data_width));
                float w = pdata[2] / img_info.trans[0];
                float h = pdata[3] / img_info.trans[1];
                int left = std::max(int((pdata[0] - img_info.trans[2]) / img_info.trans[0] - 0.5 * w + 0.5), 0);
                int top = std::max(int((pdata[1] - img_info.trans[3]) / img_info.trans[1] - 0.5 * h + 0.5), 0);
                class_ids.push_back(class_id.x);
                accus.push_back(max_conf);
                boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));

                class_count[class_id.x]++;
            }

            pdata += data_width;
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, accus, conf_thresh, iou_thresh, nms_result);

        // int target_class = 4; 

        for (int i = 0; i < nms_result.size(); ++i)
        {
            int idx = nms_result[i];
            // if (class_ids[idx] != target_class) // 원하는 클래스만 선택
            //     continue;
            boxes[idx] = boxes[idx] & cv::Rect(0, 0, img_info.raw_size.width, img_info.raw_size.height);
            Obj result = {class_ids[idx], accus[idx], boxes[idx]};
            get_mask(cv::Mat(masks[idx]).t(), output1, img_info, boxes[idx], result.mask);
            if (!result.mask.empty())
                seg_result.push_back(result);
        }

        cv::Mat output_image = raw_image.clone();
        if (seg_result.size() > 0)
        {
            draw_result(output_image, seg_result, color);
        }

        // std::cout << "Class-wise Object Count:" << std::endl;
        // for (const auto &entry : class_count)
        // {
        //     std::cout << "Class ID " << entry.first << ": " << entry.second << " objects" << std::endl;
        // }
    }

    void YoloProcessor::get_mask(const cv::Mat &mask_info, const cv::Mat &mask_data, const ImageInfo &para, cv::Rect bound, cv::Mat &mask_out)
    {
        cv::Vec4f trans = para.trans;
        int r_x = std::floor((bound.x * trans[0] + trans[2]) / _tensorWidth * seg_w);
        int r_y = std::floor((bound.y * trans[1] + trans[3]) / _tensorHeight * seg_h);
        int r_w = std::ceil(((bound.x + bound.width) * trans[0] + trans[2]) / _tensorWidth * seg_w) - r_x;
        int r_h = std::ceil(((bound.y + bound.height) * trans[1] + trans[3]) / _tensorHeight * seg_h) - r_y;
        r_w = std::max(r_w, 1);
        r_h = std::max(r_h, 1);

        if (r_x + r_w > seg_w) // crop
        {
            seg_w - r_x > 0 ? r_w = seg_w - r_x : r_x -= 1;
        }
        if (r_y + r_h > seg_h)
        {
            seg_h - r_y > 0 ? r_h = seg_h - r_y : r_y -= 1;
        }

        std::vector<cv::Range> roi_ranges = {cv::Range(0, 1), cv::Range::all(), cv::Range(r_y, r_h + r_y), cv::Range(r_x, r_w + r_x)};
        cv::Mat temp_mask = mask_data(roi_ranges).clone();

        cv::Mat protos = temp_mask.reshape(0, {seg_ch, r_w * r_h});
        cv::Mat matmul_res = (mask_info * protos).t();

        cv::Mat masks_feature = matmul_res.reshape(1, {r_h, r_w});
        cv::Mat dest;

        cv::exp(-masks_feature, dest); // sigmoid
        dest = 1.0 / (1.0 + dest);

        int left = std::floor((_tensorWidth / seg_w * r_x - trans[2]) / trans[0]);
        int top = std::floor((_tensorHeight / seg_h * r_y - trans[3]) / trans[1]);
        int width = std::ceil(_tensorWidth / seg_w * r_w / trans[0]);
        int height = std::ceil(_tensorHeight / seg_h * r_h / trans[1]);

        cv::Mat mask;
        cv::resize(dest, mask, cv::Size(width, height));
        mask_out = mask(bound - cv::Point(left, top));
        if (mask_out.empty())
        {
            std::cout << "Mask_out empty" << std::endl;
            mask_out = cv::Mat();
        }
        else
        {
            mask_out = mask(bound - cv::Point(left, top)) > mask_thresh;
        }
    }

    void YoloProcessor::draw_result(cv::Mat img, std::vector<Obj> &result, std::vector<cv::Scalar> color)
    {
        cv::Mat mask = img.clone();
        for (int i = 0; i < result.size(); i++)
        {
            int left, top;
            left = result[i].bound.x;
            top = result[i].bound.y;
            int color_num = i;
            // cv::rectangle(img, result[i].bound, color[result[i].id], 8);
            if (result[i].mask.rows && result[i].mask.cols > 0)
            {
                mask(result[i].bound).setTo(color[result[i].id], result[i].mask);
            }
            std::string label = cv::format("%s:%.2f", class_names[result[i].id].c_str(), result[i].accu);
            // cv::putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 2, color[result[i].id], 4);
        }
        cv::addWeighted(img, 0.6, mask, 0.4, 0, img);
        cv::resize(img, img, cv::Size(640, 640));
        cv::imshow("img", img);
        cv::waitKey(1);
    }
}