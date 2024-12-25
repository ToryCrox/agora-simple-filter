//
// Created by DYF on 2020/7/13.
//

#include "JniHelper.h"

#include "VideoProcessor.h"

#include <chrono>


#include "../logutils.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include "error_code.h"

#define CHECK_BEF_AI_RET_SUCCESS(ret, ...) \
if(ret != 0){\
    PRINTF_ERROR(__VA_ARGS__);\
}

namespace agora {
    namespace extension {
        using namespace rapidjson;
        bool WatermarkProcessor::initOpenGL() {
            sourceRawDataInput = gpupixel::SourceRawDataInput::create();
            sourceRawDataInput->RegLandmarkCallback([=](std::vector<float> landmarks) {
                PRINTF_INFO("RegLandmarkCallback  %zu", landmarks.size());
                ensureFaceReshapeFilter();
                ensureLipstickFilter();
                faceReshapeFilter->setProperty("face_landmark", landmarks);
                lipstickFilter->setProperty("face_landmark", landmarks);
            });
            return true;
        }

        bool WatermarkProcessor::releaseOpenGL() {
            return true;
        }

        bool WatermarkProcessor::makeCurrent() {
            return false;
        }

        bool WatermarkProcessor::detachCurrent() {
            return false;
        }

        int WatermarkProcessor::processFrame(agora::rtc::VideoFrameData &capturedFrame) {
            if (!targetRawDataOutput) {
                targetRawDataOutput = gpupixel::TargetRawDataOutput::create();
                sourceRawDataInput->addTarget(targetRawDataOutput);
            }
            int width = capturedFrame.width;
            int height = capturedFrame.height;
            int ySize = width * height;
            int uvSize = (width / 2) * (height / 2);

            uint8_t* yPlane = capturedFrame.pixels.data;
            uint8_t* uPlane = capturedFrame.pixels.data + ySize;
            uint8_t* vPlane = capturedFrame.pixels.data + ySize + uvSize;

            targetRawDataOutput->setOutputYuvFrameBuffer(capturedFrame.pixels.data);
            sourceRawDataInput->uploadBytes(width, height, yPlane, ySize, uPlane, ySize,
                                                   vPlane, ySize, capturedFrame.timestamp_ms);
            //readYUVData(capturedFrame.pixels.data, width, height);
            return 0;
        }

        void WatermarkProcessor::readYUVData(uint8_t* yuvData, int width, int height) {
            // 获取已有的 Framebuffer 对象
            auto framebuffer = sourceRawDataInput->getFramebuffer();
            if (!framebuffer) {
                PRINTF_ERROR("Framebuffer is not initialized!");
                return;
            }

            GLuint textureId = framebuffer->getTexture();
            if (textureId == 0) {
                PRINTF_ERROR("Texture ID is invalid!");
                return;
            }

            // 绑定 Framebuffer 和纹理
            //glBindFramebuffer(GL_FRAMEBUFFER, framebuffer->getFramebuffer());
            glBindTexture(GL_TEXTURE_2D, textureId);

            // 读取 RGB 数据到 CPU 内存
            std::vector<uint8_t> rgbData(width * height * 3);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);  // 确保像素存储对齐方式为1字节对齐
            glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgbData.data());

            // 将 RGB 数据转换为 YUV 数据
            cv::Mat rgbImage(height, width, CV_8UC3, rgbData.data());
            cv::Mat yuvImage;
            cv::cvtColor(rgbImage, yuvImage, cv::COLOR_RGB2YUV_I420);

            // 复制 YUV 数据到目标缓冲区
            std::memcpy(yuvData, yuvImage.data, yuvImage.total() * yuvImage.elemSize());

            // 解绑 Framebuffer 和纹理
            glBindTexture(GL_TEXTURE_2D, 0);
            //glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        void WatermarkProcessor::addWatermark(const agora::rtc::VideoFrameData &capturedFrame) {
            cv::Mat image(capturedFrame.height, capturedFrame.width, CV_8U,
                           (void*)capturedFrame.pixels.data);
            double fontSize = image.cols / 800;
            if (fontSize == 0.0) {
                fontSize = 2.0;
            }
            cv::Point point(image.rows/2, image.cols/2);
            cv::Scalar scalar(255, 0, 0);
            cv::Mat textImg = cv::Mat::zeros(image.rows, image.cols, image.type());
            cv::putText(textImg, wmStr_, point, cv::FONT_HERSHEY_DUPLEX, fontSize, scalar,3);
            cv::flip(textImg, textImg, 0);
            image = image + textImg;
        }

        void WatermarkProcessor::ensureBeautyFaceFilter() {
            if (!beautyFaceFilter) {
                beautyFaceFilter = gpupixel::BeautyFaceFilter::create();
                sourceRawDataInput->addTarget(beautyFaceFilter);
            }
        }

        void WatermarkProcessor::ensureFaceReshapeFilter() {
            if (!faceReshapeFilter) {
                faceReshapeFilter = gpupixel::FaceReshapeFilter::create();
                sourceRawDataInput->addTarget(faceReshapeFilter);
            }
        }

        void WatermarkProcessor::ensureLipstickFilter() {
            if (!lipstickFilter) {
                lipstickFilter = gpupixel::LipstickFilter::create();
                sourceRawDataInput->addTarget(lipstickFilter);
            }
        }

        int WatermarkProcessor::setProperty(std::string property, std::string parameter) {
            Document d;
            d.Parse(parameter.c_str());
            if (d.HasParseError()) {
                return -ERROR_INVALID_JSON;
            }
            if (property == "setResourceRoot") {
                if (!d.HasMember("rootPath")) {
                    return -ERROR_INVALID_JSON;
                }
                Value& rootPathValue = d["rootPath"];
                if (!rootPathValue.IsString()) {
                    return -ERROR_INVALID_JSON;
                }
                gpupixel::Util::setResourceRoot(rootPathValue.GetString());
                return 0;
            }

            Value& methodNameValue = d["methodName"];
            std::string filterName = methodNameValue.GetString();

            if (!d.HasMember("methodValue")) {
                return -ERROR_INVALID_JSON;
            }
            Value& methodValue = d["methodValue"];
            // 判断property是否以BeautyFaceFilter开头
            if (property == "BeautyFaceFilter") {
                ensureBeautyFaceFilter();
                if (!methodValue.IsFloat()) {
                    return -ERROR_INVALID_JSON;
                }
                // skin_smoothing,whiteness
                beautyFaceFilter->setProperty(filterName, methodValue.GetFloat());
            } else if (property == "FaceReshapeFilter") {
                ensureFaceReshapeFilter();
                // thin_face float
                // big_eye float
                // face_landmark float[]
                if (!methodValue.IsFloat()) {
                    return -ERROR_INVALID_JSON;
                }
                faceReshapeFilter->setProperty(filterName, methodValue.GetFloat());
            } else if (property == "LipstickFilter") {
                ensureLipstickFilter();
                // face_landmark float[]
                // blend_level float
                if (!methodValue.IsFloat()) {
                    return -ERROR_INVALID_JSON;
                }
                lipstickFilter->setProperty(filterName, methodValue.GetFloat());
            }
            return 0;
        }

        std::thread::id WatermarkProcessor::getThreadId() {
            std::thread::id id = std::this_thread::get_id();
            return id;
        }

        void WatermarkProcessor::dataCallback(const char* data){
            if (control_) {
                control_->postEvent("key", data);
            }
        }
    }
}
