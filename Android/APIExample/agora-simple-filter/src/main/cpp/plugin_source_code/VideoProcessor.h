//
// Created by DYF on 2020/7/13.
//

#ifndef AGORAWITHBYTEDANCE_VIDEOPROCESSOR_H
#define AGORAWITHBYTEDANCE_VIDEOPROCESSOR_H

#include <thread>
#include <string>
#include <mutex>
#include <vector>
#include <AgoraRtcKit/AgoraRefPtr.h>
#include "AgoraRtcKit/NGIAgoraMediaNode.h"

#include "AgoraRtcKit/AgoraMediaBase.h"
#include "opencv2/core/utility.hpp"

#include "EGLCore.h"
#include "rapidjson/rapidjson.h"
#include "gpupixel.h"

namespace agora {
    namespace extension {
        class WatermarkProcessor  : public RefCountInterface {
        public:
            bool initOpenGL();

            bool releaseOpenGL();

            bool makeCurrent();

            bool detachCurrent();

            int processFrame(agora::rtc::VideoFrameDataV2 &capturedFrame);

            int setProperty(std::string property, std::string parameter);

            std::thread::id getThreadId();

            int setExtensionControl(agora::agora_refptr<rtc::IExtensionVideoFilter::Control> control){
                control_ = control;
                return 0;
            };

        protected:
            ~WatermarkProcessor() {}
        private:
            void addWatermark(const agora::rtc::VideoFrameData &capturedFrame);
            void dataCallback(const char* data);

#if defined(__ANDROID__) || defined(TARGET_OS_ANDROID)
            EglCore *eglCore_ = nullptr;
            EGLSurface offscreenSurface_ = nullptr;
#endif
            std::mutex mutex_;
            agora::agora_refptr<rtc::IExtensionVideoFilter::Control> control_;
            bool wmEffectEnabled_ = false;
            std::string wmStr_= "agora";

            std::shared_ptr<gpupixel::BeautyFaceFilter> beautyFaceFilter;
            std::shared_ptr<gpupixel::FaceReshapeFilter> faceReshapeFilter;
            std::shared_ptr<gpupixel::LipstickFilter> lipstickFilter;
            std::shared_ptr<gpupixel::SourceRawDataInput> sourceRawDataInput;

            std::shared_ptr<gpupixel::TargetRawDataOutput> targetRawDataOutput;

            void ensureBeautyFaceFilter();
            void ensureFaceReshapeFilter();
            void ensureLipstickFilter();
            bool isLandmarkCallbackInit = false;
            void initLandmarkCallback();
            void readYUVData(uint8_t* yuvData, int width, int height);
        };
    }
}


#endif //AGORAWITHBYTEDANCE_VIDEOPROCESSOR_H
