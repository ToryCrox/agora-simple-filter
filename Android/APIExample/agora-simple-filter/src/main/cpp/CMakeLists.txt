# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
# 设置 cmkae 最低版本
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

# 设置 C++ 标准版本为 C++17
set(CMAKE_CXX_STANDARD 17)

# 设置引入的 cmake 文件路径为当前目录
SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# 定义当前模块的名称
SET(PROJECT_NAME "agora-simple-filter")
PROJECT(${PROJECT_NAME})

#link agora so
set(agora-lib-so ${PROJECT_SOURCE_DIR}/../agoraLibs/${CMAKE_ANDROID_ARCH_ABI}/libagora-rtc-sdk.so)
link_libraries(${agora-lib-so})

#link opencv so
set(opencv-lib-so ${PROJECT_SOURCE_DIR}/../jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libopencv_java4.so)
link_libraries(${opencv-lib-so})

#opencv inc
set(opencv-inc ${PROJECT_SOURCE_DIR}/../cpp/plugin_source_code)
include_directories(${opencv-inc})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")


# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

#add_library( # Sets the name of the library.
#        agora-simple-filter
#
#             # Sets the library as a shared library.
#             SHARED
#        plugin_source_code/ExtensionProvider.cpp
#        plugin_source_code/ExtensionVideoFilter.cpp
#        plugin_source_code/ExtensionAudioFilter.cpp
#        plugin_source_code/EGLCore.cpp
#        plugin_source_code/JniHelper.cpp
#        plugin_source_code/VideoProcessor.cpp
#        plugin_source_code/AudioProcessor.cpp
#        plugin_source_code/external_thread_pool.cpp
#        # Provides a relative path to your source file(s).
#        agora-simple-filter.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.




SET(CURRENT_OS "android")

# Config build output path
# --------
SET(OUTPUT_INSTALL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../output")
SET(OUTPUT_RESOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../output/resources")
SET(CMAKE_INCLUDE_OUTPUT_DIRECTORY "${OUTPUT_INSTALL_PATH}/include")
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_INSTALL_PATH}/library/${CURRENT_OS}")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_INSTALL_PATH}/library/${CURRENT_OS}")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_INSTALL_PATH}/library/${CURRENT_OS}")
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/filter
        ${CMAKE_CURRENT_SOURCE_DIR}/source
        ${CMAKE_CURRENT_SOURCE_DIR}/target
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/face_detect
        ${CMAKE_CURRENT_SOURCE_DIR}/jni
        ${CMAKE_CURRENT_SOURCE_DIR}/target/objc
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glad/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libyuv/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vnn/include
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin_source_code
)

# Add common source file
FILE(GLOB SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/filter/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/target/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/face_detect/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libyuv/source/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/plugin_source_code/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/jni/*"
)

# Add export header file
FILE(GLOB EXPORT_HEADER
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/filter/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/target/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/face_detect/*.h"
)

FILE(GLOB RESOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/*"
)
if(${CURRENT_OS} STREQUAL "ios" OR ${CURRENT_OS} STREQUAL "android")
    list(APPEND RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vnn/models/vnn_face278_data/face_mobile[1.0.0].vnnmodel")
else()
    list(APPEND RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vnn/models/vnn_face278_data/face_pc[1.0.0].vnnmodel")
endif()

LINK_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/third_party/vnn/libs/${CURRENT_OS}/${ANDROID_ABI})


# Config project
# ----------
# build shared or static lib
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${RESOURCE_FILES} agora-simple-filter.cpp)


# 设置要构建的目标库的名称和类型
add_library(vnn_kit SHARED IMPORTED)
# 设置目标库的实际路径
set_target_properties(vnn_kit PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vnn/libs/${CURRENT_OS}/${ANDROID_ABI}/libvnn_kit.so)

add_library(vnn_core SHARED IMPORTED)
# 设置目标库的实际路径
set_target_properties(vnn_core PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vnn/libs/${CURRENT_OS}/${ANDROID_ABI}/libvnn_core.so)

add_library(vnn_face SHARED IMPORTED)
# 设置目标库的实际路径
set_target_properties(vnn_face PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vnn/libs/${CURRENT_OS}/${ANDROID_ABI}/libvnn_face.so)


TARGET_LINK_LIBRARIES(
        ${PROJECT_NAME}
        log
        android
        GLESv3
        EGL
        jnigraphics
        vnn_core
        vnn_kit
        vnn_face)


find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/AgoraRtcKit)

target_link_libraries( # Specifies the target library.
        ${PROJECT_NAME}
        ${agora-lib-so}
        ${agora-ffmpeg-so}
        ${agora-soundtouch-so}
        ${agora-fdkaac-so}

        ${opencv-lib-so}
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}
        #                       ${GLESv2}
        ${GLESv3}
        EGL
        android)
