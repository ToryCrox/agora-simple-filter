#!/usr/bin/env sh

# cache gradle to /tmp/.gradle
ls ~/.gradle || (mkdir -p /tmp/.gradle && ln -s /tmp/.gradle ~/.gradle && touch ~/.gradle/ln_$(date "+%y%m%d%H") && ls ~/.gradle)

## use open jdk 17
SYSTEM=$(uname -s)
if [ "$SYSTEM" = "Linux" ];then
if [ ! -d "/tmp/jdk-17.0.2" ];then
  curl -O https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
  tar zxf openjdk-17.0.2_linux-x64_bin.tar.gz
  mv jdk-17.0.2 /tmp/
fi
export JAVA_HOME=/tmp/jdk-17.0.2
export PATH=$JAVA_HOME/bin:$PATH
java --version
fi

## config appId
if [ ! -f "local.properties" ];then
  touch local.properties
  echo "AGORA_APP_ID=${APP_ID}" >> local.properties
fi

./gradlew clean || exit 1
./gradlew :app:assembleRelease || exit 1

SDK_VERSION=""
if [ "$1" = "false" ]; then
    sdk_version_file="./gradle.properties"
    if [[ -f "$sdk_version_file" ]]; then
    rtc_sdk_version=$(grep "rtc_sdk_version" "$sdk_version_file" | cut -d'=' -f2)
    if [[ -n "$rtc_sdk_version" ]]; then
        SDK_VERSION=$(echo "$rtc_sdk_version" | sed 's/^[ \t]*//;s/[ \t]*$//')
    else
        echo "rtc_sdk_version value not found"
    fi
else
    echo "file not found: $sdk_version_file"
fi
else
    SDK_VERSION=$(echo $sdk_url | cut -d "/" -f 5)
fi

if [ "$WORKSPACE" != "" ]; then
cp app/build/outputs/apk/release/*.apk $WORKSPACE/APIExample-Compose_${BUILD_NUMBER}_${SDK_VERSION}_$(date "+%Y%m%d%H%M%S").apk
fi