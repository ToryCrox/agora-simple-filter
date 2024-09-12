﻿#include "stdafx.h"
#include "APIExample.h"
#include "TransparentBgDialog.h"

void CTransparentBgEventHandler::onJoinChannelSuccess(const char *channel, uid_t uid, int elapsed)
{
    if (m_hMsgHanlder)
    {
        ::PostMessage(m_hMsgHanlder, WM_MSGID(EID_JOINCHANNEL_SUCCESS), (WPARAM)uid, (LPARAM)elapsed);
    }
}

void CTransparentBgEventHandler::onUserJoined(uid_t uid, int elapsed)
{
    if (m_hMsgHanlder)
    {
        ::PostMessage(m_hMsgHanlder, WM_MSGID(EID_USER_JOINED), (WPARAM)uid, (LPARAM)elapsed);
    }
}

void CTransparentBgEventHandler::onUserOffline(uid_t uid, USER_OFFLINE_REASON_TYPE reason)
{
    if (m_hMsgHanlder)
    {
        ::PostMessage(m_hMsgHanlder, WM_MSGID(EID_USER_OFFLINE), (WPARAM)uid, (LPARAM)reason);
    }
}

void CTransparentBgEventHandler::onLeaveChannel(const RtcStats &stats)
{
    if (m_hMsgHanlder)
    {
        ::PostMessage(m_hMsgHanlder, WM_MSGID(EID_LEAVE_CHANNEL), 0, 0);
    }
}

void CTransparentBgEventHandler::onLocalVideoStats(VIDEO_SOURCE_TYPE source, const LocalVideoStats &stats)
{
    if (m_hMsgHanlder && report)
    {
        LocalVideoStats *s = new LocalVideoStats;
        *s = stats;
        ::PostMessage(m_hMsgHanlder, WM_MSGID(EID_LOCAL_VIDEO_STATS), (WPARAM)s, 0);
    }
}

IMPLEMENT_DYNAMIC(CTransparentBgDlg, CDialogEx)

CTransparentBgDlg::CTransparentBgDlg(CWnd *pParent /*=nullptr*/)
    : CDialogEx(IDD_DIALOG_TRANSPARENT_BG, pParent)
{
}

CTransparentBgDlg::~CTransparentBgDlg()
{
}

void CTransparentBgDlg::DoDataExchange(CDataExchange *pDX)
{
    CDialogEx::DoDataExchange(pDX);
    DDX_Control(pDX, IDC_STATIC_CHANNELNAME, m_staticChannelName);
    DDX_Control(pDX, IDC_BUTTON_JOINCHANNEL, m_bnJoinChannel);
    DDX_Control(pDX, IDC_EDIT_CHANNELNAME, m_editChannel);
    DDX_Control(pDX, IDC_STATIC_VIDEO, m_staticVideo);
    DDX_Control(pDX, IDC_STATIC_VIDEO_LEFT, m_staticVideoLeft);
    DDX_Control(pDX, IDC_STATIC_VIDEO_RIGHT, m_staticVideoRight);
    DDX_Control(pDX, IDC_LIST_INFO_BROADCASTING, m_listInfo);
}

BEGIN_MESSAGE_MAP(CTransparentBgDlg, CDialogEx)

ON_MESSAGE(WM_MSGID(EID_JOINCHANNEL_SUCCESS), &CTransparentBgDlg::OnEIDJoinChannelSuccess)
ON_MESSAGE(WM_MSGID(EID_ERROR), &CTransparentBgDlg::OnEIDError)
ON_MESSAGE(WM_MSGID(EID_LEAVE_CHANNEL), &CTransparentBgDlg::OnEIDLeaveChannel)
ON_MESSAGE(WM_MSGID(EID_USER_JOINED), &CTransparentBgDlg::OnEIDUserJoined)
ON_MESSAGE(WM_MSGID(EID_USER_OFFLINE), &CTransparentBgDlg::OnEIDUserOffline)

ON_MESSAGE(WM_MSGID(EID_LOCAL_AUDIO_STATS), &CTransparentBgDlg::onEIDLocalAudioStats)
ON_MESSAGE(WM_MSGID(EID_REMOTE_AUDIO_STATS), &CTransparentBgDlg::onEIDRemoteAudioStats)
ON_MESSAGE(WM_MSGID(EID_LOCAL_VIDEO_STATS), &CTransparentBgDlg::onEIDLocalVideoStats)
ON_MESSAGE(WM_MSGID(EID_REMOTE_VIDEO_STATS), &CTransparentBgDlg::onEIDRemoteVideoStats)

ON_WM_SHOWWINDOW()

ON_BN_CLICKED(IDC_BUTTON_JOINCHANNEL, &CTransparentBgDlg::OnBnClickedButtonJoinchannel)
ON_WM_SHOWWINDOW()
END_MESSAGE_MAP()

// CTransparentBgDlg message handlers
BOOL CTransparentBgDlg::OnInitDialog()
{
    CDialogEx::OnInitDialog();
    return TRUE;
}

// set control text from config.
void CTransparentBgDlg::InitCtrlText()
{
    m_staticChannelName.SetWindowText(commonCtrlChannel);
    m_bnJoinChannel.SetWindowText(commonCtrlJoinChannel);
    m_editChannel.SetWindowText(_T(""));
}

void CTransparentBgDlg::InvalidateVideo()
{
    m_staticVideo.Invalidate();
    m_staticVideoLeft.Invalidate();
    m_staticVideoRight.Invalidate();
}

int CTransparentBgDlg::JoinChannel(const char *channel)
{
    VideoEncoderConfiguration config;
    config.advanceOptions.encodeAlpha = true;
    m_rtcEngine->setVideoEncoderConfiguration(config);

    ChannelMediaOptions options;
    options.channelProfile = CHANNEL_PROFILE_LIVE_BROADCASTING;
    options.clientRoleType = CLIENT_ROLE_BROADCASTER;
    options.autoSubscribeAudio = false;
    options.autoSubscribeVideo = true;
    options.publishMicrophoneTrack = false;
    options.publishCameraTrack = false;
    options.publishCustomVideoTrack = true;
    // join channel in the engine.
    int ret = m_rtcEngine->joinChannel(APP_TOKEN, channel, m_uid, options);

    return ret;
}

int CTransparentBgDlg::LeaveChannel()
{
    if (!m_joinChannel)
    {
        return 0;
    };
    m_rtcEngine->stopPreview();
    int ret = m_rtcEngine->leaveChannel();
    m_joinChannel = false;
    m_remoteId = 0;
    return ret;
}

void CTransparentBgDlg::StartPlay()
{
    m_mediaPlayer = m_rtcEngine->createMediaPlayer();
    if (m_mediaPlayer)
    {
        m_mediaPlayer->setView((agora::media::base::view_t)m_staticVideo.GetSafeHwnd());
        m_mediaPlayer->setRenderMode(RENDER_MODE_FIT);
        m_mediaPlayer->registerVideoFrameObserver(this);
        m_mediaPlayer->registerPlayerSourceObserver(this);
        MediaSource mediaSource;
        CString videoUrl = GetExePath() + _T("\\yuvj_full_range_alpha_1280_540_left.mp4");
        std::string tmp = cs2utf8(videoUrl);
        mediaSource.url = tmp.c_str();
        CString pathInfo;
        int ret = m_mediaPlayer->openWithMediaSource(mediaSource);
        CString strInfo;
        strInfo.Format(_T("openWithMediaSource ret: %d"), ret);
        m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
        m_mediaPlayer->setLoopCount(-1);
    }
}

void CTransparentBgDlg::StopPlay()
{
    if (m_mediaPlayer)
    {
        m_mediaPlayer->stop();
    }
}
void CTransparentBgDlg::onFrame(const VideoFrame *frame)
{
    if (m_rtcEngine)
    {

        // 创建ExternalVideoFrame对象
        agora::media::base::ExternalVideoFrame externalFrame;
        externalFrame.alphaStitchMode = ALPHA_STITCH_LEFT;
        externalFrame.type = agora::media::base::ExternalVideoFrame::VIDEO_BUFFER_TYPE::VIDEO_BUFFER_RAW_DATA;
        externalFrame.format = agora::media::base::VIDEO_PIXEL_I420;
        externalFrame.stride = frame->width;
        externalFrame.height = frame->height;
        externalFrame.timestamp = m_rtcEngine->getCurrentMonotonicTimeInMs();

        int bufferSize = frame->yStride * frame->height + frame->uStride * (frame->height >> 1) + frame->vStride * (frame->height >> 1);
        externalFrame.buffer = new uint8_t[bufferSize];
        memcpy(externalFrame.buffer, frame->yBuffer, frame->yStride * frame->height);
        memcpy((uint8_t *)externalFrame.buffer + frame->yStride * frame->height, frame->uBuffer, frame->uStride * (frame->height >> 1));
        memcpy((uint8_t *)externalFrame.buffer + frame->yStride * frame->height + frame->uStride * (frame->height >> 1), frame->vBuffer, frame->vStride * (frame->height / 2));
        mediaEngine->pushVideoFrame(&externalFrame);
        // 释放缓冲区
        delete[] externalFrame.buffer;

        // CString strInfo;
        // strInfo.Format(_T("height ret: %d"), frame->height);
        // m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    }
}

void CTransparentBgDlg::onPlayerSourceStateChanged(media::base::MEDIA_PLAYER_STATE state, media::base::MEDIA_PLAYER_REASON reason)
{
    CString strState;
    CString strError;
    switch (state)
    {
    case agora::media::base::PLAYER_STATE_OPEN_COMPLETED:
        strState = _T("PLAYER_STATE_OPEN_COMPLETED");
        m_mediaPlayer->play();
        break;
    case agora::media::base::PLAYER_STATE_OPENING:
        strState = _T("PLAYER_STATE_OPENING");
        break;
    case agora::media::base::PLAYER_STATE_IDLE:
        strState = _T("PLAYER_STATE_IDLE");
        break;
    case agora::media::base::PLAYER_STATE_PLAYING:
        strState = _T("PLAYER_STATE_PLAYING");
        break;
    case agora::media::base::PLAYER_STATE_PLAYBACK_COMPLETED:
        strState = _T("PLAYER_STATE_PLAYBACK_COMPLETED");
        break;
    case agora::media::base::PLAYER_STATE_PLAYBACK_ALL_LOOPS_COMPLETED:
        strState = _T("PLAYER_STATE_PLAYBACK_ALL_LOOPS_COMPLETED");
        break;
    case agora::media::base::PLAYER_STATE_PAUSED:
        strState = _T("PLAYER_STATE_PAUSED");
        break;
    case agora::media::base::PLAYER_STATE_STOPPED:
        strState = _T("PLAYER_STATE_STOPPED");
        break;
    case agora::media::base::PLAYER_STATE_FAILED:
        strState = _T("PLAYER_STATE_FAILED");
        m_mediaPlayer->stop();
        break;
    default:
        strState = _T("PLAYER_STATE_UNKNOWN");
        break;
    }
    switch (reason)
    {
    case agora::media::base::PLAYER_REASON_URL_NOT_FOUND:
        strError = _T("PLAYER_ERROR_URL_NOT_FOUND");
        break;
    case agora::media::base::PLAYER_REASON_NONE:
        strError = _T("PLAYER_ERROR_NONE");
        break;
    case agora::media::base::PLAYER_REASON_CODEC_NOT_SUPPORTED:
        strError = _T("PLAYER_ERROR_NONE");
        break;
    case agora::media::base::PLAYER_REASON_INVALID_ARGUMENTS:
        strError = _T("PLAYER_ERROR_INVALID_ARGUMENTS");
        break;
    case agora::media::base::PLAYER_REASON_SRC_BUFFER_UNDERFLOW:
        strError = _T("PLAY_ERROR_SRC_BUFFER_UNDERFLOW");
        break;
    case agora::media::base::PLAYER_REASON_INTERNAL:
        strError = _T("PLAYER_ERROR_INTERNAL");
        break;
    case agora::media::base::PLAYER_REASON_INVALID_STATE:
        strError = _T("PLAYER_ERROR_INVALID_STATE");
        break;
    case agora::media::base::PLAYER_REASON_NO_RESOURCE:
        strError = _T("PLAYER_ERROR_NO_RESOURCE");
        break;
    case agora::media::base::PLAYER_REASON_OBJ_NOT_INITIALIZED:
        strError = _T("PLAYER_ERROR_OBJ_NOT_INITIALIZED");
        break;
    case agora::media::base::PLAYER_REASON_INVALID_CONNECTION_STATE:
        strError = _T("PLAYER_ERROR_INVALID_CONNECTION_STATE");
        break;
    case agora::media::base::PLAYER_REASON_UNKNOWN_STREAM_TYPE:
        strError = _T("PLAYER_ERROR_UNKNOWN_STREAM_TYPE");
        break;
    case agora::media::base::PLAYER_REASON_VIDEO_RENDER_FAILED:
        strError = _T("PLAYER_ERROR_VIDEO_RENDER_FAILED");
        break;
    }
    CString strInfo;
    strInfo.Format(_T("sta:%s,\nerr:%s"), strState, strError);
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
}

// Initialize the Agora SDK
bool CTransparentBgDlg::InitAgora()
{
    // create Agora RTC engine
    m_rtcEngine = createAgoraRtcEngine();
    if (!m_rtcEngine)
    {
        m_listInfo.InsertString(m_listInfo.GetCount(), _T("createAgoraRtcEngine failed"));
        return false;
    }
    // set message notify receiver window
    m_eventHandler.SetMsgReceiver(m_hWnd);

    RtcEngineContext context;
    std::string strAppID = GET_APP_ID;
    context.appId = strAppID.c_str();
    context.eventHandler = &m_eventHandler;
    // set channel profile in the engine to the CHANNEL_PROFILE_LIVE_BROADCASTING.
    context.channelProfile = CHANNEL_PROFILE_LIVE_BROADCASTING;
    // initialize the Agora RTC engine context.
    int ret = m_rtcEngine->initialize(context);
    if (ret != 0)
    {
        m_initialize = false;
        CString strInfo;
        strInfo.Format(_T("initialize failed: %d"), ret);
        m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
        return false;
    }
    else
    {
        m_initialize = true;
        m_listInfo.InsertString(m_listInfo.GetCount(), _T("createAgoraRtcEngine success"));

        mediaEngine.queryInterface(m_rtcEngine, agora::rtc::AGORA_IID_MEDIA_ENGINE);
        agora::base::AParameter apm(m_rtcEngine);
        mediaEngine->setExternalVideoSource(true, true);
    }
    m_rtcEngine->enableVideo();
    m_rtcEngine->enableAudio();
    return true;
}

// UnInitialize the Agora SDK
void CTransparentBgDlg::UnInitAgora()
{
    if (m_rtcEngine)
    {
        if (m_joinChannel)
            LeaveChannel();
        // m_rtcEngine->stopPreview();
        m_rtcEngine->disableVideo();
        if (m_mediaPlayer)
        {
            //  m_mediaPlayer->registerPlayerSourceObserver(nullptr);
            m_mediaPlayer->Release();
            m_mediaPlayer = nullptr;
        }

        if (m_initialize)
        {
            m_rtcEngine->release(true);
        }
        m_rtcEngine = NULL;
    }
}

LRESULT CTransparentBgDlg::OnEIDJoinChannelSuccess(WPARAM wParam, LPARAM lParam)
{
    CString strInfo;
    strInfo.Format(TEXT("self join success, wParam=%u"), wParam);
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    // notify parent window
    ::PostMessage(GetParent()->GetSafeHwnd(), WM_MSGID(EID_JOINCHANNEL_SUCCESS), TRUE, 0);
    return 0;
}

LRESULT CTransparentBgDlg::OnEIDError(WPARAM wParam, LPARAM lParam)
{
    CString strInfo;
    char *message = (char *)wParam;
    int code = lParam;
    strInfo.Format(TEXT("Error >> code=%d, message=%s"), code, message);
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    delete message;
    return 0;
}

LRESULT CTransparentBgDlg::OnEIDLeaveChannel(WPARAM wParam, LPARAM lParam)
{
    CString strInfo;
    strInfo.Format(TEXT("leave channel success"));
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    // notify parent window
    ::PostMessage(GetParent()->GetSafeHwnd(), WM_MSGID(EID_JOINCHANNEL_SUCCESS), FALSE, 0);
    InvalidateVideo();
    return 0;
}

LRESULT CTransparentBgDlg::OnEIDUserJoined(WPARAM wParam, LPARAM lParam)
{
    if (m_remoteId != 0)
    {
        m_listInfo.InsertString(m_listInfo.GetCount(), _T("user joined already"));
        return 0;
    }
    CString strInfo;
    strInfo.Format(TEXT("user %u joined"), wParam);
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    uid_t remoteUid = (uid_t)wParam;
    m_remoteId = remoteUid;
    VideoCanvas canvas;
    canvas.enableAlphaMask = true;
    canvas.uid = m_remoteId;
    canvas.view = m_staticVideoRight.GetSafeHwnd();
    canvas.renderMode = media::base::RENDER_MODE_FIT;
    m_rtcEngine->setupRemoteVideo(canvas);
    return 0;
}

LRESULT CTransparentBgDlg::OnEIDUserOffline(WPARAM wParam, LPARAM lParam)
{
    uid_t remoteUid = (uid_t)wParam;
    if (m_remoteId == remoteUid)
    {
        m_remoteId = 0;
        VideoCanvas canvas;
        canvas.uid = remoteUid;
        canvas.view = NULL;
        m_rtcEngine->setupRemoteVideo(canvas);
        CString strInfo;
        strInfo.Format(TEXT("%u offline, reason:%d"), remoteUid, lParam);
        m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    }
    InvalidateVideo();
    return 0;
}

BOOL CTransparentBgDlg::PreTranslateMessage(MSG *pMsg)
{
    if (pMsg->message == WM_KEYDOWN && pMsg->wParam == VK_RETURN)
    {
        return TRUE;
    }
    return CDialogEx::PreTranslateMessage(pMsg);
}

LRESULT CTransparentBgDlg::onEIDLocalAudioStats(WPARAM wParam, LPARAM lParam)
{
    LocalAudioStats *stats = (LocalAudioStats *)wParam;

   CString strInfo = _T("===onLocalAudioStats===");
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);

	strInfo.Format(_T("numChannels:%u"), stats->numChannels);
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
	strInfo.Format(_T("sentSampleRate:%u"), stats->sentSampleRate);
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    return 0;
}

LRESULT CTransparentBgDlg::onEIDRemoteAudioStats(WPARAM wParam, LPARAM lParam)
{
    RemoteAudioStats *stats = (RemoteAudioStats *)wParam;

    CString strInfo = _T("===onRemoteAudioStats===");
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);

	strInfo.Format(_T("uid:%u"), stats->uid);
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
	strInfo.Format(_T("quality:%d"), stats->quality);
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    return 0;
}

LRESULT CTransparentBgDlg::onEIDLocalVideoStats(WPARAM wParam, LPARAM lParam)
{
    LocalVideoStats *stats = (LocalVideoStats *)wParam;

    CString strInfo = _T("===onLocalVideoStats===");
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);

	strInfo.Format(_T("uid:%u"), stats->uid);
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
	strInfo.Format(_T("sentBitrate:%d"), stats->sentBitrate);
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
	
	strInfo.Format(_T("sentFrameRate:%d"), stats->sentFrameRate);
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
	strInfo.Format(_T("encoderOutputFrameRate:%d"), stats->encoderOutputFrameRate);
	m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);

    return 0;
}

LRESULT CTransparentBgDlg::onEIDRemoteVideoStats(WPARAM wParam, LPARAM lParam)
{
    RemoteVideoStats *stats = (RemoteVideoStats *)wParam;
    CString strInfo = _T("===onRemoteVideoStats===");
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    strInfo.Format(_T("uid:%u"), stats->uid);
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    strInfo.Format(_T("delay:%d"), stats->delay);
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);

    strInfo.Format(_T("width:%d"), stats->width);
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    strInfo.Format(_T("height:%d"), stats->height);
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);

    strInfo.Format(_T("receivedBitrate:%d"), stats->receivedBitrate);
    m_listInfo.InsertString(m_listInfo.GetCount(), strInfo);
    return 0;
}

void CTransparentBgDlg::OnBnClickedButtonJoinchannel()
{
    CString info;
    if (!m_joinChannel)
    {
        CString strChannelName;
        m_editChannel.GetWindowText(strChannelName);
        if (strChannelName.IsEmpty())
        {
            MessageBox(_T("频道号不能为空"));
            return;
        }

        VideoCanvas canvas;
        canvas.mirrorMode = VIDEO_MIRROR_MODE_DISABLED;
        canvas.sourceType = VIDEO_SOURCE_CUSTOM;
        canvas.enableAlphaMask = true;
        canvas.renderMode = media::base::RENDER_MODE_FIT;
        canvas.uid = m_uid;
        canvas.view = m_staticVideoLeft.GetSafeHwnd();
        m_rtcEngine->setupLocalVideo(canvas);
        m_rtcEngine->startPreview(VIDEO_SOURCE_CUSTOM);

        std::string szChannelName = cs2utf8(strChannelName);
        int ret = JoinChannel(szChannelName.c_str());
        if (ret == 0)
        {
            m_joinChannel = true;
            m_bnJoinChannel.SetWindowText(commonCtrlLeaveChannel);

            StartPlay();
        }
        else
        {
            const char *des = getAgoraSdkErrorDescription(ret);
            info.Format(TEXT("join channel failed >> code=%d, des=%s"), ret, utf82cs(std::string(des)));
            m_listInfo.InsertString(m_listInfo.GetCount(), info);
        }
    }
    else
    {
        int ret = LeaveChannel();
        if (0 == ret)
        {
            InitCtrlText();
            InvalidateVideo();
            StopPlay();
        }
        else
        {
            const char *des = getAgoraSdkErrorDescription(ret);
            info.Format(TEXT("leave channel failed >> code=%d, des=%s"), ret, utf82cs(std::string(des)));
            m_listInfo.InsertString(m_listInfo.GetCount(), info);
        }
    }
}

void CTransparentBgDlg::OnShowWindow(BOOL bShow, UINT nStatus)
{
    CDialogEx::OnShowWindow(bShow, nStatus);

    if (bShow)
    {
        m_listInfo.ResetContent();
    }
    else
    {
        InitCtrlText();
        InvalidateVideo();
    }
}
