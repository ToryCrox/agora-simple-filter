﻿#pragma once
#include "AGVideoWnd.h"
#include <list>
#include <string>
#include "stdafx.h"

class CTransparentBgEventHandler : public IRtcEngineEventHandler
{
public:
    void SetMsgReceiver(HWND hWnd) { m_hMsgHanlder = hWnd; }
    virtual void onJoinChannelSuccess(const char* channel, uid_t uid, int elapsed) override;
    virtual void onUserJoined(uid_t uid, int elapsed) override;
    virtual void onUserOffline(uid_t uid, USER_OFFLINE_REASON_TYPE reason) override;
	virtual void onLeaveChannel(const RtcStats& stats) override;
	virtual void onLocalAudioStats(const LocalAudioStats& stats) override{
		if (m_hMsgHanlder&& report) {
			LocalAudioStats* s = new LocalAudioStats;
			*s = stats;
			::PostMessage(m_hMsgHanlder, WM_MSGID(EID_LOCAL_AUDIO_STATS), (WPARAM)s, 0);

		}
	}
	virtual void onRemoteAudioStats(const RemoteAudioStats& stats) {
		if (m_hMsgHanlder&& report) {
			RemoteAudioStats* s = new RemoteAudioStats;
			*s = stats;
			::PostMessage(m_hMsgHanlder, WM_MSGID(EID_REMOTE_AUDIO_STATS), (WPARAM)s, 0);

		}
	}
	virtual void onRemoteVideoStats(const RemoteVideoStats& stats) {
		if (m_hMsgHanlder&& report) {
			RemoteVideoStats* s = new RemoteVideoStats;
			*s = stats;
			::PostMessage(m_hMsgHanlder, WM_MSGID(EID_REMOTE_VIDEO_STATS), (WPARAM)s, 0);
		}
	}
    virtual void onLocalVideoStats(VIDEO_SOURCE_TYPE source, const LocalVideoStats& stats) override;
    virtual void onError(int err, const char* msg) {
        if (m_hMsgHanlder) {
            char* message = new char[1024]{0};
            memcpy(message, msg, strlen(msg));
            ::PostMessage(m_hMsgHanlder, WM_MSGID(EID_ERROR), (WPARAM)message, err);

        }
    }
private:
    HWND m_hMsgHanlder = nullptr;
	bool report = true;
};

class CTransparentBgDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CTransparentBgDlg)

public:
    CTransparentBgDlg(CWnd* pParent = nullptr);   // standard constructor
	virtual ~CTransparentBgDlg();
    bool InitAgora();
    void UnInitAgora();
	enum { IDD = IDD_DIALOG_TRANSPARENT_BG};
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	DECLARE_MESSAGE_MAP()
public:
    afx_msg LRESULT OnEIDJoinChannelSuccess(WPARAM wParam, LPARAM lParam);
    afx_msg LRESULT OnEIDError(WPARAM wParam, LPARAM lParam);
    afx_msg LRESULT OnEIDLeaveChannel(WPARAM wParam, LPARAM lParam);
    afx_msg LRESULT OnEIDUserJoined(WPARAM wParam, LPARAM lParam);
    afx_msg LRESULT OnEIDUserOffline(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT onEIDLocalAudioStats(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT onEIDRemoteAudioStats(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT onEIDLocalVideoStats(WPARAM wParam, LPARAM lParam);
	afx_msg LRESULT onEIDRemoteVideoStats(WPARAM wParam, LPARAM lParam);

private:
    void InitCtrlText();
	int JoinChannel(const char* channel);
	int LeaveChannel();
    IRtcEngine* m_rtcEngine = nullptr;
    CTransparentBgEventHandler m_eventHandler;
    bool m_joinChannel = false;
    bool m_initialize = false;
	int m_uid = 1001;
	int m_remoteId = 0;
public:
	virtual BOOL OnInitDialog();
	virtual BOOL PreTranslateMessage(MSG* pMsg);

	afx_msg void OnBnClickedButtonJoinchannel();
private:
	CStatic m_staticChannelName;
	CButton m_bnJoinChannel;
	CEdit m_editChannel;
	CStatic m_staticVideo;
	CStatic m_staticVideoLeft;
	CStatic m_staticVideoRight;
	CListBox m_listInfo;
public:
	afx_msg void OnShowWindow(BOOL bShow, UINT nStatus);
};
