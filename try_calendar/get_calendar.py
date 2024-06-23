from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os

import os
import google.auth.transport.requests
import google_auth_oauthlib.flow
import googleapiclient.discovery

# 如果修改了这些范围，请删除之前的 token.json 文件
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

def authenticate_google():
    """Shows basic usage of the Google Calendar API.
    Prints the start and name of the next 10 events on the user's calendar.
    """
    creds = None
    # token.json 文件用于存储用户访问令牌，记录用户的会话
    if os.path.exists('token.json'):
        creds = google.oauth2.credentials.Credentials.from_authorized_user_file('token.json', SCOPES)
    # 如果没有（有效的）凭据，提示用户登录
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # 保存凭据以供将来使用
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_user_schedule(service, calendar_id):
    events_result = service.events().list(calendarId=calendar_id, timeMin='2024-05-05T00:00:00Z',
                                          timeMax='2024-05-07T23:59:59Z', singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])
    return events

# 认证并构建服务
creds = authenticate_google()
service = googleapiclient.discovery.build('calendar', 'v3', credentials=creds)

# 获取用户日程安排
userA_schedule = get_user_schedule(service, '')
userB_schedule = get_user_schedule(service, '')

