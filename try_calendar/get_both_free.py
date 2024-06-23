# GOOOOOOAAAAAATTTTTT

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import google.auth.transport.requests
import google_auth_oauthlib.flow
import googleapiclient.discovery
import datetime as dt
from datetime import datetime,timedelta
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser

# 如果修改了这些范围，请删除之前的 token.json 文件
SCOPES = ['https://www.googleapis.com/auth/calendar']

def authenticate_google():
    creds = None
    if os.path.exists('token.json'):
        creds = google.oauth2.credentials.Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_user_schedule(service, calendar_id, time_min, time_max):
    events_result = service.events().list(calendarId=calendar_id, timeMin=time_min.isoformat(),
                                          timeMax=time_max.isoformat(), singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])
    return events

def find_common_free_slots(events_a, events_b, time_min, time_max, max_free_minutes=60):
    busy_times = []

    # 解析用户A和用户B的忙碌时间
    for event in events_a:
        if 'dateTime' in event['start']:
            busy_start = parser.isoparse(event['start']['dateTime'])
            busy_end = parser.isoparse(event['end']['dateTime'])
        else:
            busy_start = parser.isoparse(event['start']['date'])
            busy_end = parser.isoparse(event['end']['date'])

        busy_times.append((busy_start, busy_end))

    for event in events_b:
        if 'dateTime' in event['start']:
            busy_start = parser.isoparse(event['start']['dateTime'])
            busy_end = parser.isoparse(event['end']['dateTime'])
        else:
            busy_start = parser.isoparse(event['start']['date'])
            busy_end = parser.isoparse(event['end']['date'])

        busy_times.append((busy_start, busy_end))

    busy_times.sort()

    current_time = time_min
    time_max = time_max
    free_slots = []

    # 查找空闲时间段
    for busy_start, busy_end in busy_times:
        if current_time < busy_start:
            free_slot_end = min(busy_start, time_max)
            free_slot = {'start': current_time, 'end': free_slot_end}
            free_slots.append(free_slot)
        current_time = max(current_time, busy_end)

    if current_time < time_max:
        free_slots.append({'start': current_time, 'end': time_max})

    # 筛选出小于 max_free_minutes 的空闲时间段
    common_free_slots = [slot for slot in free_slots if (slot['end'] - slot['start']).total_seconds() / 60 <= max_free_minutes]

    # 输出调试信息
    for slot in common_free_slots:
        print(f"Free slot: {slot['start']} - {slot['end']}")

    return common_free_slots

def find_route_for_free_slots(events, free_slots):
    routes = []
    for slot in free_slots:
        start_time = slot['start']
        end_time = slot['end']
        
        prev_event = None
        next_event = None

        for event in events:
            event_start = parser.isoparse(event['start'].get('dateTime', event['start'].get('date')))
            event_end = parser.isoparse(event['end'].get('dateTime', event['end'].get('date')))

            if event_end <= start_time:
                prev_event = event
            elif event_start >= end_time:
                next_event = event
                break

        start_location = prev_event['location'] if prev_event and 'location' in prev_event else "Unknown"
        end_location = next_event['location'] if next_event and 'location' in next_event else "Unknown"

        routes.append({
            'start_time': start_time,
            'end_time': end_time,
            'start_location': start_location,
            'end_location': end_location
        })

    return routes

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_free_slots(free_slots, routes_a, routes_b):
    # 确定唯一的日期
    days = sorted(list(set(slot["start"].date() for slot in free_slots)))

    # 绘图
    fig, ax = plt.subplots(figsize=(15, 10))

    # 创建竖向时间线图
    for slot in free_slots:
        start = slot["start"]
        end = slot["end"]
        day_index = days.index(start.date())
        
        # 计算时间段的高度
        start_height = start.hour + start.minute / 60
        end_height = end.hour + end.minute / 60
        
        ax.add_patch(Rectangle((day_index - 0.4, start_height), 0.8, end_height - start_height, 
                               edgecolor='black', facecolor='skyblue'))

    # 显示用户A的路线
    for route in routes_a:
        day_index = days.index(route['start_time'].date())
        start_height = route['start_time'].hour + route['start_time'].minute / 60
        end_height = route['end_time'].hour + route['end_time'].minute / 60
        ax.text(day_index - 0.4, start_height, f"A: {route['start_location']}", ha='right', fontsize=8, color='blue')
        ax.text(day_index - 0.4, end_height, f"A: {route['end_location']}", ha='right', fontsize=8, color='blue')

    # 显示用户B的路线
    for route in routes_b:
        day_index = days.index(route['start_time'].date())
        start_height = route['start_time'].hour + route['start_time'].minute / 60
        end_height = route['end_time'].hour + route['end_time'].minute / 60
        ax.text(day_index + 0.4, start_height, f"B: {route['start_location']}", ha='left', fontsize=8, color='red')
        ax.text(day_index + 0.4, end_height, f"B: {route['end_location']}", ha='left', fontsize=8, color='red')

    # 设置图表格式
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([day.strftime('%Y-%m-%d') for day in days])
    ax.set_yticks(range(8, 19))
    ax.set_yticklabels([f'{h:02}:00' for h in range(8, 19)])
    ax.set_ylim(18, 8)  # 反转y轴
    ax.set_xlim(-0.5, len(days) - 0.5)

    ax.set_title('Free Slots')
    ax.set_xlabel('Date')
    ax.set_ylabel('Time')

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




import googlemaps
import numpy as np

gmaps = googlemaps.Client(key='')

def get_route_coordinates(start, end, mode='bicycling'):
    directions_result = gmaps.directions(start, end, mode=mode)
    route = directions_result[0]['legs'][0]['steps']
    coordinates = []
    for step in route:
        start_location = step['start_location']
        coordinates.append((start_location['lat'], start_location['lng']))
        end_location = step['end_location']
        coordinates.append((end_location['lat'], end_location['lng']))
    return coordinates

def interpolate_coordinates(coords, num_points=10):
    interpolated_coords = []
    for i in range(len(coords) - 1):
        start = coords[i]
        end = coords[i + 1]
        latitudes = np.linspace(start[0], end[0], num_points)
        longitudes = np.linspace(start[1], end[1], num_points)
        interpolated_coords.extend(zip(latitudes, longitudes))
    return interpolated_coords

def find_intersections(route1, route2):
    intersections = []
    max_num = 10
    num = 0
    for coord1 in route1:
        for coord2 in route2:
            if num < max_num:
                if abs(coord1[0] - coord2[0]) < 0.0003 and abs(coord1[1] - coord2[1]) < 0.0003:
                    intersections.append(coord1)
                    num += 1
            else:
                break
    return intersections

def reverse_geocode(lat, lng):
    result = gmaps.reverse_geocode((lat, lng))
    if result:
        return result[0]['formatted_address']
    return None

from collections import Counter
def most_common_element(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0] if most_common else None

def check_routes_for_intersections(routes_a, routes_b, mode="bicycling"):
    intersections_info = []

    for route_a, route_b in zip(routes_a, routes_b):
        print(route_a['start_location'])
        print(route_a['end_location'])
        print(route_b['start_location'])
        print(route_b['end_location'])

        if route_b['start_location'] == 'Unknown':
            route_b['start_location'] = 'On Santa Teresa Street @ Lagunita Court, Stanford, CA 94305'
        if route_b['end_location'] == 'Unknown':
            route_b['end_location'] = 'On Santa Teresa Street @ Lagunita Court, Stanford, CA 94305'
        if route_a['start_location'] == 'Unknown':
            route_a['start_location'] = 'Galvez Mall, Stanford, CA 94305'
        if route_a['end_location'] == 'Unknown':
            route_a['end_location'] = 'Galvez Mall, Stanford, CA 94305'
        

        userA_route = get_route_coordinates(route_a['start_location'], route_a['end_location'], mode="bicycling")
        userB_route = get_route_coordinates(route_b['start_location'], route_b['end_location'], mode="bicycling")
        print(1)

        userA_route_interpolated = interpolate_coordinates(userA_route, num_points=10)
        userB_route_interpolated = interpolate_coordinates(userB_route, num_points=10)

        intersections = find_intersections(userA_route_interpolated, userB_route_interpolated)

        if intersections:
            nearest_places = [reverse_geocode(lat, lng) for lat, lng in intersections]
            most_common_elem = most_common_element(nearest_places)
            intersections_info.append({
                'start_time': route_a['start_time'],
                'end_time': route_a['end_time'],
                'location': most_common_elem[0],
                'count': most_common_elem[1]
            })

    return intersections_info

def create_meetup_events(service, intersections_info, userA_calendar_id, userB_calendar_id):
    for intersection in intersections_info[:1]:
        start_time = intersection['start_time'].isoformat()
        end_time = intersection['end_time'].isoformat()
        location = intersection['location']
        
        event_a = {
            'summary': 'Meetup with Sujin!',
            'location': location,
            'start': {
                'dateTime': start_time,
                'timeZone': 'America/Los_Angeles',  # 根据需要调整时区
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'America/Los_Angeles',  # 根据需要调整时区
            },
            'colorId': '6',  # 粉色
        }

        event_b = {
            'summary': 'Meetup with Jermaine!',
            'location': location,
            'start': {
                'dateTime': start_time,
                'timeZone': 'America/Los_Angeles',  # 根据需要调整时区
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'America/Los_Angeles',  # 根据需要调整时区
            },
            'colorId': '6',  # 粉色
        }

        service.events().insert(calendarId=userA_calendar_id, body=event_a).execute()
        service.events().insert(calendarId=userB_calendar_id, body=event_b).execute()

import json
from dateutil.tz import tzoffset

def write_to_json(data, path):
    for item in data:
        item['start_time'] = item['start_time'].isoformat()
        item['end_time'] = item['end_time'].isoformat()

    # Save to JSON file
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Data has been saved to {path}")

import json
from dateutil.tz import tzoffset
from datetime import date,datetime

def convert_datetime(data):
    converted_data = []
    for item in data:
        converted_item = {
            "start_time": item['start_time'].isoformat().replace("T", " "),
            "end_time": item['end_time'].isoformat().replace("T", " "),
            "location": item['location'],
            "count": item['count']
        }
        converted_data.append(converted_item)
    return converted_data


def save_human_readable(data,saving_path):

    # 找到count最高的条目
    # max_item = max(data, key=lambda x: x['count'])
    max_item = data[0]
    print(f"max_item:{max_item}")

    # 解析start_time
    start_time = datetime.fromisoformat(max_item['start_time'].isoformat())
    # 增加10分钟
    estimated_time = start_time + timedelta(minutes=10)

    # 转换成指定格式
    human_readable = f"Estimated meetup time: {estimated_time.strftime('%Y/%m/%d %I:%M %p')}; Location: \"{max_item['location']}\""

    print(human_readable)

    with open(saving_path, 'w') as txt_file:
        txt_file.write(human_readable)

    print(f"Human readable information has been saved to {saving_path}")

import urllib.parse
def generate_google_maps_embed_link_and_savae(api_key, intersections_info, saving_path):
    place_name = intersections_info[0]['location']
    base_url = "https://www.google.com/maps/embed/v1/place"
    query = urllib.parse.urlencode({'key': api_key, 'q': place_name})
    link = f"{base_url}?{query}"
    with open(saving_path, 'w') as txt_file:
        txt_file.write(link)
    print(f"Location link has been saved to {saving_path}")

    return f"{base_url}?{query}"



def main():
    api_key  = ''
    service = googleapiclient.discovery.build('calendar', 'v3', credentials=authenticate_google())

    time_min = dt.datetime(2024, 5, 5, 0, 0, 0, tzinfo=dt.timezone.utc)
    time_max = dt.datetime(2024, 5, 12, 23, 59, 59, tzinfo=dt.timezone.utc)
    interval_minutes = 60  # You can change this value

    userA_calendar_id = ''
    userB_calendar_id = ''

    userA_schedule = get_user_schedule(service, userA_calendar_id, time_min, time_max)
    userB_schedule = get_user_schedule(service, userB_calendar_id, time_min, time_max)

    print("User 1 Schedule on 2024-06-05:")
    for event in userA_schedule:
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))
        print(f"{start} - {end}: {event.get('summary', 'No Title')}")

    # 打印用户2的时间表
    print("\nUser 2 Schedule on 2024-06-05:")
    for event in userB_schedule:
        start = event['start'].get('dateTime', event['start'].get('date'))
        end = event['end'].get('dateTime', event['end'].get('date'))
        print(f"{start} - {end}: {event.get('summary', 'No Title')}")

    common_free_slots = find_common_free_slots(userA_schedule, userB_schedule, time_min, time_max, interval_minutes)
    
    # print("\nCommon Free Slots:")
    # for slot in common_free_slots:
    #     print(f"Free slot: {slot['start']} - {slot['end']}")

    routes_a = find_route_for_free_slots(userA_schedule, common_free_slots)
    routes_b = find_route_for_free_slots(userB_schedule, common_free_slots)

    # print(routes_a)
    # print(routes_b)
    # plot_free_slots(common_free_slots, routes_a, routes_b)

    intersections_info = check_routes_for_intersections(routes_a, routes_b, mode='bicycling')

    # for info in intersections_info:
    #     print(f"Intersection from {info['start_time']} to {info['end_time']} at {info['location']} with {info['count']} occurrences")

    print(intersections_info)
    
    # converted_intersections_info = convert_datetime(intersections_info)
    # write_to_json(intersections_info,"/Users/jermainezhao/try_calendar/intersections_info.json")
    save_human_readable(intersections_info,"/Users/jermainezhao/try_calendar/human_readable.txt")

    generate_google_maps_embed_link_and_savae(api_key,intersections_info,"/Users/jermainezhao/try_calendar/location_link.txt")

    create_meetup_events(service, intersections_info, userA_calendar_id, userB_calendar_id)

if __name__ == "__main__":
    main()
