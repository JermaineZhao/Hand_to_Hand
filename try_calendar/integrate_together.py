import googlemaps
import numpy as np
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import google.auth.transport.requests
import google_auth_oauthlib.flow
import googleapiclient.discovery
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser

# 初始化 Google Maps 客户端
gmaps = googlemaps.Client(key='YOUR_GOOGLE_MAPS_API_KEY')

def get_route_coordinates(start, end, mode='driving'):
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
    for coord1 in route1:
        for coord2 in route2:
            if abs(coord1[0] - coord2[0]) < 0.0003 and abs(coord1[1] - coord2[1]) < 0.0003:
                intersections.append(coord1)
    return intersections

def reverse_geocode(lat, lng):
    result = gmaps.reverse_geocode((lat, lng))
    if result:
        return result[0]['formatted_address']
    return None

def get_location_from_event(event):
    # 从事件中获取地点信息，如果没有则返回None
    return event.get('location')

def check_route_intersection(userA_event_before, userA_event_after, userB_event_before, userB_event_after):
    userA_start = get_location_from_event(userA_event_before)
    userA_end = get_location_from_event(userA_event_after)
    userB_start = get_location_from_event(userB_event_before)
    userB_end = get_location_from_event(userB_event_after)

    if not all([userA_start, userA_end, userB_start, userB_end]):
        return None  # 如果有任何一个地点信息缺失，则无法计算路线

    userA_route = get_route_coordinates(userA_start, userA_end, mode="bicycling")
    userB_route = get_route_coordinates(userB_start, userB_end, mode="bicycling")

    userA_route_interpolated = interpolate_coordinates(userA_route, num_points=10)
    userB_route_interpolated = interpolate_coordinates(userB_route, num_points=10)

    intersections = find_intersections(userA_route_interpolated, userB_route_interpolated)

    if intersections:
        nearest_places = [reverse_geocode(intersection[0], intersection[1]) for intersection in intersections]
        from collections import Counter
        most_common_elem = Counter(nearest_places).most_common(1)[0]
        return most_common_elem[0], most_common_elem[1]
    else:
        return None

def main():
    service = authenticate_google()

    time_min = datetime.now(timezone.utc)
    time_max = time_min + timedelta(days=7)  # 查找未来7天的空闲时间
    interval_minutes = 60

    userA_schedule = get_user_schedule(service, 'userA@gmail.com', time_min, time_max)
    userB_schedule = get_user_schedule(service, 'userB@gmail.com', time_min, time_max)

    common_free_slots = find_common_free_slots(userA_schedule, userB_schedule, time_min, time_max, interval_minutes)

    for slot in common_free_slots:
        print(f"\n检查空闲时间段: {slot['start']} - {slot['end']}")

        # 找到这个空闲时间段之前和之后的事件
        userA_event_before = next((event for event in reversed(userA_schedule) if event['end']['dateTime'] <= slot['start'].isoformat()), None)
        userA_event_after = next((event for event in userA_schedule if event['start']['dateTime'] >= slot['end'].isoformat()), None)
        userB_event_before = next((event for event in reversed(userB_schedule) if event['end']['dateTime'] <= slot['start'].isoformat()), None)
        userB_event_after = next((event for event in userB_schedule if event['start']['dateTime'] >= slot['end'].isoformat()), None)

        intersection_result = check_route_intersection(userA_event_before, userA_event_after, userB_event_before, userB_event_after)

        if intersection_result:
            place, count = intersection_result
            print(f"在此空闲时间段，两个用户的路线有交点。")
            print(f"最可能的交点位置是：{place}，出现次数：{count}")
        else:
            print("在此空闲时间段，两个用户的路线没有交点。")

if __name__ == '__main__':
    main()