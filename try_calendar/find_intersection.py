import googlemaps
import numpy as np

gmaps = googlemaps.Client(key='')

def get_route_coordinates(start, end, mode='driving'):
    print(1111)
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

a_start = "Building 200, 50 Embarcadero Rd, Palo Alto, CA 94301, USA"
a_end = "Building 370, Stanford University, Stanford, CA 94305, USA"
userA_route = get_route_coordinates(a_start,a_end,mode="bicycling")
# userB_route = get_route_coordinates('Crothers Hall, Stanford, CA 94305, USA', 'NVIDIA Auditorium, 475 Via Ortega, Stanford, CA 94305, USA')
userB_route = get_route_coordinates('West Campus Tennis Courts, 188 Electioneer Wy, Stanford, CA 94305', 'Duan Family Hall (EVGR, Bldg A), 757 Campus Drive, Stanford, CA 94305',mode="bicycling")

# 插值以使路线更细
userA_route_interpolated = interpolate_coordinates(userA_route, num_points=10)
userB_route_interpolated = interpolate_coordinates(userB_route, num_points=10)

# A: Cantor Art Center -> FloMo
# B: tennis court -> EVGR A

print("用户A的骑行路线坐标：", len(userA_route))
print("用户B的骑行路线坐标：", len(userB_route))

print("用户A的骑行路线坐标：", len(userA_route_interpolated))
print("用户B的骑行路线坐标：", len(userB_route_interpolated))

def find_intersections(route1, route2):
    intersections = []
    for coord1 in route1:
        for coord2 in route2:
            if abs(coord1[0] - coord2[0]) < 0.0003 and abs(coord1[1] - coord2[1]) < 0.0003:
                intersections.append(coord1)
    return intersections

intersections = find_intersections(userA_route_interpolated, userB_route_interpolated)

if intersections:
    print("交叉点坐标：", intersections)
else:
    print("没有找到交叉点")

def reverse_geocode(lat, lng):
    # 逆地理编码将坐标转换为地名
    result = gmaps.reverse_geocode((lat, lng))
    if result:
        return result[0]['formatted_address']
    return None

nearest_places = []
for intersection in intersections:
    nearest_place = reverse_geocode(intersection[0], intersection[1])
    # print("最近的地名是：", nearest_place)
    nearest_places.append(nearest_place)

from collections import Counter
def most_common_element(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)  # 返回出现次数最多的元素和它的计数
    return most_common[0] if most_common else None

most_common_elem = most_common_element(nearest_places)
print("最近的地名是：", most_common_elem[0], "，次数为：", most_common_elem[1])