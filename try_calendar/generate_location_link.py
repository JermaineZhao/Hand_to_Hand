import urllib.parse
import datetime
from dateutil.tz import tzoffset

def generate_google_maps_embed_link(api_key, place_name):
    base_url = "https://www.google.com/maps/embed/v1/place"
    query = urllib.parse.urlencode({'key': api_key, 'q': place_name})
    return f"{base_url}?{query}"

intersections_info = [{'start_time': datetime.datetime(2024, 5, 6, 12, 20, tzinfo=tzoffset(None, -25200)), 'end_time': datetime.datetime(2024, 5, 6, 13, 0, tzinfo=tzoffset(None, -25200)), 'location': '470 Via Ortega, Stanford, CA 94305, USA', 'count': 7}, {'start_time': datetime.datetime(2024, 5, 8, 10, 50, tzinfo=tzoffset(None, -25200)), 'end_time': datetime.datetime(2024, 5, 8, 11, 30, tzinfo=tzoffset(None, -25200)), 'location': '526 Lasuen Mall, Stanford, CA 94305, USA', 'count': 10}, {'start_time': datetime.datetime(2024, 5, 8, 12, 20, tzinfo=tzoffset(None, -25200)), 'end_time': datetime.datetime(2024, 5, 8, 13, 0, tzinfo=tzoffset(None, -25200)), 'location': '470 Via Ortega, Stanford, CA 94305, USA', 'count': 7}]

def generate_google_maps_embed_link_and_savae(api_key, intersections_info, saving_path):
    place_name = intersections_info[0]['location']
    base_url = "https://www.google.com/maps/embed/v1/place"
    query = urllib.parse.urlencode({'key': api_key, 'q': place_name})
    link = f"{base_url}?{query}"
    with open(saving_path, 'w') as txt_file:
        txt_file.write(link)
    print(f"Location link has been saved to {saving_path}")

    return f"{base_url}?{query}"


# 示例用法
api_key = ''  # 请替换为您的实际API密钥
place_name = '285 Panama St, Stanford, CA 94305, USA'
embed_link = generate_google_maps_embed_link_and_savae(api_key, intersections_info,"aa")
print(embed_link)