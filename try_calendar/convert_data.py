from datetime import datetime
from dateutil.tz import tzoffset

# 原始数据
data = [
    {'start_time': datetime(2024, 5, 6, 12, 20, tzinfo=tzoffset(None, -25200)),
     'end_time': datetime(2024, 5, 6, 13, 0, tzinfo=tzoffset(None, -25200)),
     'location': '470 Via Ortega, Stanford, CA 94305, USA', 'count': 7},
    {'start_time': datetime(2024, 5, 8, 10, 50, tzinfo=tzoffset(None, -25200)),
     'end_time': datetime(2024, 5, 8, 11, 30, tzinfo=tzoffset(None, -25200)),
     'location': '526 Lasuen Mall, Stanford, CA 94305, USA', 'count': 10},
    {'start_time': datetime(2024, 5, 8, 12, 20, tzinfo=tzoffset(None, -25200)),
     'end_time': datetime(2024, 5, 8, 13, 0, tzinfo=tzoffset(None, -25200)),
     'location': '470 Via Ortega, Stanford, CA 94305, USA', 'count': 7}
]

# 转换函数
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

# 转换数据
converted_data = convert_datetime(data)

# 输出结果
import json
print(json.dumps(converted_data, indent=4))