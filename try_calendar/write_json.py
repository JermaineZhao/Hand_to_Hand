import json
from datetime import datetime, timedelta
from dateutil.tz import tzoffset

# 你的数据
data = [
    {
        'start_time': datetime(2024, 5, 6, 12, 20, tzinfo=tzoffset(None, -25200)).isoformat(),
        'end_time': datetime(2024, 5, 6, 13, 0, tzinfo=tzoffset(None, -25200)).isoformat(),
        'location': '470 Via Ortega, Stanford, CA 94305, USA',
        'count': 7
    },
    {
        'start_time': datetime(2024, 5, 8, 10, 50, tzinfo=tzoffset(None, -25200)).isoformat(),
        'end_time': datetime(2024, 5, 8, 11, 30, tzinfo=tzoffset(None, -25200)).isoformat(),
        'location': '526 Lasuen Mall, Stanford, CA 94305, USA',
        'count': 6
    },
    {
        'start_time': datetime(2024, 5, 8, 12, 20, tzinfo=tzoffset(None, -25200)).isoformat(),
        'end_time': datetime(2024, 5, 8, 13, 0, tzinfo=tzoffset(None, -25200)).isoformat(),
        'location': '470 Via Ortega, Stanford, CA 94305, USA',
        'count': 7
    }
]

def save_human_readable(data,saving_path):

    # 找到count最高的条目
    # max_item = max(data, key=lambda x: x['count'])
    max_item = data[0]

    # 解析start_time
    start_time = datetime.fromisoformat(max_item['start_time'])
    # 增加10分钟
    estimated_time = start_time + timedelta(minutes=10)

    # 转换成指定格式
    human_readable = f"Estimated meetup time: {estimated_time.strftime('%Y/%m/%d %I:%M %p')}; \nLocation: \"{max_item['location']}\""

    print(human_readable)

    with open(saving_path, 'w') as txt_file:
        txt_file.write(human_readable)

    print("Human readable information has been saved to human_readable.txt")

save_human_readable(data,'human_read.txt')
# # 还原数据中的datetime对象
# for item in data:
#     item['start_time'] = datetime.fromisoformat(item['start_time'])
#     item['end_time'] = datetime.fromisoformat(item['end_time'])

# # 保存到JSON文件
# with open('data.json', 'w') as json_file:
#     json.dump(data, json_file, indent=4, default=str)

# print("Data has been saved to data.json")