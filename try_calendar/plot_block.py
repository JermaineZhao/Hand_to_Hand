# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from matplotlib.patches import Rectangle
# from datetime import datetime, timedelta

# # 定义空闲时间段
# free_slots = [
#     {"start": "2024-05-06 10:50:00", "end": "2024-05-06 11:30:00"},
#     {"start": "2024-05-06 12:20:00", "end": "2024-05-06 13:00:00"},
#     {"start": "2024-05-07 11:00:00", "end": "2024-05-07 11:30:00"},
#     {"start": "2024-05-07 13:20:00", "end": "2024-05-07 13:30:00"},
#     {"start": "2024-05-08 10:50:00", "end": "2024-05-08 11:30:00"},
#     {"start": "2024-05-08 12:20:00", "end": "2024-05-08 13:00:00"},
#     {"start": "2024-05-09 11:00:00", "end": "2024-05-09 11:30:00"},
#     {"start": "2024-05-09 13:20:00", "end": "2024-05-09 13:30:00"},
#     {"start": "2024-05-10 12:20:00", "end": "2024-05-10 13:00:00"}
# ]

# # 转换为datetime对象
# for slot in free_slots:
#     slot["start"] = datetime.strptime(slot["start"], "%Y-%m-%d %H:%M:%S")
#     slot["end"] = datetime.strptime(slot["end"], "%Y-%m-%d %H:%M:%S")

# # 确定唯一的日期
# days = sorted(list(set(slot["start"].date() for slot in free_slots)))

# # 绘图
# fig, ax = plt.subplots(figsize=(15, 10))

# # 创建竖向时间线图
# for slot in free_slots:
#     start = slot["start"]
#     end = slot["end"]
#     day_index = days.index(start.date())
#     start_time = start.time()
#     end_time = end.time()
    
#     # 计算时间段的高度
#     start_height = start_time.hour + start_time.minute / 60
#     end_height = end_time.hour + end_time.minute / 60
    
#     ax.add_patch(Rectangle((day_index - 0.4, start_height), 0.8, end_height - start_height, 
#                            edgecolor='black', facecolor='skyblue'))

# # 设置图表格式
# ax.set_xticks(range(len(days)))
# ax.set_xticklabels([day.strftime('%Y-%m-%d') for day in days])
# ax.set_yticks(range(8, 19))
# ax.set_yticklabels([f'{h:02}:00' for h in range(8, 19)])
# ax.set_ylim(18, 8)  # 反转y轴
# ax.set_xlim(-0.5, len(days) - 0.5)

# ax.set_title('Free Slots from 2024-05-06 to 2024-05-10')
# ax.set_xlabel('Date')
# ax.set_ylabel('Time')

# plt.grid(True, axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_free_slots(free_slots):
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

# 示例使用
if __name__ == "__main__":
    # 创建 UTC-7 时区
    tz = timezone(timedelta(hours=-7))
    
    # 你的 free_slots 数据
    free_slots = [
        {'start': datetime(2024, 5, 6, 10, 50, tzinfo=tz), 'end': datetime(2024, 5, 6, 11, 30, tzinfo=tz)},
        {'start': datetime(2024, 5, 6, 12, 20, tzinfo=tz), 'end': datetime(2024, 5, 6, 13, 0, tzinfo=tz)},
        {'start': datetime(2024, 5, 7, 11, 0, tzinfo=tz), 'end': datetime(2024, 5, 7, 11, 30, tzinfo=tz)},
        {'start': datetime(2024, 5, 7, 13, 20, tzinfo=tz), 'end': datetime(2024, 5, 7, 13, 30, tzinfo=tz)},
        {'start': datetime(2024, 5, 8, 10, 50, tzinfo=tz), 'end': datetime(2024, 5, 8, 11, 30, tzinfo=tz)},
        {'start': datetime(2024, 5, 8, 12, 20, tzinfo=tz), 'end': datetime(2024, 5, 8, 13, 0, tzinfo=tz)},
        {'start': datetime(2024, 5, 9, 11, 0, tzinfo=tz), 'end': datetime(2024, 5, 9, 11, 30, tzinfo=tz)},
        {'start': datetime(2024, 5, 9, 13, 20, tzinfo=tz), 'end': datetime(2024, 5, 9, 13, 30, tzinfo=tz)},
        {'start': datetime(2024, 5, 10, 12, 20, tzinfo=tz), 'end': datetime(2024, 5, 10, 13, 0, tzinfo=tz)}
    ]

    plot_free_slots(free_slots)