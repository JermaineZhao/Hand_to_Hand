from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')  # 加载YOLOv8模型
model.export(format='onnx')  # 导出为ONNX格式