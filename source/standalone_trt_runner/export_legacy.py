from ultralytics import YOLO

# Load the YOLO model
model = YOLO("/home/dji/DJI_workshop/LZP_PSDK/customed_yolo_model/yolo11x_sangu_best.pt")

# Export the model to ONNX format
export_path = model.export(format="onnx")

print(f"Model exported to {export_path}")