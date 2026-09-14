from ultralytics import YOLO

yolo_model = YOLO(r"Computer Vision Code\YOLOv26\YOLO26_Dataset Version2\Preprocessing2\YOLO26_M_V2_V2\weights\best.pt")  # Load a pretrained YOLOv8n model

pred = yolo_model.predict(source=r"Raw Data\2025\07\31\20250731_060000_Ic_flat_4k.jpg", save=True)  # Predict on an image and save the results