from ultralytics import YOLO
import os


if __name__ == "__main__":
   # Load the model

   model = YOLO(f'./yolo11n.pt')

   # Training.

   results = model.train(
      data=os.path.abspath(f"./data/data/data.yaml"),
      imgsz=416,
      # epochs=300,
      epochs=200,
      batch=32,
      name='yolov11'
   )
   val_result = model.val()


