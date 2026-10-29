from ultralytics import YOLO

# Load a small pretrained YOLO model
model = YOLO("yolo11n.pt")

# Run a test prediction on a sample image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device="cpu",
    save=True,
    project="runs",
    name="test_detection"
)

# Print detected objects
for result in results:
    print("Detected objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2f}")

print("YOLO test completed!")