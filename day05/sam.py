from ultralytics import SAM
import cv2

# Load a model
model = SAM("sam_b.pt")

# Display model information (optional)
model.info()

# Run inference
results = model("./bus.jpg")

# Display the results
cv2.imshow("SAM Inference", results[0].plot())
cv2.waitKey(0)
