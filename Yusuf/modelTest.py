import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 1️⃣ Load your trained model
model = load_model("best_inceptionresnetv2_dr.h5")

# 2️⃣ Load and preprocess your image
img_path = "C:/Users/kondk/Downloads/archive (2)/split_dataset/test/0/0a85a1e8f9e9.png"  
img = image.load_img(img_path, target_size=(384, 384))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 3️⃣ Make prediction
pred = model.predict(img_array)
class_index = np.argmax(pred)
confidence = np.max(pred)

# 4️⃣ View result
classes = ['stage:0', 'stage:1', 'stage:2', 'stage:3', 'stage:4']  # adjust if your folder names differ
print(f"Prediction: {classes[class_index]} ({confidence:.2%} confidence)")
