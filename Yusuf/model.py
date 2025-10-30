# ============================================================
# DIABETIC RETINOPATHY STAGE CLASSIFICATION
# USING InceptionResNetV2 (TensorFlow/Keras)
# ============================================================

import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============================================================
# 1️⃣ Data Generators
# ============================================================

train_dir = "C:/Users/kondk/Downloads/archive (2)/split_dataset_processed/train"
val_dir   = "C:/Users/kondk/Downloads/archive (2)/split_dataset_processed/val"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(384, 384),
    batch_size=16,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(384, 384),
    batch_size=16,
    class_mode='categorical'
)

# ============================================================
# 2️⃣ Model Definition: InceptionResNetV2 Base + Custom Head
# ============================================================

base_model = InceptionResNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(384, 384, 3)
)

# Freeze most layers for initial training
for layer in base_model.layers[:-50]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)  # helps prevent overfitting
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ============================================================
# 3️⃣ Compile Model
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 4️⃣ Callbacks for Stable Training
# ============================================================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    ModelCheckpoint("best_inceptionresnetv2_dr.h5", save_best_only=True, monitor='val_loss', mode='min')
]

# ============================================================
# 5️⃣ Train
# ============================================================

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 6️⃣ Fine-tuning: Unfreeze Entire Base for Last 5–10 Epochs
# ============================================================

for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

model.save("final_inceptionresnetv2_dr_model.h5")

print("✅ Model training complete. Best weights saved as best_inceptionresnetv2_dr.h5")
