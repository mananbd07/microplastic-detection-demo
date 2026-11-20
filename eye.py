# FILE NAME: eye.py
# This file defines our Classical "Eyes" component.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# --- 1. Define Global Classical Configuration ---

# This MUST match the N_QUBITS variable in your brain.py file.
N_OUTPUT_FEATURES = 4 

# This will be the standard size of all our cropped "particle candidate" images.
# Our data prep script (next step) and our final app will resize all blobs to this shape.
IMG_SHAPE = (64, 64, 3) 

print("--- Classical Eye Configuration ---")
print(f"Input Image Shape: {IMG_SHAPE}")
print(f"Output Feature Vector Size: {N_OUTPUT_FEATURES} (to match Qubits)")
print("-----------------------------------")


# --- 2. Define the Feature Extractor Function ---

def build_classical_feature_extractor():
    """
    This function builds our "Eyes" using Transfer Learning.
    It loads a pre-trained model and adds our new 4-feature head.
    """
    
    # 1. Load the pre-trained "Backbone" (expert eye)
    # We are using EfficientNetV2B0, a powerful and lightweight model.
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,        # This is the "chop off the head" command.
        weights='imagenet',       # Download the expert pre-trained weights.
        input_shape=IMG_SHAPE
    )
    
    # 2. Freeze the Backbone
    # We don't want to re-train the "expert eye"; we just want to use it (for now).
    base_model.trainable = False
    
    # 3. Add our new "Custom Head"
    # We stack our own layers on top of the backbone's output.
    inputs = base_model.input                   # Define the model's entry point
    x = base_model.output                     # Get the backbone's final feature map
    x = layers.GlobalAveragePooling2D()(x)    # Flatten that map into a single vector
    x = layers.Dense(128, activation='relu')(x) # Add one "thinking" layer
    x = layers.Dropout(0.2)(x)                  # Add dropout to prevent overfitting
    
    # 4. This is the final and most important layer.
    # It compresses all data down to EXACTLY 4 numbers.
    feature_output = layers.Dense(
        N_OUTPUT_FEATURES, 
        activation='linear',  # 'linear' means just output the raw numbers
        dtype=tf.float64,     # CRITICAL: Match our quantum brain's float64 precision
        name="feature_vector_output"
    )(x)

    # 5. Create the final "Eyes" model
    model = keras.Model(inputs=inputs, outputs=feature_output, name="Classical_Eyes_Extractor")
    return model

# --- 3. Verification Test (This only runs if you run "python eye.py" directly) ---
if __name__ == "__main__":
    
    print("\n--- Running component verification test ---")
    
    # Check for GPU
    gpu_list = tf.config.list_physical_devices('GPU')
    if not gpu_list:
        print("❌ WARNING: TensorFlow did not find GPU in this script.")
    else:
        print(f"✅ GPU Found: {gpu_list[0].name}")
    
    # 1. Build the model
    classical_model = build_classical_feature_extractor()
    print("\n✅ Classical 'Eyes' model built successfully.")
    print("This will download EfficientNet (about 24MB) if this is your first run.")
    
    # 2. Test the model on one "dummy image"
    print("\nTesting model on one dummy 64x64 image...")
    dummy_image_tf = tf.random.normal(shape=(1, 64, 64, 3), dtype=tf.float64)

    # 3. Get the prediction (the 4-number vector)
    feature_vector = classical_model(dummy_image_tf)
    
    print("\n--- Test Run Complete ---")
    print(f"Input image shape: {dummy_image_tf.shape}")
    print(f"Output feature vector (shape={feature_vector.shape}):")
    print(feature_vector.numpy()) # This should be a list of 4 numbers
    print("\n✅✅✅ STEP 2 COMPLETE. Your classical 'eyes' are built.")