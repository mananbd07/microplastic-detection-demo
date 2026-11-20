# FILE NAME: brain.py
# This file defines our Quantum "Brain" component.

import tensorflow as tf
import pennylane as qml
import numpy as np

# --- 1. Define Global Quantum Configuration ---
N_QUBITS = 4          # Number of qubits. This also defines our feature input size.
N_QUANTUM_LAYERS = 3  # Number of trainable quantum layers.
DEV = qml.device("default.qubit", wires=N_QUBITS) # This is our software simulator

print("--- Quantum Brain Configuration ---")
print(f"Qubits: {N_QUBITS}")
print(f"Trainable Quantum Layers: {N_QUANTUM_LAYERS}")
print("---------------------------------")

# --- 2. Define the Quantum "Brain" Circuit (VQC) ---
@qml.qnode(DEV, interface='tensorflow', diff_method='backprop')
def quantum_classifier_head(inputs, weights):
    """
    This is our pure PennyLane Quantum Brain (VQC).
    It is compiled as a native TensorFlow function.
    It takes 4 features and runs them through a trainable circuit.
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(wires=0))

print("✅ Full Hybrid QNode ('quantum_classifier_head') defined.")

# --- 3. Define a Function to Create Quantum Weights ---
def get_quantum_weights_variable():
    """Creates the trainable weights for the quantum circuit."""
    weight_shape = (N_QUANTUM_LAYERS, N_QUBITS, 3)
    
    q_weights = tf.Variable(
        tf.random.normal(shape=weight_shape, dtype=tf.float64), 
        trainable=True, 
        name="quantum_weights",
        dtype=tf.float64
    )
    return q_weights

# --- 4. Verification Test (This only runs if you run "python brain.py" directly) ---
if __name__ == "__main__":
    
    print("\n--- Running component verification test ---")
    
    gpu_list = tf.config.list_physical_devices('GPU')
    if not gpu_list:
        print("❌ WARNING: TensorFlow did not find GPU in this script.")
    else:
        print(f"✅ GPU Found: {gpu_list[0].name}")
        
    weights = get_quantum_weights_variable()
    print(f"Created weights variable with shape: {weights.shape}")
    
    features = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float64)
    print(f"Created dummy feature vector (shape={features.shape})")
    
    output = quantum_classifier_head(features, weights)
    print(f"\n✅ QNode Test Run Successful. Output: {output.numpy()}")
    
    drawer = qml.draw(quantum_classifier_head)
    print("\n--- Quantum Circuit Diagram ---")
    print(drawer(features, weights))