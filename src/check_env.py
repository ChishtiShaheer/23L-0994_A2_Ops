"""
Task 1.1: Environment & GPU Verification Script
Author: 23L-0994
Project: 23L-0994_A2_Ops
"""
import sys
import platform

def verify_environment():
    print("=" * 60)
    print("VisionGuard (23L-0994_A2_Ops) - Environment Verification")
    print("=" * 60)
    print(f"Python Version      : {platform.python_version()}")
    print(f"Python Executable   : {sys.executable}")
    print(f"Operating System    : {platform.system()} {platform.release()} ({platform.machine()})")
    print("-" * 60)

    # Core libraries check
    libraries = [
        ("tensorflow", "TensorFlow"),
        ("keras", "Keras"),
        ("mlflow", "MLflow"),
        ("dagshub", "DagsHub"),
        ("dvc", "DVC"),
        ("flask", "Flask"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn")
    ]

    for mod_name, display_name in libraries:
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "Installed")
            print(f"Library: {display_name:<16} : Version {version}")
        except ImportError:
            print(f"Library: {display_name:<16} : NOT INSTALLED")

    print("-" * 60)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow GPU Check : DETECTED {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  [{i}] {gpu.name}")
        else:
            print("TensorFlow GPU Check : NO GPU DETECTED (Running on CPU)")
    except Exception as e:
        print(f"TensorFlow GPU Check : Error checking GPU: {e}")

    print("=" * 60)
    print("Status: Environment verification complete.")

if __name__ == "__main__":
    verify_environment()
