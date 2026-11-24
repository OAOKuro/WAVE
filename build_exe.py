import os
import subprocess
import sys


def build():
    sep = os.pathsep  # ';' on Windows, ':' on Mac/Linux

    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name=WAVE",
        # Bundle MediaPipe C++ graph files (.binarypb) and .tflite models
        "--collect-all=mediapipe",
        # Bundle Faster-Whisper / CTranslate2 binaries & tokenizers
        "--collect-all=faster_whisper",
        # Hidden imports for scikit-learn MLP & local modules
        "--hidden-import=sklearn.utils._typedefs",
        "--hidden-import=sklearn.neighbors._partition_nodes",
        "--hidden-import=calibration_panel",
        "--hidden-import=error_handler",
        "--hidden-import=train_mlp",
        "--hidden-import=download_data",
        # Bundle project data folders/files
        f"--add-data=config.json{sep}.",
        f"--add-data=models{sep}models",
        "main.py",
    ]

    # Automatically include icon.ico if present in project root
    if os.path.exists("icon.ico"):
        command.insert(-1, "--icon=icon.ico")

    print("Building WAVE.exe...")
    subprocess.run(command, check=True)
    print(f"Build complete! Check dist/WAVE{'.exe' if sys.platform == 'win32' else ''}")


if __name__ == "__main__":
    build()