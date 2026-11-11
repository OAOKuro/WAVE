import subprocess
import os
import sys

def build():
    sep = os.pathsep  # ';' on Windows, ':' on Mac/Linux

    command = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=WAVE",
        f"--add-data=config.json{sep}.",
        f"--add-data=models{sep}models",
        "main.py",
    ]

    # Optional: uncomment if icon.ico exists in project root
    # command.insert(-1, "--icon=icon.ico")

    print("Building WAVE.exe...")
    subprocess.run(command, check=True)
    print(f"Build complete! Check dist/WAVE{'.exe' if sys.platform == 'win32' else ''}")

if __name__ == "__main__":
    build()