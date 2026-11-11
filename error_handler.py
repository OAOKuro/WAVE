# error_handler.py (Tkinter version)
from tkinter import messagebox

class ErrorHandler:
    @staticmethod
    def camera_not_found():
        return messagebox.showerror(
            "Camera Not Found",
            "No webcam detected. Please connect a camera.\n\n"
            "Suggested Actions:\n"
            "1. Check camera connections\n"
            "2. Ensure no other app is using the camera\n"
            "3. Restart the application"
        )

    @staticmethod
    def no_hand_detected():
        return messagebox.showwarning(
            "No Hand Detected",
            "No hand detected in camera view.\n\n"
            "Suggested Actions:\n"
            "1. Show your hand to the camera\n"
            "2. Ensure adequate lighting\n"
            "3. Adjust camera position"
        )

    @staticmethod
    def low_confidence():
        return messagebox.showwarning(
            "Low Confidence",
            "Gesture not recognized clearly.\n\n"
            "Suggested Actions:\n"
            "1. Adjust lighting sensitivity in Calibration tab\n"
            "2. Reposition your hand\n"
            "3. Lower confidence threshold"
        )

    @staticmethod
    def config_error():
        return messagebox.showwarning(
            "Configuration Error",
            "Settings file could not be loaded.\n\n"
            "Suggested Actions:\n"
            "1. Check config.json exists\n"
            "2. Reset to default settings in Calibration tab"
        )