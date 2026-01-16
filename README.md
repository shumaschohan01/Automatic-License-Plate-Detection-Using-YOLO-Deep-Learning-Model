# ⛑️ AI-Powered Helmet Detection System

An end-to-end computer vision application that detects whether motorcycle riders or construction workers are wearing helmets. This project uses the **YOLOv11 (Nano)** architecture for high-speed inference and **Streamlit** for a user-friendly web interface.



## 🌟 Key Features
* **Real-time Detection:** Process images in milliseconds using the YOLOv11n model.
* **Dual-Class Detection:** Specifically trained to identify both `Helmets` and `Persons`.
* **Interactive UI:** Simple drag-and-drop image uploader built with Streamlit.
* **Smart Alerts:** Provides clear visual feedback (Success/Warning) based on safety compliance.
* **Cloud Ready:** Fully optimized for deployment on Streamlit Community Cloud.

## 🚀 Live Demo
You can try out the live application here:(https://automatic-license-plate-detection-using-yolo-deep-learning-mod.streamlit.app/)

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning Framework:** [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
* **Web Framework:** [Streamlit](https://streamlit.io/)
* **Image Processing:** OpenCV, PIL
* **Model Version:** YOLO11n (Nano)

## 📁 Project Structure
```text
├── .streamlit/           # Streamlit configuration
├── Helmetbest.pt        # Trained YOLOv11 model weights
├── main.py              # Main Streamlit application code
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
