import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import os
import numpy as np
from model import AlexNet

# CIFAR-10 클래스 정의
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class ModelInferenceUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AlexNet CIFAR-10 Image Classifier")
        self.root.geometry("600x700")
        self.root.configure(bg='#f0f0f0')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_image = None
        self.original_image = None
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="AlexNet CIFAR-10 Image Classifier",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0'
        )
        title_label.pack(pady=10)
        
        # Device info
        device_label = tk.Label(
            self.root,
            text=f"Device: {self.device}",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='gray'
        )
        device_label.pack()
        
        # Image display frame
        image_frame = tk.Frame(self.root, bg='white', relief=tk.SUNKEN, bd=2)
        image_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(
            image_frame,
            text="Upload an image to see prediction",
            font=("Arial", 12),
            bg='white',
            fg='gray'
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # Upload button
        upload_btn = tk.Button(
            button_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Arial", 11, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=15,
            pady=8,
            relief=tk.RAISED,
            cursor="hand2"
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Predict button
        predict_btn = tk.Button(
            button_frame,
            text="🔍 Predict",
            command=self.predict,
            font=("Arial", 11, "bold"),
            bg='#2196F3',
            fg='white',
            padx=15,
            pady=8,
            relief=tk.RAISED,
            cursor="hand2"
        )
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear,
            font=("Arial", 11, "bold"),
            bg='#f44336',
            fg='white',
            padx=15,
            pady=8,
            relief=tk.RAISED,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        self.result_frame = tk.Frame(self.root, bg='#e8f5e9', relief=tk.SUNKEN, bd=2)
        self.result_frame.pack(padx=10, pady=10, fill=tk.X)
        
        self.result_label = tk.Label(
            self.result_frame,
            text="Prediction: -\nConfidence: -",
            font=("Arial", 12),
            bg='#e8f5e9',
            justify=tk.LEFT,
            padx=10,
            pady=10
        )
        self.result_label.pack(fill=tk.BOTH)
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 9),
            bg='#f0f0f0',
            fg='gray'
        )
        self.status_label.pack(pady=5)
    
    def load_model(self):
        """모델 로드"""
        try:
            self.status_label.config(text="Loading model...")
            self.root.update()
            
            model_path = "./model/alexnet_cifar10.pth"
            
            if not os.path.exists(model_path):
                messagebox.showwarning(
                    "Model Not Found",
                    f"Model file not found at: {model_path}\n\n"
                    "Please run main.py first to train and save the model."
                )
                self.status_label.config(text="Model not found")
                return False
            
            self.model = AlexNet(num_classes=10).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            self.status_label.config(text=f"✓ Model loaded from {model_path}")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_label.config(text="Error loading model")
            return False
    
    def upload_image(self):
        """이미지 선택"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
            )
            
            if file_path:
                self.original_image = Image.open(file_path).convert('RGB')
                self.display_image()
                self.status_label.config(text=f"✓ Image loaded: {os.path.basename(file_path)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def display_image(self):
        """이미지 표시"""
        if self.original_image is None:
            return
        
        # Display image
        display_image = self.original_image.copy()
        display_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def predict(self):
        """예측 수행"""
        if self.model is None:
            messagebox.showwarning("Warning", "Model not loaded. Please check model file.")
            return
        
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        try:
            self.status_label.config(text="Predicting...")
            self.root.update()
            
            # 이미지 전처리
            input_tensor = self.transform(self.original_image).unsqueeze(0).to(self.device)
            
            # 예측
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item() * 100
            
            # 결과 표시
            predicted_class = CIFAR10_CLASSES[predicted_class_idx]
            result_text = f"Prediction: {predicted_class.upper()}\nConfidence: {confidence:.2f}%"
            
            self.result_label.config(text=result_text)
            self.status_label.config(text=f"✓ Prediction complete")
            
            # 전체 확률 출력 (디버깅용)
            print("\n📊 Prediction Results:")
            print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
            print("\nAll class probabilities:")
            for idx, class_name in enumerate(CIFAR10_CLASSES):
                prob = probabilities[0][idx].item() * 100
                print(f"  {class_name:12s}: {prob:6.2f}%")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            self.status_label.config(text="Error during prediction")
    
    def clear(self):
        """초기화"""
        self.original_image = None
        self.image_label.config(image='', text="Upload an image to see prediction", fg='gray')
        self.image_label.image = None
        self.result_label.config(text="Prediction: -\nConfidence: -")
        self.status_label.config(text="Ready")


def main():
    root = tk.Tk()
    app = ModelInferenceUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
