# AlexNet CIFAR-10 Model Testing Guide

## Setup & Usage

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
Run the training script to train and save the model:
```bash
python main.py
```
This will:
- Download CIFAR-10 dataset
- Train AlexNet for 5 epochs
- Save the trained model to `./model/alexnet_cifar10.pth`

### Step 3: Test the Model with UI
After training completes, run the inference UI:
```bash
python inference_ui.py
```

## UI Features

### 📁 Upload Image Button
- Click to select any image from your computer
- Supports: JPG, JPEG, PNG, BMP, GIF formats
- Image will be automatically resized to 32×32 (CIFAR-10 format)

### 🔍 Predict Button
- Analyzes the uploaded image
- Shows predicted class and confidence percentage
- Displays all class probabilities in console

### 🗑️ Clear Button
- Clears the current image and prediction
- Resets the UI

## CIFAR-10 Classes
The model can classify images into 10 classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Example Usage

1. Start the UI: `python inference_ui.py`
2. Click "📁 Upload Image" and select an image
3. Click "🔍 Predict" to get the classification
4. See the prediction result with confidence percentage
5. The console also shows all class probabilities for detailed analysis

## Tips

- **Best results** with clear, centered objects similar to CIFAR-10 training images
- **The model works best** with images containing one main object
- **Confidence score** above 70% generally indicates reliable predictions
- Check the console output for detailed probability scores for all classes

## Troubleshooting

### Model file not found?
- Make sure you ran `main.py` first to train the model
- The model should be saved at `./model/alexnet_cifar10.pth`

### Prediction seems wrong?
- The model was trained on small 32×32 CIFAR-10 images
- Works better with images similar to CIFAR-10 dataset
- If confidence is low, the model is uncertain about the classification

### GPU/CUDA issues?
- The UI automatically detects GPU availability
- If no GPU, it will use CPU (slower but works)
- Check the "Device: " indicator in the UI
