import numpy as np
import cv2
import os
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping

MODEL_PATH = 'digit_recognizer.h5'
WINDOW_SIZE = 450  # 450x450 drawing window

def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Reshape and normalize
    train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
    
    # Add padding to make digits more centered
    train_images = np.array([cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0) for img in train_images])
    test_images = np.array([cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0) for img in test_images])
    
    return (train_images, train_labels), (test_images, test_labels)

def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(36, 36, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

def train_and_save_model():
    (train_images, train_labels), _ = load_data()
    model = build_model()
    
    # EarlyStopping callback to stop training early if validation loss doesn't improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    model.fit(train_images, train_labels, epochs=50, batch_size=128, validation_split=0.1, callbacks=[early_stopping])
    model.save(MODEL_PATH)
    return model

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    return None

def drawing_window(model):
    img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 1), dtype=np.float32)
    drawing = False
    last_prediction = None
    
    def draw(event, x, y, flags, param):
        nonlocal drawing, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(img, (x, y), 15, (1,), -1)  # Larger brush for bigger window
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(img, (x, y), 15, (1,), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
    
    window_name = f"Draw Digit - {WINDOW_SIZE}x{WINDOW_SIZE} (Press 'p' to predict, 'c' to clear)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw)
    
    while True:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Show last prediction if available
        if last_prediction is not None:
            digit, conf = last_prediction
            cv2.putText(display_img, f"Prediction: {digit}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(display_img, f"Confidence: {conf*100:.1f}%", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        else:
            # Show instructions when no prediction
            cv2.putText(display_img, "Draw a digit", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display_img, "Press 'p' to predict", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display_img, "Press 'c' to clear", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Clear everything
            img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 1), dtype=np.float32)
            last_prediction = None
        elif key == ord('p'):
            # Preprocess and predict
            resized = cv2.resize(img, (36, 36))
            gray = (resized * 255).astype(np.uint8)
            
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                roi = gray[y:y+h, x:x+w]
                
                if h > w:
                    resized_roi = cv2.resize(roi, (int(w*20/h), 20))
                else:
                    resized_roi = cv2.resize(roi, (20, int(h*20/w)))
                
                # Center in 36x36
                h_pad = (36 - resized_roi.shape[0]) // 2
                w_pad = (36 - resized_roi.shape[1]) // 2
                centered = np.zeros((36, 36), dtype=np.uint8)
                centered[h_pad:h_pad+resized_roi.shape[0], w_pad:w_pad+resized_roi.shape[1]] = resized_roi
                
                # Predict
                input_img = centered.reshape(1, 36, 36, 1).astype('float32') / 255
                pred = model.predict(input_img, verbose=0)
                digit = np.argmax(pred)
                conf = np.max(pred)
                last_prediction = (digit, conf)
            else:
                last_prediction = None  # If no contours found
                
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    # Try to load existing model
    model = load_trained_model()
    
    # If model doesn't exist, train and save it
    if model is None:
        print("No trained model found. Training a new model...")
        model = train_and_save_model()
        print("Model trained and saved successfully!")
    else:
        print("Loaded pre-trained model successfully!")
    
    # Start the drawing interface
    drawing_window(model)
