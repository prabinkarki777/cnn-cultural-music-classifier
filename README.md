
# Cultural Music Classification Using CNN

This project classifies different types of cultural music based on their Mel Spectrograms using Convolutional Neural Networks (CNN). It uses audio data (in WAV or MP3 format) and extracts Mel Spectrograms as feature representations, then trains a CNN model to predict the cultural music type.

## Setup Instructions

### 1. Install Dependencies
To get started, you'll need to install the required dependencies. This can be done using `pip` by installing the following packages:

```bash
pip install librosa numpy matplotlib scikit-learn tensorflow
```

You can also install other dependencies like `skimage` and `seaborn` as needed.

### 2. Prepare the Dataset

You will need a folder of audio files from different cultural music types. Place your music files under a directory structure like this:

```
music/
    ├── Deuda/
    ├── Newari/
    ├── Ratauli/
    ├── Tharu/
    └── tamang_selo/
```

Each folder should contain `.wav` or `.mp3` audio files. This script will read all files under these subdirectories and use them to train the model.

### 3. Audio Preprocessing

The following steps are performed on each audio file:

- **Loading**: Audio files are loaded using `librosa` with a sample rate of 22,050 Hz and a duration of 30 seconds.
- **Augmentation**: The script applies augmentation techniques such as time-stretching, pitch-shifting, and adding white noise to the audio.
- **Mel Spectrogram Extraction**: The audio is converted into a Mel Spectrogram representation using `librosa.feature.melspectrogram`.
- **Normalization**: The Mel Spectrograms are normalized to a range between 0 and 1.
  
### 4. Model Training

The model architecture is a CNN with the following layers:

- **Conv2D Layers**: To capture spatial patterns in Mel Spectrograms.
- **MaxPooling2D Layers**: To downsample the feature maps.
- **Leaky ReLU Activations**: For non-linearity and to allow for better gradient flow.
- **Dropout**: To prevent overfitting.
- **Dense Layer**: To classify the spectrograms into one of the cultural music types.

The model is trained using the Adam optimizer with a very small learning rate of 0.0001 and early stopping to prevent overfitting.

### 5. Model Evaluation

After training the model, the following evaluation steps are done:

- **Accuracy**: The training and validation accuracy are plotted.
- **Loss**: The training and validation loss are plotted.
- **Confusion Matrix**: A confusion matrix is displayed to visualize the classification results.
- **ROC Curve**: A multiclass ROC curve is plotted for each class.
- **Precision-Recall Curve**: A precision-recall curve is generated for each cultural music type.

### 6. Model Prediction

After training, the model is saved, and it can be loaded for prediction. The `predict_cultural_music_type()` function accepts the path of an audio file, processes the file into a Mel Spectrogram, and predicts the cultural music type.

Example:
```python
cultural_music_type, confidence = predict_cultural_music_type('path_to_sample_audio.wav')
print(f"Predicted: {cultural_music_type} with {confidence:.2%} confidence")
```

### 7. Sample Output

When running the prediction function on a sample audio file, you can expect output like this:

```
🎵 sample_audio.wav: Deuda (95.67% confidence)
```

The model will display the predicted music type and the confidence percentage.

## Conclusion

This project shows how to build a simple yet effective music classification system using Mel Spectrograms and CNNs. You can easily extend this project to include more cultural music types or apply it to other audio classification tasks.

### Files

- `best_model_leaky.h5`: The trained model.
- `cultural_music_type_label_encoder.pkl`: The label encoder for cultural music types.
