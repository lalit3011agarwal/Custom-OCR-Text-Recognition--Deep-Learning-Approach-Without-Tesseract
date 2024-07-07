# OCR Text Recognition 
Created By: Lalit Agarwal

## Dataset Exploration
- The model is trained on the [TextOCR dataset](https://textvqa.org/textocr/).
- The dataset includes diverse and complex text in various scenes, which challenges the OCR model to recognize text accurately.
- Text appears in different fonts, sizes, and orientations, increasing the difficulty for the OCR system.
- The dataset contains a significant amount of noise and occlusions, requiring robust preprocessing steps.

## Preprocessing
Below are the preprocessing steps performed on the images:
- Converted images to grayscale for consistent input to the OCR model.
- Applied Gaussian blur to reduce noise and improve text region detection.
- Used adaptive thresholding to create binary images, highlighting text regions.
- Extracted text regions using contour detection to locate bounding boxes around potential text.
- Cropped and resized text regions to a fixed height while maintaining aspect ratio, followed by padding to standardize input size for the model.

## Model Architecture
- The OCR model is a Convolutional Recurrent Neural Network (CRNN) combining convolutional layers for feature extraction and recurrent layers for sequence modeling.
- The model is trained with a Connectionist Temporal Classification (CTC) loss function to handle varying lengths of text sequences.
- Character mappings are created using TensorFlow's `StringLookup` for encoding and decoding predicted text sequences.

## Text Prediction
- The model processes each cropped text region and predicts character sequences.
- Uses TensorFlow's CTC decoding to convert model predictions into readable text.
- The final text output is obtained by concatenating decoded sequences from all detected text regions in the image.

## Tools and Libraries
- This project does not use any pre-built OCR tools like Tesseract.
- All text recognition tasks are implemented using custom models and TensorFlow/Keras libraries.

## Limitations
- Text region detection may miss or inaccurately capture text in highly cluttered or noisy images.
- The model's accuracy can be affected by variations in text appearance, such as unusual fonts or extreme distortions.
- Limited ability to handle multilingual text due to the training dataset's focus on a specific character set.

## Potential Model Improvements
- Training the model on a more extensive and diverse dataset could improve its robustness and accuracy.
- Enhancing preprocessing steps to better handle noise and occlusions in images.
- Incorporating advanced techniques for multilingual OCR to support a wider range of languages and scripts.
- Fine-tuning the model with domain-specific text data for better performance in targeted applications.
