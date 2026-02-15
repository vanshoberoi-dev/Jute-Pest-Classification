import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Reduce TensorFlow logging verbosity and optimize loading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for faster startup
tf.get_logger().setLevel('ERROR')

# Configure TensorFlow for faster loading
tf.config.optimizer.set_jit(False)  # Disable XLA for faster startup

# Set page config
st.set_page_config(
    page_title="Jute Pest Classifier",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class names from your model
CLASS_NAMES = [
    'Beet Armyworm', 'Black Hairy', 'Cutworm', 'Field Cricket', 
    'Jute Aphid', 'Jute Hairy', 'Jute Red Mite', 'Jute Semilooper', 
    'Jute Stem Girdler', 'Jute Stem Weevil', 'Leaf Beetle', 'Mealybug', 
    'Pod Borer', 'Scopula Emissaria', 'Termite', 'Termite odontotermes (Rambur)', 
    'Yellow Mite'
]

@st.cache_resource
def load_model():
    """Load the TFLite model"""
    model_path = 'jute_pest_model.tflite'
    
    # Download model if not exists (for Streamlit Cloud)
    if not os.path.exists(model_path):
        try:
            import requests
            pass  # Download silently
            url = "https://github.com/Vansh462/Jute-Pest-Classification/raw/main/jute_pest_model.tflite"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to 480x480 as per TFLite model requirements
    image = image.resize((480, 480))
    
    # Convert to array and normalize
    image_array = np.array(image)
    
    # Ensure RGB format
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Normalize to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    else:
        return None

def predict_pest(interpreter, image_array):
    """Make prediction using the TFLite interpreter"""
    try:
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(predictions).numpy()
        
        # Get top prediction
        predicted_class_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class_idx])
        
        return predicted_class_idx, confidence, probabilities
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    st.title("🐛 Jute Pest Classifier")
    st.markdown("### Upload an image to identify jute pests")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Vansh462/Jute-Pest-Classification)")

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses an optimized TFLite model to classify 17 different types of jute pests.

        **Supported Pest Types:**
        """)
        for i, pest in enumerate(CLASS_NAMES, 1):
            st.write(f"{i}. {pest}")

        st.markdown("---")
        st.write("**Model Performance:**")
        st.write("- Test Accuracy: 95.5%")
        st.write("- Model: TFLite (Optimized)")
        st.write("- Input Size: 480x480 pixels")
        st.write("- Model Size: ~40MB")

    # Handle model loading with proper UI
    with st.spinner("🤖 Loading TFLite model... This should be fast!"):
        model = load_model()

    # Model loaded silently

    if model is None:
        st.error("Failed to load the model.")
        st.markdown("""
        ### How to fix this issue:

        1. Make sure `jute_pest_model.tflite` exists in the project directory
        2. If missing, run the conversion script: `python convert_to_tflite.py`
        3. Ensure the TFLite file is not corrupted
        """)
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image of a jute pest for classification"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Show image details
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
        
        with col2:
            st.subheader("Prediction Results")
            
            # Preprocess image
            with st.spinner("Processing image..."):
                processed_image = preprocess_image(image)
            
            if processed_image is not None:
                # Make prediction
                with st.spinner("Classifying pest..."):
                    pred_idx, confidence, probabilities = predict_pest(model, processed_image)
                
                if pred_idx is not None:
                    # Display main prediction
                    predicted_pest = CLASS_NAMES[pred_idx]
                    
                    st.success(f"**Predicted Pest:** {predicted_pest}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                    
                    # Show confidence meter
                    st.progress(confidence)
                    
                    # Show top 3 predictions
                    st.subheader("Top 3 Predictions")
                    top_3_indices = np.argsort(probabilities)[-3:][::-1]
                    
                    for i, idx in enumerate(top_3_indices):
                        pest_name = CLASS_NAMES[idx]
                        prob = probabilities[idx]
                        
                        if i == 0:
                            st.write(f"🥇 **{pest_name}**: {prob:.2%}")
                        elif i == 1:
                            st.write(f"🥈 **{pest_name}**: {prob:.2%}")
                        else:
                            st.write(f"🥉 **{pest_name}**: {prob:.2%}")
                    
                    # Show all probabilities in an expandable section
                    with st.expander("View All Class Probabilities"):
                        prob_data = []
                        for i, prob in enumerate(probabilities):
                            prob_data.append({
                                'Pest Type': CLASS_NAMES[i],
                                'Probability': f"{prob:.4f}",
                                'Percentage': f"{prob:.2%}"
                            })
                        
                        # Sort by probability
                        prob_data.sort(key=lambda x: float(x['Probability']), reverse=True)
                        st.table(prob_data)
                
            else:
                st.error("Error processing image. Please make sure it's a valid RGB image.")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Upload an image of a jute pest using the file uploader above
    2. Wait for the model to process and classify the image
    3. View the prediction results and confidence scores
    4. Check the top 3 predictions for alternative possibilities
    
    ### Tips for better results:
    - Use clear, well-lit images
    - Ensure the pest is the main subject of the image
    - Higher resolution images generally work better
    - Avoid blurry or heavily distorted images
    """)

if __name__ == "__main__":
    main()
