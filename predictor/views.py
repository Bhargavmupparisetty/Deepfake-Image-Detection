from django.shortcuts import render
import numpy as np
import tensorflow
from django.http import JsonResponse
from PIL import Image
import io
import time
from django.http import FileResponse
from reportlab.pdfgen import canvas
from io import BytesIO
import json
import cv2
import tempfile
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
import uuid
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings



# Step 1: Define the Meso4 architecture
def build_model():
    input_layer = tensorflow.keras.layers.Input(shape=(256, 256, 3))
    x = tensorflow.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(input_layer)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tensorflow.keras.layers.Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tensorflow.keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tensorflow.keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    x = tensorflow.keras.layers.MaxPooling2D((4, 4), padding='same')(x)

    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dropout(0.5)(x)
    x = tensorflow.keras.layers.Dense(16)(x)
    x = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tensorflow.keras.layers.Dropout(0.5)(x)
    output = tensorflow.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tensorflow.keras.models.Model(inputs=input_layer, outputs=output)
    return model

# Step 2: Initialize and load weights
model = build_model()
model.load_weights("predictor/Meso4_rigorous_finetuned_model.h5")

# Step 3: Prediction View


def generate_report(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        prediction = data.get('prediction')
        confidence = data.get('confidence')
        
        # Generate PDF
        buffer = BytesIO()
        p = canvas.Canvas(buffer)
        p.drawString(100, 750, f"Prediction: {prediction}")
        p.drawString(100, 735, f"Confidence: {confidence}")
        p.showPage()
        p.save()
        buffer.seek(0)
        
        return FileResponse(buffer, as_attachment=True, filename='deepfake_report.pdf')
    return JsonResponse({'error': 'Invalid request'}, status=400)




def index(request):
    if request.method == "POST" and request.FILES.get("image"):
        start_time = time.time()
        img_file = request.FILES["image"]
        try:
            # Open and preprocess the image in memory
            img = Image.open(io.BytesIO(img_file.read())).convert("RGB").resize((256, 256))
            img_array = np.array(img) /255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Get prediction from the model
            pred = model.predict(img_array)[0][0] # Sigmoid output between 0 and 1
            processing_time = time.time() - start_time

            # Prepare result
            result = {
                "prediction": "REAL" if pred > 0.8 else "FAKE",
                "confidence": float(pred),  # Confidence score as float
                "processing_time": processing_time  # Time taken in seconds
            }
            # Return JSON for AJAX requests
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse(result)
            # For non-AJAX (fallback), render template
            return render(request, "predictor/index.html", {"prediction": result})
        except Exception as e:
            error_response = {"error": str(e)}
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse(error_response, status=400)
            return render(request, "predictor/index.html", {"error": str(e)})
    # GET request: render the initial page
    return render(request, "predictor/index.html")




@csrf_exempt
@require_POST
def analyze_video(request):
    """
    Process uploaded video file for deepfake detection.
    Analyzes key frames and returns prediction results.
    """
    # Check if video file exists in request
    if 'video' not in request.FILES:
        return JsonResponse({'error': 'No video file uploaded.'}, status=400)

    video_file = request.FILES['video']
    
    # Validate file type
    valid_extensions = ['.mp4', '.avi', '.mov', '.webm']
    file_ext = os.path.splitext(video_file.name)[1].lower()
    if file_ext not in valid_extensions:
        return JsonResponse({
            'error': f'Unsupported file format. Please upload one of: {", ".join(valid_extensions)}'
        }, status=400)
    
    # Create unique filename to prevent overwriting and conflicts in multi-user scenarios
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    temp_video_path = os.path.join(settings.MEDIA_ROOT, 'temp', unique_filename)
    
    # Ensure temp directory exists
    os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
    
    try:
        # Save uploaded video temporarily
        with open(temp_video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        # Open video file
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            os.remove(temp_video_path)
            return JsonResponse({'error': 'Unable to read video file.'}, status=400)

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Start timing the processing
        processing_start = time.time()
        
        # Set parameters for analysis
        max_frames_to_check = 20
        early_stopping_threshold = 0.8  # If 80% of frames agree, we can stop early
        
        # Adaptive interval based on video length
        if total_frames <= 300:  # Short video (<10s at 30fps)
            interval = max(1, total_frames // 15)
        elif total_frames <= 900:  # Medium video (<30s at 30fps)
            interval = max(1, total_frames // 20)
        else:  # Long video
            interval = max(1, total_frames // 25)
        
        print(f"Video analysis starting: {total_frames} total frames, sampling every {interval} frames")
        
        # Initialize counters and storage
        frame_data = []
        fake_count = 0
        real_count = 0
        sampled_frames = 0
        
        # Create directory for frame thumbnails if it doesn't exist
        thumbnails_dir = os.path.join(settings.MEDIA_ROOT, 'thumbnails')
        os.makedirs(thumbnails_dir, exist_ok=True)
        
        # Process in batches to reduce memory usage
        batch_size = 5
        
        for batch_start in range(0, total_frames, interval * batch_size):
            # Process a batch of frames
            batch_frames = []
            
            # Collect frames for this batch
            for i in range(batch_start, min(batch_start + interval * batch_size, total_frames), interval):
                if sampled_frames >= max_frames_to_check:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                batch_frames.append((i, frame))
                
            # Process each frame in the batch
            for i, frame in batch_frames:
                # Generate a thumbnail (limit to every other frame to reduce storage)
                if sampled_frames % 2 == 0:
                    thumbnail_path = os.path.join(thumbnails_dir, f"frame_{unique_filename}_{i}.jpg")
                    try:
                        frame_small = cv2.resize(frame, (160, 90))  # Even smaller thumbnail
                        cv2.imwrite(thumbnail_path, frame_small, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Reduce quality
                        thumbnail_url = f"/media/thumbnails/frame_{unique_filename}_{i}.jpg"
                    except Exception as thumb_error:
                        print(f"Thumbnail error: {str(thumb_error)}")
                        thumbnail_url = None
                else:
                    thumbnail_url = None
                
                # Preprocess frame for model
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (256, 256))
                    frame_array = np.expand_dims(frame_resized / 255.0, axis=0)
                    
                    # Predict using model
                    pred = model.predict(frame_array)[0][0]
                    label = "REAL" if pred > 0.5 else "FAKE"
                    confidence = float(pred) if label == "REAL" else 1.0 - float(pred)
                    
                    # Update counts
                    if label == "FAKE":
                        fake_count += 1
                    else:
                        real_count += 1
                    
                    # Add frame data
                    frame_data.append({
                        "frame_number": i,
                        "time": round(i / fps, 2) if fps > 0 else 0,
                        "prediction": label,
                        "confidence": confidence,
                        "thumbnail_url": thumbnail_url
                    })
                    
                    # Update counter
                    sampled_frames += 1
                    
                    print(f"Processed frame {sampled_frames}/{max_frames_to_check} (frame #{i}): {label}")
                    
                    # Check if we can stop early after processing enough frames
                    if sampled_frames >= 10:  # Need minimum 10 frames for reliable decision
                        ratio = max(fake_count, real_count) / sampled_frames
                        if ratio >= early_stopping_threshold:
                            print(f"Early stopping at {sampled_frames} frames with {ratio:.2f} agreement")
                            break
                            
                except Exception as frame_error:
                    print(f"Frame processing error at frame {i}: {str(frame_error)}")
                    continue
                    
            # Break the batch loop if we've collected enough frames or triggered early stopping
            if sampled_frames >= max_frames_to_check:
                print("Reached maximum frames to check")
                break
                
            # Force garbage collection after each batch
            import gc
            gc.collect()
            
            # Print progress
            print(f"Progress: {min(90, int(90 * sampled_frames / max_frames_to_check))}%")
        
        # Clean up video file
        cap.release()
        try:
            os.remove(temp_video_path)
            print(f"Removed temporary video file: {temp_video_path}")
        except Exception as cleanup_error:
            print(f"Error removing temp file: {str(cleanup_error)}")
        
        print("Video frames processed, calculating final results...")
        
        # Calculate final prediction
        if sampled_frames == 0:
            return JsonResponse({'error': 'Could not analyze any frames from the video.'}, status=400)
            
        fake_ratio = fake_count / sampled_frames
        final_prediction = "FAKE" if fake_ratio > 0.3 else "REAL"  # Consider fake if 30%+ frames are fake
        
        # Calculate overall confidence based on frame predictions
        if final_prediction == "FAKE":
            confidence = min(0.95, fake_ratio)  # Cap at 0.95 for reasonable confidence
        else:
            confidence = min(0.95, 1.0 - fake_ratio)
            
        processing_time = time.time() - processing_start
        
        # Limit frame data to essential information to reduce response size
        compact_frame_data = []
        for frame in frame_data:
            compact_frame_data.append({
                "frame_number": frame["frame_number"],
                "prediction": frame["prediction"],
                "confidence": frame["confidence"],
                "thumbnail_url": frame["thumbnail_url"]
            })
        
        print("Preparing final response...")
        
        # Prepare and return response
        response_data = {
            "status": "success",
            "prediction": final_prediction,
            "confidence": confidence,
            "total_frames": total_frames,
            "fake_frames": fake_count,
            "processing_time": processing_time,
            "video_info": {
                "duration": duration,
                "fps": fps,
                "resolution": f"{width}x{height}"
            },
            "frame_data": compact_frame_data
        }
        
        print("Analysis complete!")
        return JsonResponse(response_data)

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                print(f"Removed temporary video file after error: {temp_video_path}")
            except:
                pass
            
        import traceback
        error_details = traceback.format_exc()
        print(f"Video processing error: {error_details}")
        
        return JsonResponse({
            'error': 'Video processing failed',
            'details': str(e)
        }, status=500)