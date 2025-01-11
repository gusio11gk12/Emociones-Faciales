import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Configura el backend antes de importar pyplot
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración de la carpeta de carga
UPLOAD_FOLDER = 'static/uploads'  # Carpeta donde se almacenarán los archivos subidos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Extensiones permitidas para las imágenes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Tamaño máximo permitido de archivo: 16MB

# Asegura que la carpeta de carga exista
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Función para validar si un archivo tiene una extensión permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para analizar la imagen facial y detectar emociones
def analyze_face(image_path):
    try:
        # Inicializa MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            min_detection_confidence=0.7
        )

        # Carga el modelo de emociones (FER+ en formato ONNX)
        emotion_model_path = 'emotion-ferplus-8.onnx'
        if not os.path.exists(emotion_model_path):
            raise Exception(f"El modelo de emociones no se encuentra en {emotion_model_path}")
        emotion_model = cv2.dnn.readNetFromONNX(emotion_model_path)

        # Etiquetas de emociones
        emotion_labels = [
            "Neutral", "Happiness", "Surprise", "Sadness", "Anger",
            "Disgust", "Fear", "Contempt"
        ]

        # Carga la imagen y la convierte a RGB
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen.")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape

        # Detecta los puntos clave del rostro
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise Exception("No se detectó ningún rostro en la imagen.")

        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130]

        # Extraer la región del rostro para la detección de emociones
        for face_landmarks in results.multi_face_landmarks:
            x_min = int(min([landmark.x for landmark in face_landmarks.landmark]) * width)
            y_min = int(min([landmark.y for landmark in face_landmarks.landmark]) * height)
            x_max = int(max([landmark.x for landmark in face_landmarks.landmark]) * width)
            y_max = int(max([landmark.y for landmark in face_landmarks.landmark]) * height)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            face_region = gray_image[y_min:y_max, x_min:x_max]
            if face_region.size == 0:
                raise Exception("No se pudo extraer la región del rostro.")

            blob = cv2.dnn.blobFromImage(face_region, scalefactor=1/255.0, size=(64, 64), mean=(0, 0, 0), swapRB=True, crop=True)
            emotion_model.setInput(blob)
            emotion_predictions = emotion_model.forward()
            print("Predicciones de emociones (raw):", emotion_predictions)
            emotion_idx = np.argmax(emotion_predictions)
            detected_emotion = emotion_labels[emotion_idx]
            print("Índice de emoción detectada:", emotion_idx)
            print("Emoción detectada:", detected_emotion)
        # Dibuja puntos clave sobre la imagen original
        keypoint_image = image.copy()
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(keypoint_image, (x, y), 5, (0, 255, 0), -1)

        # Convierte la imagen con puntos clave en base64
        _, buffer = cv2.imencode('.png', keypoint_image)
        keypoint_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepara transformaciones para visualización
        transformations = [
            ("Original", gray_image),
            ("Volteado horizontalmente", cv2.flip(gray_image, 1)),
            ("Aumentado en brillo", cv2.convertScaleAbs(gray_image, alpha=1.2, beta=50)),
            ("Invertido verticalmente", cv2.flip(gray_image, 0))
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for ax, (title, img) in zip(axes, transformations):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return keypoint_image_base64, image_base64, detected_emotion

    except Exception as e:
        print(f"Error en analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

# Ruta principal para mostrar el índice
@app.route('/')
def home():
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

# Ruta para analizar una imagen
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404

        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        else:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

        keypoint_image, transformations_image, detected_emotion = analyze_face(filepath)

        return jsonify({
            'success': True,
            'keypoint_image': keypoint_image,
            'transformations_image': transformations_image,
            'emotion': detected_emotion
        })

    except Exception as e:
        print(f"Error en /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Ruta para servir los archivos subidos
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
