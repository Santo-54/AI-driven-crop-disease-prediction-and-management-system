"""
Flask Web Application for AI Crop Disease Prediction and Management
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import io

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables to store loaded models
paddy_model = None
tomato_model = None
chilli_model = None

# --- 1. HARDCODED CLASS LISTS ---

# Tomato Classes
TOMATO_CLASS_LIST = [
    'Tomato_Yellow Leaf_Curl_Virus',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
    'invalid'
]

# Chilli Classes
CHILLI_CLASS_LIST = [
    'chilies_healthy',
    'chilies_leaf curl',
    'chilies_leaf spot',
    'chilies_whitefly',
    'chilies_yellowish',
    'invalid'
]

# Paddy Classes (UPDATED WITH YOUR CLASSES)
# IMPORTANT: These must be in the exact alphabetical order of your training folders.
PADDY_CLASS_LIST = [
    'Bacterial Blight',
    'Brown Spot',
    'Healthy',
    'Hispa',
    'Leaf Blast',
    'Leaf Scald',
    'Leaf Smut',
    'Narrow Brown Spot',
    'Tungro',
    'invalid'
]

# --- DISEASE MANAGEMENT TIPS ---
DISEASE_TIPS = {
    # --- TOMATO ---
    'Tomato_Yellow Leaf_Curl_Virus': {'name': 'Tomato Yellow Leaf Curl Virus', 'prevention': [
        'Install yellow sticky traps @ 12/ha to monitor and trap the whitefly vector.'
        'Cover the nursery with 40-mesh nylon net to prevent whitefly entry during the seedling stage.'
        'Spray Imidacloprid 17.8 SL @ 3 ml/10 liters or Thiamethoxam 25 WG @ 4 g/10 liters to control whiteflies.'
        'Remove and destroy infected plants immediately (roguing) to prevent the virus from spreading to healthy plants.'
], 'treatment': ['Remove infected plants'], 'pesticide': ['Imidacloprid'], 'soil': ['Crop rotation']},
    'Tomato___Bacterial_spot': {'name': 'Bacterial Spot', 'prevention': [
'Seed treatment: Soak seeds in hot water (50°C) for 25 minutes, then dry before sowing to kill seed-borne bacteria.'
'Spray Streptocycline 1 g mixed with Copper Oxychloride 30 g in 10 liters of water at 15-day intervals.'
'Avoid overhead sprinkler irrigation, as splashing water spreads the bacteria from leaf to leaf.'
'Follow crop rotation with non-solanaceous crops (like maize or beans) for at least 2-3 years.'
], 'treatment': ['Copper bactericides'], 'pesticide': ['Copper Hydroxide'], 'soil': ['Good drainage']},
    'Tomato___Early_blight': {'name': 'Early Blight', 'prevention': [
        'Spray Mancozeb 75 WP @ 2 g/liter or Chlorothalonil 75 WP @ 2 g/liter of water at 10-15 day intervals.'
        'Remove and destroy the lower, older leaves which are usually the first to be infected.'
        'Mulch the soil around the base of the plants with straw or plastic to prevent soil-borne spores from splashing onto leaves during rain.'
        'Maintain proper plant spacing to ensure sunlight and air circulation reach the lower leaves.'
], 'treatment': ['Mancozeb'], 'pesticide': ['Chlorothalonil'], 'soil': ['Well-drained']},
    'Tomato___Late_blight': {'name': 'Late Blight', 'prevention': [
        'Prophylactic spray (prevention) of Mancozeb 75 WP @ 2 g/liter of water during cool, cloudy weather.'
        'If disease appears, spray Metalaxyl 8% + Mancozeb 64% (Ridomil MZ) @ 2 g/liter of water.'
        'Stake the plants to keep foliage off the ground and improve air circulation.'
        'Destroy cull piles (discarded potatoes or tomatoes) near the field, as these are primary sources of infection.'
], 'treatment': ['Metalaxyl'], 'pesticide': ['Mancozeb'], 'soil': ['Avoid waterlogging']},
    'Tomato___Leaf_Mold': {'name': 'Leaf Mold', 'prevention': [
        'Improve air circulation by pruning excessive foliage and increasing spacing between plants.'
        'Spray Carbendazim 50 WP @ 1 g/liter or Copper Oxychloride 50 WP @ 3 g/liter of water.'
        'Reduce humidity in polyhouses by ensuring proper ventilation; avoid wetting the leaves when watering.'
        'Collect and burn infected crop residues after harvest to reduce overwintering spores.'
], 'treatment': ['Fungicides'], 'pesticide': ['Azoxystrobin'], 'soil': ['Proper drainage']},
    'Tomato___Septoria_leaf_spot': {'name': 'Septoria Leaf Spot', 'prevention': [
        'Remove and destroy infected lower leaves to slow the upward spread of the fungus.'
        'Spray Zineb 75 WP or Mancozeb 75 WP @ 2 g/liter of water.'
        'Keep the field weed-free, as some weeds (like horse nettle) can host the fungus.'
        'Rotate crops for 2 years away from tomato, potato, and eggplant.'

], 'treatment': ['Copper fungicides'], 'pesticide': ['Mancozeb'], 'soil': ['Sanitation']},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'name': 'Spider Mites', 'prevention': [
        'Spray a strong jet of water on the undersides of leaves to dislodge mites and destroy their webs.'
        'Spray Dicofol 18.5 EC @ 2.5 ml/liter or Spiromesifen 22.9 SC @ 1 ml/liter of water.'
        'Apply Neem oil 3% (30 ml/liter) ensuring coverage on the undersurface of leaves.'
        'Avoid excessive use of synthetic pyrethroids, which can kill natural mite predators and cause outbreaks.'

], 'treatment': ['Acaricides'], 'pesticide': ['Abamectin'], 'soil': ['Moisture']},
    'Tomato___Target_Spot': {'name': 'Target Spot', 'prevention': [
        'Spray Azoxystrobin 23 SC @ 1 ml/liter or Pyraclostrobin 20 WG @ 1 g/liter of water.'
        'Remove infected plant debris and burn it; do not compost it.'
        'Ensure adequate calcium application, as calcium deficiency can make plants more susceptible to lesions.'
        'Maintain uniform soil moisture to prevent plant stress.'
], 'treatment': ['Mancozeb'], 'pesticide': ['Chlorothalonil'], 'soil': ['Sanitation']},
    'Tomato___Tomato_mosaic_virus': {'name': 'Mosaic Virus', 'prevention': [
        'Seed treatment: Soak seeds in a 10% solution of Trisodium Phosphate (TSP) for 30 minutes to inactivate the virus on the seed coat.'
        'Wash hands thoroughly with soap and water before handling plants, especially if you use tobacco products (tobacco mosaic virus is related).'
        'Soak seeds in a 10% milk solution or spray skimmed milk on seedlings to minimize contact transmission during handling.'
        'Sterilize pruning tools and stakes with a 10% bleach solution between plants.'
], 'treatment': ['Remove plants'], 'pesticide': ['None'], 'soil': ['Disinfect']},
    'Tomato___healthy': {'name': 'Healthy Tomato', 'prevention': [], 'treatment': [], 'pesticide': [], 'soil': []},

    # --- CHILLI ---
    'chilies_healthy': {'name': 'Healthy Chilli', 'prevention': [
        'Maintain proper spacing of 60 x 45 cm to ensure good air circulation and reduce humidity around the plants.'
        'Apply balanced fertilizers (NPK 120:60:60 kg/ha) in split doses to maintain plant vigor.'
        'Regularly monitor the field for early signs of pests or diseases.'], 'treatment': [], 'pesticide': [], 'soil': []},
    'chilies_leaf curl': {'name': 'Leaf Curl', 'prevention': [
        'Seed treatment with Imidacloprid 70 WS @ 5 g/kg seed to protect seedlings from sucking pests like thrips and mites.',
        'Grow 2-3 rows of maize or sorghum as a barrier crop around the chilli field to restrict the movement of vectors.',
        'Spray Neem oil 3% or Neem Seed Kernel Extract (NSKE) 5% to repel vectors.',
        'Install blue sticky traps @ 12/ha for thrips and yellow sticky traps @ 12/ha for whiteflies.',], 'treatment': ['Remove plants'], 'pesticide': ['Imidacloprid'], 'soil': []},
    'chilies_leaf spot': {'name': 'Leaf Spot', 'prevention': [
        'Seed treatment with Thiram or Captan @ 4 g/kg of seed to eliminate seed-borne infection.',
        'Spray Mancozeb 0.2% (2 g/l) or Copper Oxychloride 0.25% (2.5 g/l) at 15-day intervals starting from disease appearance.',
        'Collect and burn infected plant debris to reduce the primary source of inoculum.',
        'Follow crop rotation with non-solanaceous crops like cereals or pulses.'], 'treatment': ['Fungicides'], 'pesticide': ['Mancozeb'], 'soil': []},
    'chilies_whitefly': {'name': 'Whitefly', 'prevention': [
        'Install yellow sticky traps @ 12/ha coated with castor oil or grease to trap adult whiteflies.',
        'Spray Neem oil 3% or Fish Oil Rosin Soap 25 g/litre of water.',
        'Spray Imidacloprid 17.8 SL @ 3 ml/10 litres or Thiamethoxam 25 WG @ 4 g/10 litres of water.',
        'Avoid excessive application of nitrogenous fertilizers, as this promotes succulent growth which attracts whiteflies.'], 'treatment': ['Insecticides'], 'pesticide': ['Neem Oil'], 'soil': []},
    'chilies_yellowish': {'name': 'Yellowing', 'prevention': [
        'Foliar spray of Magnesium Sulfate (10 g/l) + Urea (10 g/l) if yellowing is due to magnesium deficiency (interveinal chlorosis).',
        'Spray Ferrous Sulfate (5 g/l) neutralized with lime if yellowing is due to iron deficiency (young leaves turn yellow).',
        'Improve soil drainage immediately if yellowing is caused by waterlogging.',
        'Apply well-decomposed farmyard manure (FYM) @ 25 t/ha during last ploughing to improve soil health.'], 'treatment': ['Micronutrients'], 'pesticide': [], 'soil': ['Check pH']},
    
    # --- PADDY (NEWLY ADDED) ---
    'Bacterial Blight': {
        'name': 'Bacterial Leaf Blight', 
        'prevention': ['Seed treatment with bleaching powder (100g/l) and zinc sulfate (2%) reduce bacterial blight.',
                        'Seed treatment - seed soaking for 8 hours in Agrimycin (0.025%) and wettable ceresan (0.05%) followed by hot water treatment for 30 min at 52-54oC.'
                        'Spray neem oil 3% or NSKE 5%'
                        'Spray fresh cowdung extract for the control of bacterial blight. Dissolve 20 g cowdung in one litre of water; allow to settle and sieve. Use supernatant liquid.'], 
        'treatment': ['Immediately drain the water from the field.', 
            'Allow the field to dry out for 3-4 days to stop the spread.'], 
        'pesticide': ['Spray Streptocycline (1g) mixed with Copper Oxychloride (30g) in 10 liters of water.',
            'Copper Hydroxide sprays.'], 
        'soil': ['Avoid excess Nitrogen',
                 'Keep field drained during infection.']
    },
    'Brown Spot': {
        'name': 'Brown Spot', 
        'prevention': ['Seed treatment with Pseudomonas fluorescens @ 10g/kg of seed followed by seedling dip @ of 2.5 kg or products/ha dissolved in 100 litres and dipping for 30 minutes.',
                        'seed soak / seed treatment with Captan or  Thiram at 2.0g /kg of seed'
                        'Since the fungus is seed transmitted, a hot water seed treatment (53-54°C) for 10-12 minutes may be effective before sowing. This treatment controls primary infection at the seedling stage. Presoaking the seed in cold water for 8 hours increases effectivity of the treatment.'], 
        'treatment': ['Apply Potassium (potash) fertilizer, as this disease often happens in weak soils.'
                       'Remove weeds from the field bunds.'], 
        'pesticide': ['Spray Mancozeb (2.5g/liter) or Carbendazim (1g/liter).',
            'Apply Propiconazole if infection is severe.'], 
        'soil': ['Add Potassium fertilizer; avoid water stress.']
    },
    'Healthy': {
        'name': 'Healthy Paddy', 
        'prevention': ['Continue your current good practices.', 'Monitor fields weekly.'], 'treatment': ['None required.'], 'pesticide': ['None required.'], 'soil': ['Maintain proper water levels and fertilization.']
    },
    'Hispa': {
        'name': 'Rice Hispa', 
        'prevention': ['Remove grassy weeds', 'Monitor early'], 
        'treatment': ['Clip off the tips of seedlings before transplanting (eggs are often laid there).',
            'Use a sweeping net to catch beetles in the morning.'
], 
        'pesticide': [ 'Spray Chlorpyriphos 20 EC (2.5ml/liter).',
            'Spray Quinalphos 25 EC (2ml/liter).'], 
        'soil': []
    },
    'Leaf Blast': {
        'name': 'Leaf Blast', 
        'prevention': ['Avoid drying field', 'Resistant varieties'], 
        'treatment': [ 'Flood the field immediately (Blast loves dry soil).',
            ' Burn straw and stubble of infected crops after harvest.'
], 
        'pesticide': [ 'Spray Tricyclazole 75 WP (0.6g/liter) - most effective.',
            'Spray Isoprothiolane 40 EC (1.5ml/liter).'
], 
        'soil': ['Avoid high Nitrogen'
                 'Keep soil wet/flooded to reduce severity.']
    },
    'Leaf Scald': {
        'name': 'Leaf Scald', 
        'prevention': ['Clean seeds', 'Weed control'], 
        'treatment': ['Fungicides'], 
        'pesticide': ['Benomyl', 'Carbendazim'], 
        'soil': ['Drainage']
    },
    'Leaf Smut': {
        'name': 'Leaf Smut', 
        'prevention': [ 'Use clean, healthy seeds.',
            'Avoid planting too closely (give plants space to breathe).'], 
        'treatment': ['Copper fungicides'], 
        'pesticide': ['Copper Oxychloride'], 
        'soil': ['Deep ploughing']
    },
    'Narrow Brown Spot': {
        'name': 'Narrow Brown Spot', 
        'prevention': [ 'Use disease-free seeds.',
            'Collect and burn straw after harvest.'
], 
        'treatment': ['Foliar spray'], 
        'pesticide': ['Propiconazole', 'Carbendazim'], 
        'soil': ['Potassium application']
    },
    'Tungro': {
        'name': 'Rice Tungro Disease', 
        'prevention': [], 
        'treatment': ['Destroy infected plants', 'Light traps'], 
        'pesticide': ['Imidacloprid (for vectors)'], 
        'soil': ['Crop rotation']
    },

    'invalid': {'name': 'Invalid Image', 'prevention': [], 'treatment': ['Please upload a valid crop leaf image.'], 'pesticide': [], 'soil': []}
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_file):
    image = Image.open(io.BytesIO(image_file.read()))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = img_array.astype('float32') 
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_model_custom(crop_type):
    """Load appropriate model based on crop type"""
    global paddy_model, tomato_model, chilli_model
    
    # Path configuration
    if crop_type == 'tomato':
        model_path = 'model/tomato_disease_model.h5' 
    elif crop_type == 'chilli':
        model_path = 'model/chilli_disease_model.h5' 
    elif crop_type == 'paddy':
        # Make sure your file is named exactly this in your model folder!
        model_path = 'model/paddy_disease_model.h5'
    else:
        model_path = f'model/{crop_type}_disease_model.h5'
    
    # Debug print
    print(f"Attempting to load model from: {model_path}")

    # Check file existence
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Model file not found at {model_path}")
        return None
    
    # Load Logic
    if crop_type == 'paddy':
        if paddy_model is None:
            paddy_model = keras.models.load_model(model_path)
        return paddy_model
        
    elif crop_type == 'tomato':
        if tomato_model is None:
            tomato_model = keras.models.load_model(model_path)
        return tomato_model
        
    elif crop_type == 'chilli':
        if chilli_model is None:
            chilli_model = keras.models.load_model(model_path)
        return chilli_model
    
    return None


def get_class_names(crop_type):
    if crop_type == 'tomato':
        return TOMATO_CLASS_LIST
    elif crop_type == 'paddy':
        return PADDY_CLASS_LIST
    elif crop_type == 'chilli':
        return CHILLI_CLASS_LIST
    return []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'crop_type' not in request.form:
            return jsonify({'error': 'Please select a crop type'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Normalize Input
        raw_crop_type = request.form['crop_type'].lower().strip()
        
        if raw_crop_type in ['chilly', 'chilies', 'chilli']:
            crop_type = 'chilli'
        elif raw_crop_type in ['tomato', 'tomatoes']:
            crop_type = 'tomato'
        elif raw_crop_type in ['paddy', 'rice']:
            crop_type = 'paddy'
        else:
            crop_type = raw_crop_type

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # 2. Load Model
        model = load_model_custom(crop_type)
        if model is None:
            return jsonify({'error': f'Model for {crop_type} not found. Check server logs.'}), 404
        
        # 3. Get Class Names
        class_names = get_class_names(crop_type)
        if not class_names:
            return jsonify({'error': f'Class list for {crop_type} is empty'}), 404
        
        # 4. Preprocess & Predict
        img_array = preprocess_image(file)
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        predicted_class = class_names[predicted_class_idx]
        
        # 5. Handle "Invalid" class
        if predicted_class == 'invalid':
            return render_template('result.html',
                             crop_type=crop_type.capitalize(),
                             disease_name="Invalid Image / Not a Leaf",
                             confidence=round(confidence, 2),
                             prevention=[], treatment=["Please upload a clear photo of a single leaf."],
                             pesticides=[], soil_recommendations=[])

        # 6. Get Tips
        tips = DISEASE_TIPS.get(predicted_class, {
            'name': predicted_class.replace('_', ' ').title(),
            'prevention': ['Consult agricultural expert'],
            'treatment': ['Isolate plant'],
            'pesticide': [],
            'soil': []
        })
        
        if 'name' not in tips: tips['name'] = predicted_class.replace('_', ' ').title()
        
        return render_template('result.html',
                             crop_type=crop_type.capitalize(),
                             disease_name=tips['name'],
                             confidence=round(confidence, 2),
                             prevention=tips.get('prevention', []),
                             treatment=tips.get('treatment', []),
                             pesticides=tips.get('pesticide', []),
                             soil_recommendations=tips.get('soil', []))
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


if __name__ == '__main__':
    print("AI Crop Disease App Started...")
    app.run(debug=True, host='0.0.0.0', port=5000)