# app.py - Complete Code (with enhanced AI debugging)

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
import os
import re
from dotenv import load_dotenv
import requests
import random # For simulating ML predictions and average rainfall
from openai import OpenAI

# Load environment variables at the very beginning
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'Jaibalayya')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_recycle': 280}

# --- Mail Configuration ---
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME', 'PixelPirates.AST@gmail.com') # Use env var or fallback
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")

# --- Initialize extensions ---
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'signin'
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
mail = Mail(app) # Initialize Mail after app config

# API keys for external services
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
GEOAPIFY_API_KEY = os.getenv('GEOAPIFY_API_KEY')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # For OpenRouter/OpenAI

# Initialize OpenAI client for OpenRouter
# Ensure OPENAI_API_KEY is available before initializing client
if OPENAI_API_KEY:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
else:
    print("WARNING: OPENAI_API_KEY is not set. AI recommendations will not work.")
    client = None # Set client to None if API key is missing

# --- Helper functions for external API calls ---

def get_weather_data(latitude, longitude):
    """Get weather using latitude/longitude."""
    weather_data = None
    weather_error = None
    owm_location_name = None

    if not OPENWEATHER_API_KEY:
        return None, "OpenWeatherMap API Key not configured.", None
    if latitude is None or longitude is None:
        return None, "Latitude and Longitude are required for weather data.", None

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={OPENWEATHER_API_KEY}&units=metric"
        resp = requests.get(url)
        resp.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        weather_data = resp.json()
        if weather_data and 'name' in weather_data:
            owm_location_name = weather_data['name']
    except requests.exceptions.RequestException as e:
        weather_error = f"Could not fetch weather data: {e}"
    except Exception as e:
        weather_error = f"An unexpected error occurred while fetching weather: {e}"

    return weather_data, weather_error, owm_location_name

def get_geoapify_location_name(latitude, longitude):
    """
    Fetches a human-readable location name, state, and district using Geoapify Reverse Geocoding API.
    Prioritizes 'city', then 'suburb', falls back to 'formatted' address for general name.
    Extracts 'state' and 'county' (as district) specifically.
    """
    location_name = None
    state = None
    district = None
    geoapify_error = None

    if not GEOAPIFY_API_KEY:
        return None, None, None, "Geoapify API Key not configured."
    if latitude is None or longitude is None:
        return None, None, None, "Latitude and Longitude are required for Geoapify Geocoding."

    try:
        url = f"https://api.geoapify.com/v1/geocode/reverse?lat={latitude}&lon={longitude}&apiKey={GEOAPIFY_API_KEY}"
        resp = requests.get(url)
        resp.raise_for_status()
        geocode_data = resp.json()

        if geocode_data and 'features' in geocode_data and geocode_data['features']:
            properties = geocode_data['features'][0]['properties']

            if 'city' in properties and properties['city']:
                location_name = properties['city']
            elif 'suburb' in properties and properties['suburb']:
                location_name = properties['suburb']
            elif 'formatted' in properties and properties['formatted']:
                location_name = properties['formatted']
            
            if 'state' in properties and properties['state']:
                state = properties['state']
            
            # Geoapify might use 'county' or 'district' for what we consider a district
            if 'county' in properties and properties['county']:
                district = properties['county']
            elif 'district' in properties and properties['district']:
                district = properties['district']

        elif geocode_data and 'features' in geocode_data and not geocode_data['features']:
            geoapify_error = "No detailed location found for these coordinates by Geoapify."
        else:
            geoapify_error = f"Geoapify API error or unexpected response structure (Status: {geocode_data.get('status', 'N/A')})."

    except requests.exceptions.RequestException as e:
        geoapify_error = f"Could not connect to Geoapify API: {e}"
    except Exception as e:
        geoapify_error = f"An unexpected error occurred with Geoapify: {e}"

    return location_name, state, district, geoapify_error


# --- SIMULATED ML Model Prediction ---
def predict_pest_risk(crop, temperature_c, humidity_percent, state, district, soil_type, avg_rainfall_mm_per_month):
    """
    This is a fully simulated function for pest risk prediction.
    It does NOT use any loaded ML models.
    """
    common_pests = {
        "Sugarcane": ["Sugarcane Borers", "Mealy Bugs", "Whiteflies"],
        "Arhar/Tur": ["Pod Borer", "Aphids"],
        "Wheat": ["Aphids", "Rusts"],
        "Garlic": ["Thrips", "Mites"],
        "Rice": ["Brown Plant Hopper", "Stem Borer"],
        "Bajra": ["Downy Mildew", "Shoot Fly"],
        "Maize": ["Maize Stem Borer", "Fall Armyworm"],
        "Moong(Green Gram)": ["Yellow Mosaic Virus", "Whiteflies"],
        "Onion": ["Thrips", "Purple Blotch"],
        "Gram": ["Pod Borer", "Aphids"],
        "Urad": ["Yellow Mosaic Virus", "Whiteflies"],
        "Jowar": ["Shoot Fly", "Stem Borer"],
        "Sunflower": ["Head Borer", "Rust"],
        "Barley": ["Aphids", "Loose Smut"],
        "Potato": ["Late Blight", "Potato Tuber Moth"],
        "Groundnut": ["Leaf Miner", "Rust"],
        "Soyabean": ["Girdle Beetle", "Yellow Mosaic Virus"],
    }
    
    # Ensure avg_rainfall_mm_per_month has a value for simulation
    if avg_rainfall_mm_per_month is None:
        avg_rainfall_mm_per_month = random.uniform(50, 250)

    severity = random.randint(1, 5) # Base random severity

    # Simulate some logic based on inputs
    if temperature_c is not None and humidity_percent is not None:
        if temperature_c > 28 and humidity_percent > 75:
            severity += random.randint(1, 3)
        if avg_rainfall_mm_per_month > 150:
            severity += random.randint(0, 2)
    if "Black Soil" in soil_type and crop in ["Sugarcane", "Cotton"]: # Dummy rule
        severity += 1

    pest_severity_index = min(severity, 10) # Cap at 10

    likely_pest_species = random.choice(common_pests.get(crop, ["General Pest"]))

    return pest_severity_index, likely_pest_species

# --- Function to generate recommendations using OpenAI (OpenRouter) ---
def generate_pest_recommendations(crop, pest_severity_index, likely_pest_species):
    """
    Generates low-chemical pesticide recommendations and crop management tips
    using an LLM based on predicted pest and severity.
    """
    # Check if the client object was successfully initialized
    if client is None:
        print("DEBUG AI: OpenAI client not initialized. Cannot generate AI recommendations.")
        return "AI recommendations are not available (API key missing or client initialization failed)."

    prompt = (
        f"Based on the occurrence of '{likely_pest_species}' on '{crop}' with a severity index of {pest_severity_index}/10, "
        "provide highly concise, practical recommendations for pest control and crop management.\n\n"
        "**CRITICAL INSTRUCTIONS FOR OUTPUT FORMAT:**\n"
        "1. Focus ONLY on low-chemical, organic, or integrated pest management (IPM) solutions for pesticide recommendations.\n"
        "2. Provide concrete, actionable crop management tips to mitigate the pest and prevent future outbreaks.\n"
        "3. **STRICTLY ADHERE TO THE FOLLOWING MARKDOWN FORMAT (No extra text, introduction, or conclusion):**\n"
        "   **Recommended Pesticides (Low-Chemical Focus):**\n"
        "   * [Pesticide 1 name/type] (e.g., Neem oil, Pyrethrin-based spray, Insecticidal soap)\n"
        "   * [Pesticide 2 name/type] (if applicable, be specific about type)\n"
        "   **Crop Management Tips:**\n"
        "   * [Tip 1] (e.g., Crop rotation, Sanitation, Proper spacing)\n"
        "   * [Tip 2] (e.g., Introduce beneficial insects, Optimize irrigation, Soil health)\n"
        "   * [Tip 3] (e.g., Regular monitoring, Use resistant varieties, Timely weeding)\n"
        "4. Keep the entire response extremely concise: between 8-12 lines in total (including headers and bullet points)."
    )

    try:
        print(f"DEBUG AI: Attempting to call OpenAI API for recommendations. Prompt length: {len(prompt)}") # DEBUG
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free", # Using the model you specified
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=400, # Increased max_tokens slightly to give more room for adherence
            temperature=0.7 # Add temperature to control creativity; lower for more factual/direct
        )
        if response and response.choices and len(response.choices) > 0:
            recommendations_text = response.choices[0].message.content
            print(f"DEBUG AI: Successfully received recommendations from API. Length: {len(recommendations_text)}") # DEBUG
            return recommendations_text
        else:
            print("DEBUG AI: AI API returned an empty or malformed response.") # DEBUG
            return "AI recommendations could not be generated (empty response)."
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error traceback
        print(f"DEBUG AI: AI recommendation generation failed with an exception: {e}") # DEBUG
        return f"AI recommendations could not be generated (error: {e})."


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(512), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Authentication Routes ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        # Basic email format validation and Gmail domain check
        if not re.match(r"^[a-zA-Z0-9._%+-]+@gmail\.com$", email):
            flash('Please enter a valid Gmail address (e.g., example@gmail.com).', 'danger')
            return redirect(url_for('signup'))

        if not isinstance(password, str) or len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('signup'))

        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Account created! Please sign in.', 'success')
        return redirect(url_for('signin'))

    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        # Basic email format validation and Gmail domain check
        if not re.match(r"^[a-zA-Z0-9._%+-]+@gmail\.com$", email):
            flash('Please enter a valid Gmail address.', 'danger')
            return redirect(url_for('signin'))

        # Password length check for user feedback (actual check is check_password_hash)
        if len(password) < 8:
            flash('Invalid email or password.', 'danger') # Generic message for security
            return redirect(url_for('signin'))

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Signed in successfully.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('signin'))

    return render_template('signin.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('signin'))

@app.route('/dashboard')
@login_required
def dashboard():
    # This route is just for demonstration, you might want to integrate it into your main app structure
    return f"Hello, {current_user.email}! <a href='/logout'>Logout</a> <a href='/change_password'>Change Password</a>"

# --- Password Reset Routes ---
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].strip()
        user = User.query.filter_by(email=email).first()
        if user:
            token = s.dumps(user.email, salt='password-reset-salt')
            reset_url = url_for('reset_password', token=token, _external=True)

            msg = Message(
                subject="Password Reset Request",
                sender=app.config['MAIL_USERNAME'],
                recipients=[user.email]
            )
            msg.body = f"""Hello,

You're receiving this e-mail because you or someone else has requested a password reset for your user account.

{reset_url}

If you did not request this password reset, you can ignore this email.

Thanks,
Your feedback analyser Team (CHARAN)
"""
            try:
                mail.send(msg)
                flash('Password reset link has been sent to your email.', 'info')
            except Exception as e:
                flash(f'Something went wrong while sending email: {e}', 'danger')
        else:
            flash('Email not found.', 'danger')
        return redirect(url_for('signin'))

    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600) # Token valid for 1 hour
    except Exception:
        flash('The reset link is invalid or has expired.', 'danger')
        return redirect(url_for('signin'))

    if request.method == 'POST':
        user = User.query.filter_by(email=email).first()
        if user:
            password = request.form['password'].strip()
            if len(password) < 8:
                flash('Password must be at least 8 characters long.', 'danger')
                return render_template('reset_password.html', token=token)
            user.set_password(password)
            db.session.commit()
            flash('Your password has been updated.', 'success')
            return redirect(url_for('signin'))
        else:
            flash('User not found.', 'danger') # Should not happen if token is valid
            return redirect(url_for('signin'))
    return render_template('reset_password.html', token=token)

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        old_password = request.form['old_password'].strip()
        new_password = request.form['new_password'].strip()

        if not current_user.check_password(old_password):
            flash('Old password is incorrect.', 'danger')
            return redirect(url_for('change_password'))
        if len(new_password) < 8:
            flash('New password must be at least 8 characters long.', 'danger')
            return redirect(url_for('change_password'))
        
        current_user.set_password(new_password)
        db.session.commit()
        flash('Password changed successfully. Please sign in again.', 'success')
        logout_user()
        return redirect(url_for('signin'))
    return render_template('change_password.html')

@app.route('/', methods=['GET'])
@login_required
def index():
    # This route is the main entry point after login
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    crop = request.form.get('crop')
    latitude_str = request.form.get('latitude')
    longitude_str = request.form.get('longitude')
    soil_type = request.form.get('soil_type')

    latitude = None
    longitude = None
    
    # Try converting latitude and longitude to float
    if latitude_str and longitude_str:
        try:
            latitude = float(latitude_str)
            longitude = float(longitude_str)
        except ValueError:
            flash("Invalid Latitude or Longitude values. Please ensure location is detected correctly.", 'danger')
            return redirect(url_for('index'))
    else:
        flash("Location not detected. Please click 'Detect My Location' and allow access.", 'danger')
        return redirect(url_for('index'))

    # Fetch weather data
    weather_data, weather_error, owm_location_name = get_weather_data(latitude, longitude)
    if weather_error:
        flash(f"Weather data error: {weather_error}", 'danger')
        return redirect(url_for('index'))

    # Extract temperature and humidity
    temperature_c = weather_data['main']['temp'] if weather_data and 'main' in weather_data else None
    humidity_percent = weather_data['main']['humidity'] if weather_data and 'main' in weather_data else None
    
    # Fetch Geoapify location data
    geoapify_location_name, state, district, geoapify_error = get_geoapify_location_name(latitude, longitude)
    if geoapify_error:
        flash(f"Location details error: {geoapify_error}", 'danger')
        return redirect(url_for('index'))

    # Simulate average rainfall for ML input
    avg_rainfall_mm_per_month_for_ml = random.uniform(50, 250) 

    pest_severity_index = "N/A"
    likely_pest_species = "Cannot predict due to missing data."
    ai_recommendations = "AI recommendations could not be generated."

    # Check if all necessary data for ML prediction is available
    missing_data_for_ml = []
    if not crop: missing_data_for_ml.append("Crop")
    if temperature_c is None: missing_data_for_ml.append("Temperature")
    if humidity_percent is None: missing_data_for_ml.append("Humidity")
    if not state: missing_data_for_ml.append("State")
    if not district: missing_data_for_ml.append("District")
    if not soil_type: missing_data_for_ml.append("Soil Type")
    # avg_rainfall_mm_per_month_for_ml is always simulated, so no need to check it here

    if not missing_data_for_ml: # If list is empty, all data is present
        # Call the simulated ML prediction function
        pest_severity_index, likely_pest_species = predict_pest_risk(
            crop, temperature_c, humidity_percent, state, district, soil_type,
            avg_rainfall_mm_per_month_for_ml
        )
        
        # Generate AI recommendations ONLY if ML prediction was successful and valid
        if (pest_severity_index != "N/A" and 
            "Cannot predict" not in likely_pest_species and 
            "General Pest" not in likely_pest_species and # Exclude generic fallback
            "Unknown Pest" not in likely_pest_species):
            ai_recommendations = generate_pest_recommendations(crop, pest_severity_index, likely_pest_species)
        else:
            ai_recommendations = "AI recommendations could not be generated based on the current prediction."
    else:
        flash(f"Missing data for full pest prediction: {', '.join(missing_data_for_ml)}. Please ensure all inputs are valid and location is detected.", 'danger')
        return redirect(url_for('index'))

    # --- DEBUGGING PRINTS (Optional, you can remove these in production) ---
    print(f"Final Latitude: {latitude}, Final Longitude: {longitude}")
    print(f"Selected Crop: {crop}, Soil Type: {soil_type}")
    print(f"Weather Data: {weather_data}")
    print(f"OpenWeatherMap Location Name: {owm_location_name}")
    print(f"Weather Error: {weather_error}")
    print(f"Geoapify Location Name (Reverse Geocode): {geoapify_location_name}")
    print(f"Detected State: {state}")
    print(f"Detected District: {district}")
    print(f"Geoapify Error: {geoapify_error}")
    print(f"ML Prediction: Severity={pest_severity_index}, Pest={likely_pest_species}")
    print(f"AI Recommendations: {ai_recommendations}")
    # --- END DEBUGGING PRINTS ---

    return render_template(
        'prediction_result.html',
        crop=crop,
        latitude=latitude,
        longitude=longitude,
        soil_type=soil_type,
        weather_data=weather_data,
        weather_error=weather_error,
        geoapify_location_name=geoapify_location_name,
        geoapify_error=geoapify_error,
        state=state,
        district=district,
        pest_severity_index=pest_severity_index, # Pass prediction
        likely_pest_species=likely_pest_species, # Pass prediction
        avg_rainfall_mm_per_month=round(avg_rainfall_mm_per_month_for_ml, 2), # Pass for display
        ai_recommendations=ai_recommendations # Pass AI recommendations to template
    )

if __name__ == '__main__':
    # Create database tables within the application context
    with app.app_context():
        db.create_all()
    
    # Define port from environment or fallback
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True) # Set debug=True for development
