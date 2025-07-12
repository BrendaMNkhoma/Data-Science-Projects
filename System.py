# -------------------------------
# 📦 Package Imports
# -------------------------------
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
from datetime import date, datetime, timedelta
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import ImageEnhance
import hashlib
import os
import sys
import time
import shutil
import re
import uuid
import json
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

# -------------------------------
# ⚙️ Configuration Constants
# -------------------------------
DB_NAME = "cataract_system.db"
MODEL_DIR = r"D:/cataract-system/SAVED MODELS"  # Updated model directory
MODEL_FILENAME = "MobileNetV2.h5"  # Model filename
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)  # Full model path
CLASS_NAMES = ['conjunctival_growth', 'mild', 'normal', 'severe']
SESSION_TIMEOUT_MINUTES = 60  # Increased timeout to 1 hour
REPO_ROOT = Path(__file__).parent  # Added for path resolution

# -------------------------------
# 🔐 Enhanced Authentication Functions
# -------------------------------
def hash_password(password):
    """Hash password using SHA256 with salt for better security"""
    salt = "admin_salt"  # In production, use a unique salt per user
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed

def create_user(full_name, email, password, role='assistant'):
    """Create new user account (pending approval)"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('''INSERT INTO users (full_name, email, password, role, status)
                          VALUES (?, ?, ?, ?, ?)''', 
                          (full_name, email.lower(), hash_password(password), role, 'pending'))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def approve_user(user_id, admin_id):
    """Approve a pending user registration"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE users 
            SET status = 'approved',
                approved_by = ?,
                approved_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (admin_id, user_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error approving user: {e}")
        return False
    finally:
        conn.close()

def reject_user(user_id, admin_id):
    """Reject a pending user registration"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE users 
            SET status = 'rejected',
                approved_by = ?,
                approved_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (admin_id, user_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error rejecting user: {e}")
        return False
    finally:
        conn.close()

def get_pending_registrations():
    """Get all pending user registrations"""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM users WHERE status = 'pending'", conn)
        return df
    except Exception as e:
        st.error(f"Error getting pending registrations: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_user_by_email(email):
    """Get user by email with all fields (case-insensitive)"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM users WHERE LOWER(email) = LOWER(?)', (email,))
        user = cursor.fetchone()
        return user
    except Exception as e:
        st.error(f"Error getting user by email: {str(e)}")
        return None
    finally:
        conn.close()

def get_approved_users(role=None):
    """Get all approved users, optionally filtered by role"""
    conn = sqlite3.connect(DB_NAME)
    try:
        if role:
            df = pd.read_sql_query("SELECT * FROM users WHERE status = 'approved' AND role = ?", 
                                  conn, params=(role,))
        else:
            df = pd.read_sql_query("SELECT * FROM users WHERE status = 'approved'", conn)
        return df
    except Exception as e:
        st.error(f"Error getting approved users: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def create_session_token(user_id):
    """Create a new session token for the user"""
    token = hashlib.sha256(f"{user_id}{datetime.now()}{os.urandom(16)}".encode()).hexdigest()
    expires_at = datetime.now() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        # Delete any existing tokens for this user
        cursor.execute("DELETE FROM session_tokens WHERE user_id = ?", (user_id,))
        # Insert new token
        cursor.execute('''
            INSERT INTO session_tokens (user_id, token, expires_at)
            VALUES (?, ?, ?)
        ''', (user_id, token, expires_at))
        conn.commit()
        return token
    except Exception as e:
        print(f"Error creating session token: {e}")
        return None
    finally:
        conn.close()

def validate_session_token(token):
    """Validate a session token and return user if valid"""
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.* FROM users u
            JOIN session_tokens st ON u.id = st.user_id
            WHERE st.token = ? AND st.expires_at > CURRENT_TIMESTAMP
        ''', (token,))
        user = cursor.fetchone()
        return user
    except Exception as e:
        print(f"Error validating session token: {e}")
        return None
    finally:
        conn.close()

def delete_session_token(token):
    """Delete a session token (logout)"""
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM session_tokens WHERE token = ?", (token,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting session token: {e}")
        return False
    finally:
        conn.close()

def verify_session():
    """Verify and maintain session state across page refreshes"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Check for existing session cookie
    if not st.session_state.logged_in and 'auth_token' in st.session_state:
        user = validate_session_token(st.session_state.auth_token)
        if user:
            st.session_state.logged_in = True
            st.session_state.user_email = user[2]
            st.session_state.user_name = user[1]
            st.session_state.user_role = user[4]
            st.session_state.user_id = user[0]
            return True
    
    return st.session_state.logged_in

def login_user(email, password):
    """Set session state after successful login"""
    user = get_user_by_email(email)
    if not user:
        st.error("Invalid email or password")
        return False
    
    # Verify password
    if not verify_password(password, user[3]):
        st.error("Invalid email or password")
        return False
    
    # Check account status
    if user[5] != 'approved':
        st.error("Your account is pending approval")
        return False
    
    # Create session token
    token = create_session_token(user[0])
    if not token:
        st.error("Failed to create session")
        return False
    
    # Set session state
    st.session_state.logged_in = True
    st.session_state.user_email = user[2]
    st.session_state.auth_token = token
    st.session_state.user_name = user[1]
    st.session_state.user_role = user[4]
    st.session_state.user_id = user[0]
    
    return True

def logout_user():
    """Clear session state on logout"""
    if 'auth_token' in st.session_state:
        delete_session_token(st.session_state.auth_token)
    st.session_state.clear()
    st.session_state.logged_in = False

# -------------------------------
# 🗄️ Database Initialization
# -------------------------------
def init_database():
    """Initialize the database with schema migrations and automatic repairs"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Create migrations table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY,
                version INTEGER NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Get current version
        cursor.execute("SELECT MAX(version) FROM migrations")
        result = cursor.fetchone()
        current_version = result[0] if result[0] is not None else 0
        
        # Migration 1: Initial schema (v1)
        if current_version < 1:
            try:
                # Create tables with new schema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        full_name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        role TEXT DEFAULT 'assistant',
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        approved_by INTEGER,
                        approved_at TIMESTAMP,
                        FOREIGN KEY (approved_by) REFERENCES users(id)
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        full_name TEXT NOT NULL,
                        gender TEXT,
                        age INTEGER,
                        village TEXT,
                        traditional_authority TEXT,
                        district TEXT,
                        marital_status TEXT,
                        registration_date DATE DEFAULT CURRENT_DATE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id INTEGER,
                        detection_date DATE DEFAULT CURRENT_DATE,
                        result TEXT,
                        confidence REAL,
                        attended_by TEXT,
                        notes TEXT,
                        FOREIGN KEY (patient_id) REFERENCES patients(id)
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS appointments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id INTEGER,
                        patient_name TEXT,
                        gender TEXT,
                        age INTEGER,
                        village TEXT,
                        traditional_authority TEXT,
                        district TEXT,
                        marital_status TEXT,
                        appointment_date DATE,
                        appointment_time TEXT,
                        booked_by TEXT,
                        doctor_email TEXT,
                        notes TEXT,
                        status TEXT DEFAULT 'Pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (patient_id) REFERENCES patients(id)
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sender_email TEXT NOT NULL,
                        receiver_email TEXT NOT NULL,
                        subject TEXT,
                        message TEXT,
                        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_deleted BOOLEAN DEFAULT 0,
                        deleted_at TIMESTAMP,
                        deleted_by TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS message_attachments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_id INTEGER NOT NULL,
                        file_name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size INTEGER,
                        file_type TEXT,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (message_id) REFERENCES messages(id)
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS session_tokens (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        token TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                ''')
                
                # Create admin user
                cursor.execute('''
                    INSERT OR IGNORE INTO users (
                        full_name, email, password, role, status, approved_by
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    'Admin User', 
                    'admin@gmail.com', 
                    hash_password("admin123"),
                    'admin',
                    'approved',
                    1  # self-approved
                ))
                
                # Record migration
                cursor.execute("INSERT INTO migrations (version) VALUES (1)")
                conn.commit()
                current_version = 1
            except Exception as e:
                conn.rollback()
                raise Exception(f"Migration 1 failed: {str(e)}")
        
        # Migration 2: Add model_versions and system_logs tables (v2)
        if current_version < 2:
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL,
                        description TEXT,
                        release_notes TEXT,
                        path TEXT NOT NULL,
                        uploaded_by INTEGER,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        performance_metrics TEXT,
                        FOREIGN KEY (uploaded_by) REFERENCES users(id)
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_id INTEGER,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                ''')
                
                cursor.execute("INSERT INTO migrations (version) VALUES (2)")
                conn.commit()
                current_version = 2
            except Exception as e:
                conn.rollback()
                raise Exception(f"Migration 2 failed: {str(e)}")
        
        # Migration 3: Ensure admin password is correct (v3)
        if current_version < 3:
            try:
                # Verify admin password
                cursor.execute("SELECT id, password FROM users WHERE email = 'admin@gmail.com'")
                admin = cursor.fetchone()
                
                if admin:
                    admin_id, current_password = admin
                    if not verify_password("admin123", current_password):
                        cursor.execute('''
                            UPDATE users SET password = ? WHERE id = ?
                        ''', (hash_password("admin123"), admin_id))
                        conn.commit()
                
                cursor.execute("INSERT INTO migrations (version) VALUES (3)")
                conn.commit()
                current_version = 3
            except Exception as e:
                conn.rollback()
                raise Exception(f"Migration 3 failed: {str(e)}")
        
        return True
        
    except Exception as e:
        st.error(f"Database initialization failed: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

# Initialize database on startup
if not init_database():
    st.error("Failed to initialize database. Please check the logs.")
    st.stop()
        
# 👥 Patient Management
# -------------------------------
def add_patient(full_name, gender, age, village, traditional_authority, district, marital_status):
    """Add new patient to database"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('''INSERT INTO patients (full_name, gender, age, village, traditional_authority, district, marital_status)
                          VALUES (?, ?, ?, ?, ?, ?, ?)''',
                          (full_name, gender, age, village, traditional_authority, district, marital_status))
        patient_id = cursor.lastrowid
        conn.commit()
        return patient_id
    except Exception as e:
        st.error(f"Error adding patient: {str(e)}")
        return None
    finally:
        conn.close()

def get_patients():
    """Get all patients"""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY registration_date DESC", conn)
        return df
    except Exception as e:
        st.error(f"Error getting patients: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_patient_by_id(patient_id):
    """Get patient by ID"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        patient = cursor.fetchone()
        return patient
    except Exception as e:
        st.error(f"Error getting patient by ID: {str(e)}")
        return None
    finally:
        conn.close()

# -------------------------------
# 👁️ Detection Functions
# -------------------------------
@st.cache_resource
def load_detection_model():
    """Load the cataract detection model with caching"""
    try:
        # Check for active model in session state first
        if 'current_model' in st.session_state and os.path.exists(st.session_state.current_model):
            model = load_model(st.session_state.current_model)
            return model
        
        # Fall back to default model
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            st.session_state.current_model = MODEL_PATH
            return model
        
        st.error("No valid model found")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_active_model_info():
    """Get information about the currently active model"""
    active_model = st.session_state.get('current_model', MODEL_PATH)
    
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT version, description, uploaded_at, 
                   (SELECT full_name FROM users WHERE id = uploaded_by) as uploaded_by
            FROM model_versions
            WHERE path = ? OR path = ?
            ORDER BY uploaded_at DESC
            LIMIT 1
        ''', (
            active_model,
            str(Path(active_model).relative_to(REPO_ROOT)) if REPO_ROOT in Path(active_model).parents else None
        )
        
        model_info = cursor.fetchone()
        
        if model_info:
            return {
                "path": active_model,
                "version": model_info[0],
                "description": model_info[1],
                "upload_date": model_info[2],
                "uploaded_by": model_info[3]
            }
        
        return {
            "path": active_model,
            "version": "Default Model",
            "description": "Initial model provided with system",
            "upload_date": "N/A",
            "uploaded_by": "System"
        }
        
    except Exception as e:
        st.error(f"Error getting model info: {str(e)}")
        return {
            "path": active_model,
            "version": "Unknown",
            "description": "Could not retrieve model details",
            "upload_date": "N/A",
            "uploaded_by": "Unknown"
        }
    finally:
        if conn:
            conn.close()

def enhance_image_quality(img_pil):
    """Enhance image clarity, brightness, contrast, and sharpness."""
    try:
        # Convert to OpenCV BGR format
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Resize for faster enhancement
        img_cv = cv2.resize(img_cv, (224, 224))

        # Apply detail enhancement
        img_cv = cv2.detailEnhance(img_cv, sigma_s=10, sigma_r=0.15)

        # Convert back to PIL (RGB)
        enhanced_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        # Enhance brightness
        enhancer = ImageEnhance.Brightness(enhanced_pil)
        enhanced_pil = enhancer.enhance(1.2)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(enhanced_pil)
        enhanced_pil = enhancer.enhance(1.3)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced_pil)
        enhanced_pil = enhancer.enhance(2.0)

        return enhanced_pil
    except Exception as e:
        st.warning(f"⚠️ Image enhancement failed: {str(e)}")
        return img_pil  # fallback to original

def preprocess_image(img_pil):
    """Resize, enhance, and normalize image for model input (224x224)."""
    try:
        # Step 1: Enhance image quality
        enhanced_img = enhance_image_quality(img_pil)

        # Step 2: Resize and convert to array
        img = enhanced_img.resize((224, 224))
        img_array = image.img_to_array(img)

        # Step 3: Normalize and add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        return img_array
    except Exception as e:
        st.error(f"❌ Image preprocessing failed: {str(e)}")
        return None

def predict_image(img_path, model):
    """Make prediction on uploaded image"""
    try:
        # Load original image
        img = Image.open(img_path)
        
        # Preprocess the image (includes enhancement)
        img_array = preprocess_image(img)
        
        if img_array is None:
            return None, None

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None
    
def save_detection(patient_id, result, confidence, attended_by, notes=""):
    """Save detection results to database with robust error handling"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Start transaction
        cursor.execute("BEGIN TRANSACTION")
        
        # Insert detection record
        cursor.execute('''
            INSERT INTO detections 
            (patient_id, result, confidence, attended_by, notes, detection_date)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (patient_id, result, confidence, attended_by, notes))
        
        detection_id = cursor.lastrowid
        
        # Commit transaction
        conn.commit()
        return detection_id
        
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        st.error(f"Database error saving detection: {str(e)}")
        return None
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Error saving detection: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def get_detections():
    """Get all detections with patient info"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        query = '''
            SELECT 
                d.id,
                d.patient_id,
                p.full_name,
                p.gender,
                p.age,
                d.result,
                d.confidence,
                d.attended_by,
                d.notes,
                d.detection_date
            FROM detections d
            JOIN patients p ON d.patient_id = p.id
            ORDER BY d.detection_date DESC, d.id DESC
        '''
        df = pd.read_sql_query(query, conn)
        return df

    except sqlite3.Error as e:
        st.error(f"Database error getting detections: {str(e)}")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error getting detections: {str(e)}")
        return pd.DataFrame()

    finally:
        if conn:
            conn.close()

def get_detection_by_id(detection_id):
    """Get a specific detection by ID"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT d.*, p.full_name 
            FROM detections d
            JOIN patients p ON d.patient_id = p.id
            WHERE d.id = ?
        ''', (detection_id,))
        detection = cursor.fetchone()
        return detection

    except Exception as e:
        st.error(f"Error getting detection by ID: {str(e)}")
        return None

    finally:
        if conn:
            conn.close()

def update_detection(detection_id, new_result, new_confidence, new_notes):
    """Hard update of detection record with verification"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Start transaction
        cursor.execute("BEGIN TRANSACTION")
        
        # Update detection
        cursor.execute('''
            UPDATE detections 
            SET result = ?, 
                confidence = ?, 
                notes = ?,
                detection_date = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (new_result, new_confidence, new_notes, detection_id))
        
        # Verify exactly one row was updated
        if cursor.rowcount != 1:
            raise ValueError(f"Expected to update 1 row, updated {cursor.rowcount}")
        
        # Commit transaction
        conn.commit()
        return True
        
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        st.error(f"Database error updating detection: {str(e)}")
        return False
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Error updating detection: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def delete_detection(detection_id):
    """Hard delete detection record with verification"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Start transaction
        cursor.execute("BEGIN TRANSACTION")
        
        # Delete detection
        cursor.execute('DELETE FROM detections WHERE id = ?', (detection_id,))
        
        # Verify exactly one row was deleted
        if cursor.rowcount != 1:
            raise ValueError(f"Expected to delete 1 row, deleted {cursor.rowcount}")
        
        # Commit transaction
        conn.commit()
        return True
        
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        st.error(f"Database error deleting detection: {str(e)}")
        return False
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Error deleting detection: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

# -------------------------------
# 📅 Appointment Functions 
# -------------------------------
def add_appointment(patient_id, patient_name, gender, age, village, traditional_authority, 
                    district, marital_status, appointment_date, appointment_time, booked_by, 
                    doctor_email, notes="", status="Pending"):
    """Add new appointment"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO appointments (
                patient_id, patient_name, gender, age, village, traditional_authority,
                district, marital_status, appointment_date, appointment_time, booked_by,
                doctor_email, notes, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, patient_name, gender, age, village, traditional_authority,
            district, marital_status, appointment_date, appointment_time, booked_by,
            doctor_email, notes, status
        ))
        appointment_id = cursor.lastrowid
        conn.commit()
        return appointment_id
    except Exception as e:
        st.error(f"Error adding appointment: {str(e)}")
        return None
    finally:
        conn.close()

def get_appointments():
    """Get all appointments"""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query('''
            SELECT * FROM appointments 
            ORDER BY appointment_date DESC, appointment_time DESC
        ''', conn)
        return df
    except Exception as e:
        st.error(f"Error getting appointments: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_doctors():
    """Get all approved doctors"""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM users WHERE role = 'doctor' AND status = 'approved'", conn)
        return df
    except Exception as e:
        st.error(f"Error getting doctors: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_doctor_by_email(email):
    """Get doctor details by email"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM users WHERE email = ? AND role = "doctor"', (email.lower(),))
        doctor = cursor.fetchone()
        return doctor
    except Exception as e:
        st.error(f"Error getting doctor by email: {str(e)}")
        return None
    finally:
        conn.close()

# -------------------------------
# 📧 Messaging Functions
# -------------------------------
def send_message(sender_email, receiver_email, subject, message, attachments=None):
    """Send message between users with attachments"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Insert message
        cursor.execute('''
            INSERT INTO messages (sender_email, receiver_email, subject, message)
            VALUES (?, ?, ?, ?)
        ''', (sender_email.lower(), receiver_email.lower(), subject, message))
        message_id = cursor.lastrowid
        
        # Handle attachments
        if attachments:
            for attachment in attachments:
                if attachment is not None:
                    os.makedirs("message_attachments", exist_ok=True)
                    file_ext = os.path.splitext(attachment.name)[1]
                    unique_filename = f"{uuid.uuid4()}{file_ext}"
                    file_path = os.path.join("message_attachments", unique_filename)
                    
                    with open(file_path, "wb") as f:
                        f.write(attachment.getbuffer())
                    
                    cursor.execute('''
                        INSERT INTO message_attachments (
                            message_id, file_name, file_path, file_size, file_type
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (
                        message_id,
                        attachment.name,
                        file_path,
                        len(attachment.getvalue()),
                        attachment.type
                    ))
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        st.error(f"Failed to send message: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_messages(email, include_deleted=False):
    """Get messages for a user"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        query = '''
            SELECT 
                m.id,
                m.sender_email,
                m.receiver_email,
                m.subject,
                m.message,
                datetime(m.sent_at, 'localtime') as sent_at,
                m.is_deleted,
                m.deleted_at,
                m.deleted_by,
                (SELECT COUNT(*) FROM message_attachments WHERE message_id = m.id) as attachment_count
            FROM messages m
            WHERE (m.sender_email = ? OR m.receiver_email = ?)
        '''
        if not include_deleted:
            query += " AND m.is_deleted = 0"
        query += " ORDER BY m.sent_at DESC"
        
        return pd.read_sql_query(query, conn, params=(email.lower(), email.lower()))
    except Exception as e:
        st.error(f"Failed to load messages: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def delete_message(message_id, deleted_by_email):
    """Soft delete a message"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages 
            SET is_deleted = 1, 
                deleted_at = CURRENT_TIMESTAMP,
                deleted_by = ?
            WHERE id = ?
        ''', (deleted_by_email, message_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Failed to delete message: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_message_attachments(message_id):
    """Get attachments for a message"""
    conn = None
    try:
        return pd.read_sql_query('''
            SELECT 
                id,
                file_name,
                file_path,
                file_size,
                file_type,
                datetime(uploaded_at, 'localtime') as uploaded_at
            FROM message_attachments
            WHERE message_id = ?
        ''', conn, params=(message_id,))
    except Exception as e:
        st.error(f"Failed to load attachments: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def empty_trash(user_email):
    """Permanently delete messages in trash"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Get message IDs to delete
        cursor.execute('''
            SELECT id FROM messages 
            WHERE is_deleted = 1 AND deleted_by = ?
        ''', (user_email,))
        message_ids = [row[0] for row in cursor.fetchall()]
        
        if message_ids:
            # Delete attachments first
            cursor.executemany('''
                DELETE FROM message_attachments WHERE message_id = ?
            ''', [(msg_id,) for msg_id in message_ids])
            
            # Then delete messages
            cursor.execute('''
                DELETE FROM messages 
                WHERE is_deleted = 1 AND deleted_by = ?
            ''', (user_email,))
            
        conn.commit()
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Failed to empty trash: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def restore_message(message_id):
    """Restore a deleted message"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages 
            SET is_deleted = 0,
                deleted_at = NULL,
                deleted_by = NULL
            WHERE id = ?
        ''', (message_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Failed to restore message: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

# -------------------------------
# 🔐 Authentication UI
# -------------------------------
def show_auth_page():
    """Show login/register interface with working switch"""
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        background-image: url('https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=500&q=80');
        background-size: cover;
        background-position: center;
        background-blend-mode: overlay;
        background-color: rgba(255,255,255,0.9);
    }
    .auth-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        color: #2d3748;
    }
    .auth-subtitle {
        text-align: center;
        font-size: 1rem;
        color: #718096;
        margin-bottom: 2rem;
    }
    .auth-switch {
        text-align: center;
        margin-top: 1.5rem;
        font-size: 0.9rem;
        color: #718096;
    }
    .error-message {
        color: #e53e3e;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize auth mode
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = "login"

    # Login Form
    if st.session_state.auth_mode == "login":
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown('<div class="auth-title">Welcome Back</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-subtitle">Sign in to your account</div>', unsafe_allow_html=True)
            
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", placeholder="your@email.com").strip()
                password = st.text_input("Password", type="password")
                
                login_button = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                
                if login_button:
                    if not email or not password:
                        st.error("Please enter both email and password", icon="⚠️")
                    else:
                        if login_user(email, password):
                            st.success("Login successful! Redirecting...")
                            time.sleep(1)
                            st.experimental_rerun()
                        else:
                            st.error("Login failed - please try again", icon="🚨")

            # Registration switch
            st.markdown('<div class="auth-switch">Don\'t have an account?</div>', unsafe_allow_html=True)
            if st.button("Register here", key="to_register", use_container_width=True):
                st.session_state.auth_mode = "register"
                st.experimental_rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    # Registration Form
    else:
        with st.container():
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown('<div class="auth-title">Create Account</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-subtitle">Get started in seconds</div>', unsafe_allow_html=True)
            
            with st.form("register_form", clear_on_submit=False):
                full_name = st.text_input("Full Name", placeholder="John Doe").strip()
                email = st.text_input("Email", placeholder="your@email.com").strip()
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                role = st.selectbox("Role", ["assistant", "doctor"])
                
                register_button = st.form_submit_button("Register", type="primary", use_container_width=True)
                
                if register_button:
                    if not all([full_name, email, password, confirm_password]):
                        st.error("Please fill in all fields", icon="⚠️")
                    elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                        st.error("Please enter a valid email address", icon="✉️")
                    elif len(password) < 8:
                        st.error("Password must be at least 8 characters", icon="🔒")
                    elif password != confirm_password:
                        st.error("Passwords don't match", icon="🔁")
                    elif get_user_by_email(email):
                        st.error("Email already registered", icon="⛔")
                    else:
                        if create_user(full_name, email, password, role):
                            st.success("Registration submitted for admin approval", icon="✅")
                            time.sleep(1)
                            st.session_state.auth_mode = "login"
                            st.experimental_rerun()
                        else:
                            st.error("Registration failed - please try again", icon="🚨")

            # Login switch
            st.markdown('<div class="auth-switch">Already have an account?</div>', unsafe_allow_html=True)
            if st.button("Sign in here", key="to_login", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.experimental_rerun()

            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# 🏠 Home Page
# -------------------------------
def show_home_page():
    """Show modern landing page with interactive elements"""
    st.markdown("""
    <style>
    .hero {
        background: linear-gradient(135deg, #48bb78, #38a169);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        transition: all 0.5s ease;
        background-image: url('https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=1200&q=80');
        background-size: cover;
        background-position: center;
        background-blend-mode: overlay;
    }
    .hero:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.15);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-family: 'Playfair Display', serif;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1.5rem;
    }
    .feature-card {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        border-color: #48bb78;
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #38a169;
        transition: all 0.3s ease;
    }
    .feature-card:hover .feature-icon {
        transform: scale(1.1);
        color: #2c7a4d;
    }
    .stats-container {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .quick-action {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .quick-action:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(72,187,120,0.15);
        background: linear-gradient(135deg, #f0fff4, #ffffff);
    }
    .action-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #38a169;
        transition: all 0.3s ease;
    }
    .quick-action:hover .action-icon {
        transform: scale(1.2);
    }
    .model-info-card {
        background: rgba(255,255,255,0.9);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 4px solid #38a169;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">Munthandiz Cataract Detection</div>
        <div class="hero-subtitle">Advanced AI-powered eye care diagnostics</div>
        <div style="font-size: 1rem; opacity: 0.8;">Welcome back, {st.session_state.user_name} 👋</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Grid
    st.markdown('<div class="section-title">Key Features</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">👁️</div>
            <h3>AI Detection</h3>
            <p>State-of-the-art cataract classification with 95%+ accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Real-time Analytics</h3>
            <p>Comprehensive dashboards with patient insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤝</div>
            <h3>Collaborative Care</h3>
            <p>Seamless communication between medical teams</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats
    conn = sqlite3.connect(DB_NAME)
    try:
        patients_count = pd.read_sql_query("SELECT COUNT(*) FROM patients", conn).iloc[0,0]
        detections_count = pd.read_sql_query("SELECT COUNT(*) FROM detections", conn).iloc[0,0]
        positive_cases = pd.read_sql_query("SELECT COUNT(*) FROM detections WHERE result != 'normal'", conn).iloc[0,0]
        avg_confidence = pd.read_sql_query("SELECT AVG(confidence) FROM detections", conn).iloc[0,0] or 0
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")
        patients_count = 0
        detections_count = 0
        positive_cases = 0
        avg_confidence = 0
    finally:
        conn.close()
    
    st.markdown("""
    <div class="stats-container">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="font-size: 2rem; font-weight: 700; color: #38a169;">{patients_count}</div>
                <div style="font-size: 0.9rem; color: #4a5568;">Patients Registered</div>
            </div>
            <div>
                <div style="font-size: 2rem; font-weight: 700; color: #38a169;">{detections_count}</div>
                <div style="font-size: 0.9rem; color: #4a5568;">Diagnoses Performed</div>
            </div>
            <div>
                <div style="font-size: 2rem; font-weight: 700; color: #38a169;">{positive_cases}</div>
                <div style="font-size: 0.9rem; color: #4a5568;">Positive Cases</div>
            </div>
        </div>
    </div>
    """.format(patients_count=patients_count, detections_count=detections_count, positive_cases=positive_cases), 
    unsafe_allow_html=True)
    
    # Display active model information
    model_info = get_active_model_info()
    st.markdown(f"""
    <div class="model-info-card">
        <h3>Active Model Information</h3>
        <p><strong>Version:</strong> {model_info['version']}</p>
        <p><strong>Description:</strong> {model_info['description']}</p>
        <p><strong>Uploaded by:</strong> {model_info['uploaded_by']}</p>
        <p><strong>Upload date:</strong> {model_info['upload_date']}</p>
        <p><strong>Path:</strong> <code>{model_info['path']}</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown('<div class="section-title">Quick Actions</div>', unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔍 New Detection", use_container_width=True):
            st.session_state.nav = "Detection"
            st.experimental_rerun()
    
    with action_col2:
        if st.button("📅 Schedule", use_container_width=True):
            st.session_state.nav = "Appointments"
            st.experimental_rerun()
    
    with action_col3:
        if st.button("✉️ Messages", use_container_width=True):
            st.session_state.nav = "Messages"
            st.experimental_rerun()
    
    # Recent Activity
    st.markdown('<div class="section-title">Recent Activity</div>', unsafe_allow_html=True)
    
    conn = sqlite3.connect(DB_NAME)
    try:
        recent_detections = pd.read_sql_query('''
            SELECT p.full_name, d.result, d.confidence, d.detection_date 
            FROM detections d
            JOIN patients p ON d.patient_id = p.id
            ORDER BY d.detection_date DESC
            LIMIT 5
        ''', conn)
        
        if not recent_detections.empty:
            st.dataframe(recent_detections, use_container_width=True)
        else:
            st.info("No recent activity found")
    except Exception as e:
        st.error(f"Error loading recent activity: {str(e)}")
    finally:
        conn.close()

# -------------------------------
# 👁️ Detection Page
# -------------------------------
def show_detection_page():
    """Show the cataract detection interface with edit/delete functionality"""
    st.markdown('<h1 class="section-title">👁️ Cataract Detection</h1>', unsafe_allow_html=True)

    model_info = get_active_model_info()
    with st.expander("ℹ️ Current Model Information", expanded=True):
        st.markdown(f"""
        - **Version:** {model_info['version']}
        - **Description:** {model_info['description']}
        - **Uploaded by:** {model_info['uploaded_by']}
        - **Path:** `{model_info['path']}`
        """)

    tab1, tab2 = st.tabs(["New Detection", "Manage Detections"])

    # ----------- TAB 1: New Detection -----------
    with tab1:
        use_camera = st.checkbox("🎥 Use Camera", value=False)

        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
        subtab1, subtab2 = st.tabs(["Existing Patient", "New Patient"])

        patient_id = None

        with subtab1:
            patients = get_patients()
            if patients.empty:
                st.info("No patients found. Please register a new patient.")
            else:
                patient_options = patients['full_name'] + " | " + patients['village'] + " | " + patients['district']
                selected_patient = st.selectbox("Select Patient", options=patient_options)
                if selected_patient:
                    patient_id = patients.iloc[patient_options.tolist().index(selected_patient)]['id']

        with subtab2:
            st.markdown("### Register New Patient")
            with st.form("patient_form"):
                full_name = st.text_input("Full Name")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                age = st.number_input("Age", min_value=0, max_value=120)
                village = st.text_input("Village")
                traditional_authority = st.text_input("Traditional Authority")
                district = st.text_input("District")
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

                if st.form_submit_button("Register Patient"):
                    patient_id = add_patient(full_name, gender, age, village, traditional_authority, district, marital_status)
                    if patient_id:
                        st.success(f"Patient {full_name} registered successfully!")
                        st.experimental_rerun()

        if patient_id:
            st.markdown('<div class="section-title">Capture Eye Image</div>', unsafe_allow_html=True)

            img = st.camera_input("Take an eye photo") if use_camera else st.file_uploader("Upload an eye image...", type=["jpg", "jpeg", "png"])
            if img:
                st.image(img, caption="Eye Image", use_column_width=True)

                if st.button("Analyze Image", key="analyze_btn"):
                    temp_file = "temp_eye_image.jpg"
                    try:
                        with open(temp_file, "wb") as f:
                            f.write(img.getbuffer() if use_camera else img.getvalue())

                        model = load_detection_model()
                        if model:
                            with st.spinner("Analyzing image..."):
                                predicted_class, confidence = predict_image(temp_file, model)
                                if predicted_class:
                                    st.session_state["predicted_class"] = predicted_class
                                    st.session_state["confidence"] = confidence
                                    st.session_state["image_ready"] = True
                    finally:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

            if st.session_state.get("image_ready", False):
                st.markdown(f'''
                <div class="feature-card">
                    <h3>Analysis Results</h3>
                    <p><strong>Prediction:</strong> {st.session_state["predicted_class"]}</p>
                    <p><strong>Confidence:</strong> {st.session_state["confidence"]:.2f}%</p>
                </div>
                ''', unsafe_allow_html=True)

                notes = st.text_area("Additional Notes")

                if st.button("Save Detection Results", key="save_btn"):
                    detection_id = save_detection(
                        patient_id=patient_id,
                        result=st.session_state["predicted_class"],
                        confidence=st.session_state["confidence"],
                        attended_by=st.session_state.user_name,
                        notes=notes
                    )
                    if detection_id:
                        st.success("Detection results saved successfully!")
                        st.session_state["image_ready"] = False
                        st.experimental_rerun()
                    else:
                        st.error("Failed to save detection results")
        else:
            st.warning("Please select or register a patient first")

    # ----------- TAB 2: Manage Detections -----------
    with tab2:
        st.markdown('<div class="section-title">Manage Detection Results</div>', unsafe_allow_html=True)
        detections = get_detections()

        if detections.empty:
            st.info("No detection records found")
            return

        # Ensure ID is included
        if 'id' not in detections.columns:
            st.error("Missing 'id' in detections data")
            return

        gb = GridOptionsBuilder.from_dataframe(detections)
        gb.configure_default_column(editable=False, filterable=True, sortable=True)
        gb.configure_column("result", editable=True)
        gb.configure_column("confidence", editable=True)
        gb.configure_column("notes", editable=True)
        gb.configure_column("detection_date", header_name="Date")
        gb.configure_column("id", header_name="ID", editable=False)
        gb.configure_selection("multiple", use_checkbox=True)
        grid_options = gb.build()

        grid_response = AgGrid(
            detections,
            gridOptions=grid_options,
            height=500,
            width='100%',
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            theme='streamlit'
        )

        selected_rows = grid_response.get("selected_rows", [])

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("💾 Save All Changes", key="save_all_btn"):
                updated = grid_response['data']
                success_count = 0
                for _, row in updated.iterrows():
                    if update_detection(row['id'], row['result'], row['confidence'], row['notes']):
                        success_count += 1
                st.success(f"Updated {success_count} records")
                st.experimental_rerun()

        with col2:
            if st.button("🔄 Refresh Data", key="refresh_btn"):
                st.experimental_rerun()

        with col3:
            if selected_rows and st.button("🗑️ Delete Selected", key="delete_btn"):
                delete_count = 0
                for row in selected_rows:
                    if 'id' in row and delete_detection(row['id']):
                        delete_count += 1
                st.success(f"Deleted {delete_count} records")
                st.experimental_rerun()

        # Detail View
        if selected_rows and len(selected_rows) == 1:
            r = selected_rows[0]
            st.markdown("---")
            st.markdown("### Detailed View")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Patient:** {r['full_name']}")
                st.markdown(f"**Gender:** {r['gender']}")
                st.markdown(f"**Age:** {r['age']}")
            with col2:
                st.markdown(f"**Result:** {r['result']}")
                st.markdown(f"**Confidence:** {r['confidence']:.2f}%")
                st.markdown(f"**Date:** {r['detection_date']}")

            st.markdown("**Notes:**")
            st.write(r['notes'])
            
# -------------------------------
# 📅 Appointments Page 
# -------------------------------
def show_appointments_page():
    """Show appointments management interface with direct doctor selection"""
    st.markdown('<h1 class="section-title">📅 Appointments</h1>', unsafe_allow_html=True)
    
    # Tab interface for different functions
    tab1, tab2 = st.tabs(["New Appointment", "Manage Appointments"])
    
    with tab1:
        st.markdown('<div class="section-title">Book New Appointment</div>', unsafe_allow_html=True)
        
        # Get patients for selection
        patients = get_patients()
        if patients.empty:
            st.warning("No patients available. Please register patients first.")
        else:
            patient_options = patients['full_name'] + " | " + patients['village']
            selected_patient = st.selectbox("Select Patient", options=patient_options)
            patient_id = patients.iloc[patient_options.tolist().index(selected_patient)]['id']
            patient_data = patients[patients['id'] == patient_id].iloc[0]
            
            # Get available doctors with direct selection
            doctors = get_doctors()
            if doctors.empty:
                st.warning("No doctors available. Please have an admin register doctors.")
            else:
                # Enhanced doctor selection with search and details
                st.markdown("### Select Doctor")
                doctor_col1, doctor_col2 = st.columns([2, 1])
                
                with doctor_col1:
                    doctor_options = doctors['full_name'] + " (" + doctors['email'] + ")"
                    selected_doctor = st.selectbox(
                        "Choose Doctor",
                        options=doctor_options,
                        index=0,
                        key="doctor_select"
                    )
                    doctor_email = doctors.iloc[doctor_options.tolist().index(selected_doctor)]['email']
                    doctor_data = doctors[doctors['email'] == doctor_email].iloc[0]
                
                with doctor_col2:
                    st.markdown("**Selected Doctor**")
                    st.markdown(f"👨‍⚕️ {doctor_data['full_name']}")
                    st.markdown(f"📧 {doctor_data['email']}")
                
                # Appointment details
                st.markdown("### Appointment Details")
                col1, col2 = st.columns(2)
                with col1:
                    appointment_date = st.date_input("Date", min_value=date.today())
                with col2:
                    appointment_time = st.time_input("Time")
                
                notes = st.text_area("Notes", placeholder="Additional information about the appointment")
                
                if st.button("Book Appointment", type="primary"):
                    appointment_id = add_appointment(
                        patient_id=patient_id,
                        patient_name=patient_data['full_name'],
                        gender=patient_data['gender'],
                        age=patient_data['age'],
                        village=patient_data['village'],
                        traditional_authority=patient_data['traditional_authority'],
                        district=patient_data['district'],
                        marital_status=patient_data['marital_status'],
                        appointment_date=appointment_date,
                        appointment_time=appointment_time.strftime("%H:%M"),
                        booked_by=st.session_state.user_name,
                        doctor_email=doctor_email,
                        notes=notes
                    )
                    
                    if appointment_id:
                        st.success(f"Appointment booked successfully (ID: {appointment_id})")
                        st.balloons()
                        time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.error("Failed to book appointment")
    
    with tab2:
        st.markdown('<div class="section-title">Manage Appointments</div>', unsafe_allow_html=True)
        
        # Get all appointments with filtering options
        appointments = get_appointments()
        if not appointments.empty:
            # Add filters
            st.markdown("### Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_filter = st.selectbox(
                    "Status",
                    options=["All", "Pending", "Completed", "Cancelled"],
                    index=0
                )
            
            with col2:
                date_filter = st.selectbox(
                    "Date Range",
                    options=["All", "Today", "This Week", "This Month", "Upcoming"],
                    index=0
                )
            
            with col3:
                doctor_filter = st.selectbox(
                    "Doctor",
                    options=["All"] + sorted(appointments['doctor_email'].unique().tolist()),
                    index=0
                )
            
            # Apply filters
            filtered_appointments = appointments.copy()
            if status_filter != "All":
                filtered_appointments = filtered_appointments[filtered_appointments['status'] == status_filter]
            
            if date_filter != "All":
                today = date.today()
                if date_filter == "Today":
                    filtered_appointments = filtered_appointments[filtered_appointments['appointment_date'] == today]
                elif date_filter == "This Week":
                    start_date = today - timedelta(days=today.weekday())
                    end_date = start_date + timedelta(days=6)
                    filtered_appointments = filtered_appointments[
                        (filtered_appointments['appointment_date'] >= start_date) &
                        (filtered_appointments['appointment_date'] <= end_date)
                    ]
                elif date_filter == "This Month":
                    start_date = date(today.year, today.month, 1)
                    end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)
                    filtered_appointments = filtered_appointments[
                        (filtered_appointments['appointment_date'] >= start_date) &
                        (filtered_appointments['appointment_date'] <= end_date)
                    ]
                elif date_filter == "Upcoming":
                    filtered_appointments = filtered_appointments[
                        filtered_appointments['appointment_date'] >= today
                    ]
            
            if doctor_filter != "All":
                filtered_appointments = filtered_appointments[filtered_appointments['doctor_email'] == doctor_filter]
            
            # Display filtered appointments
            st.dataframe(filtered_appointments, use_container_width=True)
            
            # Appointment actions
            st.markdown("### Appointment Actions")
            selected_appointment = st.selectbox(
                "Select an appointment to manage",
                options=filtered_appointments['patient_name'] + " - " + 
                        filtered_appointments['appointment_date'].astype(str) + " " + 
                        filtered_appointments['appointment_time'],
                key="appointment_select"
            )
            
            if selected_appointment:
                appointment_id = filtered_appointments.iloc[
                    (filtered_appointments['patient_name'] + " - " + 
                     filtered_appointments['appointment_date'].astype(str) + " " + 
                     filtered_appointments['appointment_time']).tolist().index(selected_appointment)
                ]['id']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    new_status = st.selectbox(
                        "Update Status",
                        options=["Pending", "Completed", "Cancelled"],
                        index=["Pending", "Completed", "Cancelled"].index(
                            filtered_appointments[filtered_appointments['id'] == appointment_id]['status'].iloc[0]
                        )
                    )
                    
                    if st.button("Update Status"):
                        conn = sqlite3.connect(DB_NAME)
                        try:
                            conn.execute(
                                "UPDATE appointments SET status = ? WHERE id = ?",
                                (new_status, appointment_id)
                            )
                            conn.commit()
                            st.success("Status updated successfully!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error updating status: {str(e)}")
                        finally:
                            conn.close()
                
                with col2:
                    if st.button("Reschedule Appointment"):
                        st.session_state.reschedule_appt = appointment_id
                        st.experimental_rerun()
                
                with col3:
                    if st.button("Cancel Appointment", type="secondary"):
                        conn = sqlite3.connect(DB_NAME)
                        try:
                            conn.execute(
                                "UPDATE appointments SET status = 'Cancelled' WHERE id = ?",
                                (appointment_id,)
                            )
                            conn.commit()
                            st.success("Appointment cancelled successfully!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error cancelling appointment: {str(e)}")
                        finally:
                            conn.close()
            
            # Reschedule interface
            if 'reschedule_appt' in st.session_state:
                appointment_to_reschedule = appointments[appointments['id'] == st.session_state.reschedule_appt].iloc[0]
                
                st.markdown("---")
                st.markdown("### Reschedule Appointment")
                st.markdown(f"Patient: {appointment_to_reschedule['patient_name']}")
                st.markdown(f"Current Date/Time: {appointment_to_reschedule['appointment_date']} {appointment_to_reschedule['appointment_time']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    new_date = st.date_input("New Date", min_value=date.today())
                with col2:
                    new_time = st.time_input("New Time")
                
                if st.button("Confirm Reschedule"):
                    conn = sqlite3.connect(DB_NAME)
                    try:
                        conn.execute(
                            "UPDATE appointments SET appointment_date = ?, appointment_time = ?, status = 'Pending' WHERE id = ?",
                            (new_date, new_time.strftime("%H:%M"), st.session_state.reschedule_appt)
                        )
                        conn.commit()
                        st.success("Appointment rescheduled successfully!")
                        del st.session_state.reschedule_appt
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error rescheduling appointment: {str(e)}")
                    finally:
                        conn.close()
                
                if st.button("Cancel Reschedule", type="secondary"):
                    del st.session_state.reschedule_appt
                    st.experimental_rerun()
        else:
            st.info("No appointments found")
            
# -------------------------------
# 📊 Analytics Page 
# -------------------------------
def show_analytics_page():
    """Show analytics dashboard with interactive components"""
    st.markdown('<h1 class="section-title">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Get data with patient info
    conn = sqlite3.connect(DB_NAME)
    try:
        detections = pd.read_sql_query('''
            SELECT d.*, p.full_name, p.gender, p.age, p.district, p.village
            FROM detections d
            JOIN patients p ON d.patient_id = p.id
        ''', conn)
        patients = pd.read_sql_query("SELECT * FROM patients", conn)
        appointments = pd.read_sql_query("SELECT * FROM appointments", conn)
        
        # Convert detection_date to datetime if it exists
        if not detections.empty and 'detection_date' in detections.columns:
            detections['detection_date'] = pd.to_datetime(detections['detection_date']).dt.date
        
        # Create tab interface
        tab1, tab2, tab3 = st.tabs(["📈 Overview", "👁️ Detection Analytics", "📅 Appointment Analytics"])
        
        with tab1:
            _display_overview_analytics(patients, detections, appointments)
        
        with tab2:
            _display_detection_analytics(detections)
        
        with tab3:
            _display_appointment_analytics(appointments)
            
    except Exception as e:
        st.error(f"Error loading analytics data: {str(e)}")
    finally:
        conn.close()

def _display_overview_analytics(patients: pd.DataFrame, detections: pd.DataFrame, appointments: pd.DataFrame):
    """Display overview analytics with KPIs and trends"""
    # Calculate KPIs
    total_patients = len(patients)
    total_detections = len(detections)
    positive_cases = len(detections[detections['result'] != 'normal']) if not detections.empty else 0
    avg_confidence = detections['confidence'].mean() if not detections.empty else 0
    upcoming_appointments = len(appointments[appointments['status'] == 'Pending']) if not appointments.empty else 0
    completed_appointments = len(appointments[appointments['status'] == 'Completed']) if not appointments.empty else 0
    
    # Display KPIs in grid
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Total Patients</div>
            <div class="metric-value">{total_patients}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Avg. Confidence</div>
            <div class="metric-value">{avg_confidence:.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Total Detections</div>
            <div class="metric-value">{total_detections}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Positive Cases</div>
            <div class="metric-value">{positive_cases}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Upcoming Appointments</div>
            <div class="metric-value">{upcoming_appointments}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">Completed Appointments</div>
            <div class="metric-value">{completed_appointments}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Trends over time
    st.markdown('<div class="section-title">Trends Over Time</div>', unsafe_allow_html=True)
    
    if not detections.empty and 'detection_date' in detections.columns:
        # Monthly detection trend
        detections['month'] = pd.to_datetime(detections['detection_date']).dt.to_period('M')
        monthly_counts = detections.groupby('month').size().reset_index()
        monthly_counts.columns = ['Month', 'Count']
        monthly_counts['Month'] = monthly_counts['Month'].astype(str)
        
        fig = px.line(
            monthly_counts, 
            x='Month', 
            y='Count',
            title="Monthly Detection Trend",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if not appointments.empty and 'appointment_date' in appointments.columns:
        # Monthly appointment trend
        appointments['month'] = pd.to_datetime(appointments['appointment_date']).dt.to_period('M')
        appt_monthly = appointments.groupby(['month', 'status']).size().reset_index()
        appt_monthly.columns = ['Month', 'Status', 'Count']
        appt_monthly['Month'] = appt_monthly['Month'].astype(str)
        
        fig = px.bar(
            appt_monthly,
            x='Month',
            y='Count',
            color='Status',
            title="Monthly Appointment Status",
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def _display_detection_analytics(detections: pd.DataFrame):
    """Display detection-specific analytics"""
    st.markdown('<div class="section-title">Detection Analytics</div>', unsafe_allow_html=True)
    
    if detections.empty:
        st.info("No detection data available")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        result_filter = st.multiselect(
            "Filter Results",
            options=detections['result'].unique(),
            default=detections['result'].unique()
        )
    with col2:
        age_range = st.slider(
            "Age Range",
            min_value=int(detections['age'].min()) if not detections.empty else 0,
            max_value=int(detections['age'].max()) if not detections.empty else 100,
            value=(
                int(detections['age'].min()) if not detections.empty else 0,
                int(detections['age'].max()) if not detections.empty else 100
            )
        )
    with col3:
        district_filter = st.multiselect(
            "Filter District",
            options=detections['district'].unique(),
            default=detections['district'].unique()
        )
    
    # Apply filters
    filtered = detections[
        (detections['result'].isin(result_filter)) &
        (detections['age'] >= age_range[0]) &
        (detections['age'] <= age_range[1]) &
        (detections['district'].isin(district_filter))
    ]
    
    if filtered.empty:
        st.warning("No data matching filters")
        return
    
    # Visualizations in tabs
    tab1, tab2, tab3 = st.tabs(["Results Breakdown", "Confidence Analysis", "Demographics"])
    
    with tab1:
        # Results breakdown
        result_counts = filtered['result'].value_counts().reset_index()
        result_counts.columns = ['Result', 'Count']
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                result_counts, 
                values='Count', 
                names='Result',
                title="Result Distribution",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                result_counts,
                x='Result',
                y='Count',
                color='Result',
                title="Results by Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Confidence analysis
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                filtered,
                x='confidence',
                nbins=20,
                title="Confidence Distribution",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                filtered,
                x='result',
                y='confidence',
                color='result',
                title="Confidence by Result",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Demographics
        col1, col2 = st.columns(2)
        with col1:
            gender_counts = filtered['gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            
            fig = px.pie(
                gender_counts,
                values='Count',
                names='Gender',
                title="Gender Distribution",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            age_groups = pd.cut(
                filtered['age'],
                bins=[0, 18, 30, 45, 60, 100],
                labels=['0-18', '19-30', '31-45', '46-60', '60+']
            )
            age_counts = age_groups.value_counts().reset_index()
            age_counts.columns = ['Age Group', 'Count']
            
            fig = px.bar(
                age_counts,
                x='Age Group',
                y='Count',
                title="Age Group Distribution",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed data
    st.markdown('<div class="section-title">Detailed Data</div>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True)

def _display_appointment_analytics(appointments: pd.DataFrame):
    """Display appointment-specific analytics"""
    st.markdown('<div class="section-title">Appointment Analytics</div>', unsafe_allow_html=True)
    
    if appointments.empty:
        st.info("No appointment data available")
        return
    
    # Status breakdown
    status_counts = appointments['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            status_counts,
            values='Count',
            names='Status',
            title="Appointment Status",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            status_counts,
            x='Status',
            y='Count',
            color='Status',
            title="Appointments by Status",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Doctor workload
    if 'doctor_email' in appointments.columns:
        doctor_counts = appointments['doctor_email'].value_counts().reset_index()
        doctor_counts.columns = ['Doctor', 'Count']
        
        st.markdown('<div class="section-title">Doctor Workload</div>', unsafe_allow_html=True)
        fig = px.bar(
            doctor_counts,
            x='Doctor',
            y='Count',
            title="Appointments per Doctor",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed data
    st.markdown('<div class="section-title">Detailed Data</div>', unsafe_allow_html=True)
    st.dataframe(appointments, use_container_width=True)

# -------------------------------
# 📧 Messaging Page
# -------------------------------
def show_messages_page():
    """Show enhanced messaging interface with attachments and deletion"""
    st.markdown('<h1 class="section-title">📧 Messaging Center</h1>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📥 Inbox", "📤 Sent", "🗑️ Trash"])
    
    with tab1:
        _display_inbox_tab()
    
    with tab2:
        _display_sent_tab()
    
    with tab3:
        _display_trash_tab()
    
    # Compose message section
    _display_compose_section()

def _display_inbox_tab():
    """Display the inbox tab content"""
    st.markdown('<div class="section-title">Incoming Messages</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Inbox", key="refresh_inbox"):
            st.experimental_rerun()
    
    try:
        with st.spinner("Loading messages..."):
            messages = get_messages(st.session_state.user_email)
            
            if not messages.empty:
                received_msgs = messages[messages['receiver_email'] == st.session_state.user_email.lower()]
                
                if not received_msgs.empty:
                    received_msgs['date'] = pd.to_datetime(received_msgs['sent_at']).dt.date
                    grouped = received_msgs.groupby(['sender_email', 'date'])
                    
                    for (sender, date), group in grouped:
                        with st.expander(f"📨 From: {sender} | 📅 {date.strftime('%Y-%m-%d')}"):
                            for _, msg in group.iterrows():
                                _display_message(msg, inbox=True)
                else:
                    st.info("📭 Your inbox is empty")
            else:
                st.info("📭 Your inbox is empty")
                
    except Exception as e:
        st.error(f"⚠️ Failed to load messages: {str(e)}")
        st.info("Please try refreshing the page")

def _display_sent_tab():
    """Display the sent messages tab content"""
    st.markdown('<div class="section-title">Sent Messages</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Sent", key="refresh_sent"):
            st.experimental_rerun()
    
    try:
        with st.spinner("Loading sent messages..."):
            messages = get_messages(st.session_state.user_email)
            
            if not messages.empty:
                sent_msgs = messages[messages['sender_email'] == st.session_state.user_email.lower()]
                
                if not sent_msgs.empty:
                    sent_msgs['date'] = pd.to_datetime(sent_msgs['sent_at']).dt.date
                    grouped = sent_msgs.groupby(['receiver_email', 'date'])
                    
                    for (receiver, date), group in grouped:
                        with st.expander(f"📨 To: {receiver} | 📅 {date.strftime('%Y-%m-%d')}"):
                            for _, msg in group.iterrows():
                                _display_message(msg, sent=True)
                else:
                    st.info("📭 No sent messages found")
            else:
                st.info("📭 No sent messages found")
                
    except Exception as e:
        st.error(f"⚠️ Failed to load sent messages: {str(e)}")
        st.info("Please try refreshing the page")

def _display_trash_tab():
    """Display the trash tab content"""
    st.markdown('<div class="section-title">Deleted Messages</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Trash", key="refresh_trash"):
            st.experimental_rerun()
        if st.button("🚮 Empty Trash", type="secondary"):
            if empty_trash(st.session_state.user_email):
                st.success("Trash emptied successfully")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error("Failed to empty trash")
    
    try:
        with st.spinner("Loading deleted messages..."):
            messages = get_messages(st.session_state.user_email, include_deleted=True)
            
            if not messages.empty:
                deleted_msgs = messages[messages['is_deleted'] == 1]
                
                if not deleted_msgs.empty:
                    deleted_msgs['type'] = deleted_msgs.apply(
                        lambda x: "📤 Sent" if x['sender_email'] == st.session_state.user_email.lower() else "📥 Received", 
                        axis=1
                    )
                    deleted_msgs['date'] = pd.to_datetime(deleted_msgs['sent_at']).dt.date
                    grouped = deleted_msgs.groupby(['type', 'date'])
                    
                    for (msg_type, date), group in grouped:
                        with st.expander(f"{msg_type} | 📅 {date.strftime('%Y-%m-%d')}"):
                            for _, msg in group.iterrows():
                                _display_message(msg, trash=True)
                else:
                    st.info("🗑️ Trash is empty")
            else:
                st.info("🗑️ Trash is empty")
                
    except Exception as e:
        st.error(f"⚠️ Failed to load deleted messages: {str(e)}")
        st.info("Please try refreshing the page")

def _display_message(msg, inbox=False, sent=False, trash=False):
    """Display a single message with appropriate actions"""
    cols = st.columns([4, 1])
    with cols[0]:
        if trash:
            st.markdown(f"""
            **{'To' if msg['type'] == '📤 Sent' else 'From'}:** {msg['receiver_email'] if msg['type'] == '📤 Sent' else msg['sender_email']}  
            **Subject:** {msg['subject']}  
            **{'Sent' if msg['type'] == '📤 Sent' else 'Received'}:** {msg['sent_at']}  
            **Deleted on:** {msg['deleted_at']}  
            **Message:**  
            {msg['message']}
            """)
        else:
            st.markdown(f"""
            **Subject:** {msg['subject']}  
            **{'Received' if inbox else 'Sent'}:** {msg['sent_at']}  
            **Message:**  
            {msg['message']}
            """)
        
        # Show attachments if any
        if msg['attachment_count'] > 0:
            with st.expander(f"📎 Attachments ({msg['attachment_count']})"):
                attachments = get_message_attachments(msg['id'])
                for _, attachment in attachments.iterrows():
                    with open(attachment['file_path'], "rb") as file:
                        st.download_button(
                            label=f"📄 {attachment['file_name']} ({attachment['file_size']/1024:.1f} KB)",
                            data=file,
                            file_name=attachment['file_name'],
                            mime=attachment['file_type']
                        )
    
    with cols[1]:
        if trash:
            if (st.session_state.user_role == 'admin' or 
                msg['deleted_by'] == st.session_state.user_email):
                if st.button("↩️ Restore", key=f"restore_{msg['id']}"):
                    if restore_message(msg['id']):
                        st.success("Message restored")
                        time.sleep(1)
                        st.experimental_rerun()
                    else:
                        st.error("Failed to restore message")
        else:
            if st.button("🗑️ Delete", key=f"del_{'inbox' if inbox else 'sent'}_{msg['id']}"):
                if delete_message(msg['id'], st.session_state.user_email):
                    st.success("Message moved to trash")
                    time.sleep(1)
                    st.experimental_rerun()
                else:
                    st.error("Failed to delete message")
    
    st.markdown("---")

def _display_compose_section():
    """Display the message composition section"""
    st.markdown("---")
    with st.expander("✍️ Compose New Message", expanded=False):
        conn = None
        try:
            with st.spinner("Loading contacts..."):
                conn = sqlite3.connect(DB_NAME)
                users = pd.read_sql_query(
                    """SELECT u.email, u.full_name 
                       FROM users u 
                       WHERE email != ? AND status = 'approved'
                       ORDER BY u.full_name""", 
                    conn, 
                    params=(st.session_state.user_email,)
                )
                
            if not users.empty:
                users['display'] = users['full_name'] + " (" + users['email'] + ")"
                
                with st.form("compose_message_form"):
                    recipient = st.selectbox(
                        "To:",
                        options=users['display'],
                        help="Select recipient from approved users"
                    )
                    recipient_email = users[users['display'] == recipient]['email'].values[0]
                    
                    subject = st.text_input(
                        "Subject:", 
                        max_chars=100,
                        placeholder="Message subject",
                        help="Maximum 100 characters"
                    )
                    
                    message = st.text_area(
                        "Message:", 
                        height=200, 
                        max_chars=2000,
                        placeholder="Type your message here...",
                        help="Maximum 2000 characters"
                    )
                    
                    attachments = st.file_uploader(
                        "Attachments (optional)",
                        type=['pdf', 'doc', 'docx', 'jpg', 'jpeg', 'png', 'txt'],
                        accept_multiple_files=True,
                        help="Maximum 5MB per file"
                    )
                    
                    if st.form_submit_button("✉️ Send Message", type="primary"):
                        if not subject.strip():
                            st.warning("Please enter a subject")
                        elif not message.strip():
                            st.warning("Please enter a message content")
                        else:
                            valid_attachments = []
                            if attachments:
                                for attachment in attachments:
                                    if attachment.size > 5 * 1024 * 1024:
                                        st.error(f"Attachment {attachment.name} exceeds 5MB limit")
                                        return
                                    valid_attachments.append(attachment)
                            
                            if send_message(
                                sender_email=st.session_state.user_email,
                                receiver_email=recipient_email,
                                subject=subject.strip(),
                                message=message.strip(),
                                attachments=valid_attachments
                            ):
                                st.success("✅ Message sent successfully!")
                                time.sleep(1)
                                st.experimental_rerun()
                            else:
                                st.error("❌ Failed to send message. Please try again.")
            else:
                st.warning("No other active users available to message")
                
        except Exception as e:
            st.error(f"⚠️ Error loading contacts: {str(e)}")
        finally:
            if conn:
                conn.close()

# -------------------------------
# ⚙️ Admin Panel 
# -------------------------------
def show_admin_panel():
    """Show comprehensive admin management interface with all error fixes"""
    if st.session_state.user_role != 'admin':
        st.warning("⛔ Admin access only")
        return
    
    # CSS Styling
    st.markdown("""
    <style>
    .admin-tab {
        padding: 1rem;
        border-radius: 0.5rem;
        background: rgba(255,255,255,0.8);
        margin-bottom: 1rem;
    }
    .admin-card {
        background: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .model-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .system-card {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .action-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="section-title">⚙️ Admin Panel</h1>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Pending Approvals", 
        "👥 User Management", 
        "🤖 Model Management", 
        "⚙ System Settings"
    ])
    
    # Tab 1: Pending Approvals
    with tab1:
        _display_pending_approvals()
    
    # Tab 2: User Management
    with tab2:
        _display_user_management()
    
    # Tab 3: Model Management
    with tab3:
        _display_model_management()
    
    # Tab 4: System Settings
    with tab4:
        _display_system_settings()

def _display_pending_approvals():
    """Display pending user approvals with Streamlit buttons"""
    st.markdown('<div class="section-title">Pending Registrations</div>', unsafe_allow_html=True)
    pending_users = get_pending_registrations()
    
    if pending_users is not None and not pending_users.empty:
        for _, user in pending_users.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="admin-card">
                    <h3>{user['full_name']} ({user['email']})</h3>
                    <p><strong>Role:</strong> {user['role'].capitalize()}</p>
                    <p><strong>Registered:</strong> {user['created_at']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Approve {user['id']}", key=f"approve_{user['id']}"):
                        if approve_user(user['id'], st.session_state.user_id):
                            st.success("User approved")
                            st.experimental_rerun()
                with col2:
                    if st.button(f"❌ Reject {user['id']}", key=f"reject_{user['id']}"):
                        if reject_user(user['id'], st.session_state.user_id):
                            st.success("User rejected")
                            st.experimental_rerun()
    else:
        st.info("No pending registrations found")

def _display_user_management():
    """Display user management interface with working action buttons"""
    st.markdown('<div class="section-title">User Accounts</div>', unsafe_allow_html=True)

    users_df = _get_users_data()
    if users_df is None or users_df.empty:
        st.info("No users found in the database")
        _display_add_user_form()
        return

    gb = GridOptionsBuilder.from_dataframe(users_df)
    gb.configure_default_column(editable=False, filterable=True, sortable=True)
    gb.configure_selection('single', use_checkbox=True)
    grid_options = gb.build()

    grid_response = AgGrid(
        users_df,
        gridOptions=grid_options,
        height=400,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        theme='streamlit',
        fit_columns_on_grid_load=True
    )

    selected_rows = grid_response.get("selected_rows", [])

    if isinstance(selected_rows, list):
        selected_user = selected_rows[0] if len(selected_rows) > 0 else None
    elif isinstance(selected_rows, pd.DataFrame):
        selected_user = selected_rows.iloc[0].to_dict() if not selected_rows.empty else None
    else:
        selected_user = None

    st.markdown("### User Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ Activate", disabled=not selected_user):
            _update_user_status(selected_user['id'], 'approved')

    with col2:
        if st.button("🚫 Deactivate", disabled=not selected_user):
            _update_user_status(selected_user['id'], 'suspended')

    with col3:
        if st.button("🔒 Reset Password", disabled=not selected_user):
            _reset_password_form(selected_user)

    with col4:
        confirm_key = "confirm_delete"
        if selected_user:
            confirm = st.checkbox("Confirm delete", key=confirm_key)
        else:
            confirm = False

        if st.button("🗑️ Delete", disabled=not (selected_user and confirm)):
            _delete_user(selected_user)

    _display_add_user_form()
        
def _display_model_management():
    """Display model management interface with robust deployment capabilities"""
    with st.container():
        st.markdown("""
        <div class="model-card">
            <h3>Model Deployment</h3>
        """, unsafe_allow_html=True)
        
        # Display current active model
        current_model = st.session_state.get('current_model', MODEL_PATH)
        st.markdown(f"**Current Model:** `{current_model}`")
        
        with st.form("model_update_form", clear_on_submit=True):
            cols = st.columns(2)
            with cols[0]:
                new_model = st.file_uploader(
                    "Upload model file (.h5)", 
                    type=["h5"],
                    help="Supported formats: Keras .h5 files"
                )
            with cols[1]:
                version = st.text_input(
                    "Version Number*", 
                    placeholder="e.g., 1.2.0",
                    help="Follow semantic versioning (major.minor.patch)"
                )
            
            description = st.text_area(
                "Description", 
                placeholder="Brief description of model improvements",
                max_chars=500
            )
            
            release_notes = st.text_area(
                "Release Notes", 
                placeholder="Detailed changes in this version",
                max_chars=1000
            )
            
            if st.form_submit_button("🚀 Deploy Model", use_container_width=True):
                if new_model and version:
                    try:
                        # Validate version format
                        if not re.match(r'^\d+\.\d+\.\d+$', version):
                            raise ValueError("Version must follow semantic versioning (e.g., 1.2.0)")
                        
                        # Create model directory if it doesn't exist
                        os.makedirs(MODEL_DIR, exist_ok=True)
                        
                        # Save the new model file
                        model_filename = f"model_v{version.replace('.', '_')}.h5"
                        model_path = os.path.join(MODEL_DIR, model_filename)
                        
                        with open(model_path, "wb") as f:
                            f.write(new_model.getbuffer())
                        
                        # Validate the model can be loaded
                        try:
                            test_model = load_model(model_path)
                            input_shape = str(test_model.input_shape)
                            del test_model  # Clean up
                        except Exception as e:
                            os.remove(model_path)
                            raise ValueError(f"Invalid model file: {str(e)}")
                        
                        # Store relative path in database for portability
                        rel_path = str(Path(model_path).relative_to(REPO_ROOT))
                        
                        # Save model info to database
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO model_versions 
                            (version, description, release_notes, path, uploaded_by, performance_metrics)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            version, 
                            description if description else None,
                            release_notes if release_notes else None,
                            rel_path,  # Store relative path
                            st.session_state.user_id,
                            json.dumps({"input_shape": input_shape})
                        ))
                        conn.commit()
                        
                        # Update the current model in session state
                        st.session_state.current_model = model_path
                        
                        st.success(f"✅ Model v{version} deployed successfully!")
                        st.balloons()
                        time.sleep(1)
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Deployment failed: {str(e)}")
                        if 'model_path' in locals() and os.path.exists(model_path):
                            try:
                                os.remove(model_path)
                            except:
                                pass
                else:
                    st.warning("⚠️ Please upload a model file and specify version number")
        
        st.markdown("""
            <div class="small-text">
                * Required field<br>
                Max model size: 500MB<br>
                Supported frameworks: TensorFlow/Keras
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display model history
        st.markdown("""
        <div class="model-card">
            <h3>Model Version History</h3>
        """, unsafe_allow_html=True)
        
        conn = None
        try:
            conn = sqlite3.connect(DB_NAME)
            models = pd.read_sql_query('''
                SELECT 
                    id, version, description, 
                    datetime(uploaded_at, 'localtime') as uploaded_at,
                    path,
                    (SELECT full_name FROM users WHERE id = uploaded_by) as uploaded_by
                FROM model_versions
                ORDER BY uploaded_at DESC
            ''', conn)
            
            if not models.empty:
                # Convert relative paths to absolute and check existence
                models['full_path'] = models['path'].apply(
                    lambda x: str(REPO_ROOT / x) if x and not os.path.isabs(x) else x
                )
                models['exists'] = models['full_path'].apply(
                    lambda x: os.path.exists(x) if x else False
                )
                models['Current'] = models['full_path'] == st.session_state.get('current_model', '')
                
                # Filter out models with missing files
                valid_models = models[models['exists']]
                
                if not valid_models.empty:
                    gb = GridOptionsBuilder.from_dataframe(valid_models)
                    gb.configure_default_column(
                        editable=False,
                        filterable=True,
                        sortable=True,
                        resizable=True
                    )
                    gb.configure_column(
                        "Current",
                        header_name="Active",
                        cellRenderer="agCheckboxCellRenderer",
                        width=80
                    )
                    grid_options = gb.build()
                    
                    grid_response = AgGrid(
                        valid_models,
                        gridOptions=grid_options,
                        height=300,
                        width='100%',
                        theme='streamlit',
                        update_mode=GridUpdateMode.SELECTION_CHANGED
                    )
                    
                    # Model actions
                    selected_rows = grid_response['selected_rows']
                    if selected_rows:
                        selected_model = selected_rows[0]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔍 View Details", use_container_width=True):
                                st.json({
                                    "Version": selected_model['version'],
                                    "Uploaded By": selected_model['uploaded_by'],
                                    "Upload Date": selected_model['uploaded_at'],
                                    "Path": selected_model['full_path'],
                                    "Description": selected_model['description']
                                })
                        
                        with col2:
                            if st.button("🔄 Set as Active", use_container_width=True,
                                       disabled=selected_model['Current']):
                                try:
                                    if os.path.exists(selected_model['full_path']):
                                        st.session_state.current_model = selected_model['full_path']
                                        st.success(f"Model v{selected_model['version']} is now active")
                                        time.sleep(1)
                                        st.experimental_rerun()
                                    else:
                                        st.error("Model file not found")
                                except Exception as e:
                                    st.error(f"Error activating model: {str(e)}")
                else:
                    st.warning("No valid model files found in the database")
                
                # Show missing models if any
                missing_models = models[~models['exists']]
                if not missing_models.empty:
                    with st.expander("⚠️ Missing Model Files", expanded=False):
                        st.write("The following models are registered but files are missing:")
                        st.dataframe(missing_models[['version', 'path', 'uploaded_at']])
            else:
                st.info("No model versions found in database")
                
        except Exception as e:
            st.error(f"Error loading model history: {str(e)}")
        finally:
            if conn:
                conn.close()
        
        st.markdown("</div>", unsafe_allow_html=True)

def _display_system_settings():
    """Display system settings interface"""
    st.markdown('<div class="section-title">System Configuration</div>', unsafe_allow_html=True)
    
    # Database Maintenance
    with st.container():
        st.markdown("""
        <div class="system-card">
            <h3>Database Maintenance</h3>
        """, unsafe_allow_html=True)
        
        conn = None
        try:
            conn = sqlite3.connect(DB_NAME)
            db_info = {
                "Database Path": os.path.abspath(DB_NAME),
                "Size": f"{os.path.getsize(DB_NAME)/1024/1024:.2f} MB",
                "Tables": pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist(),
                "Schema Version": pd.read_sql_query("SELECT MAX(version) FROM migrations", conn).iloc[0,0]
            }
            st.json(db_info)
            
            # Backup and restore
            st.markdown("### Backup & Restore")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Create Backup", help="Create a full database backup", use_container_width=True):
                    _create_db_backup(conn)
            
            with col2:
                backup_files = _get_backup_files()
                selected_backup = st.selectbox("Select backup to restore", backup_files)
                
                if selected_backup and st.button("🔄 Restore Backup", 
                                              help="Restore from selected backup",
                                              use_container_width=True,
                                              type="secondary"):
                    _restore_backup(selected_backup)
        
        except Exception as e:
            st.error(f"Database error: {str(e)}")
        finally:
            if conn:
                conn.close()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System Logs
    with st.container():
        st.markdown("""
        <div class="system-card">
            <h3>System Logs</h3>
        """, unsafe_allow_html=True)
        
        log_level = st.selectbox("Log level filter", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        num_entries = st.slider("Number of entries", 10, 500, 100)
        
        conn = None
        try:
            conn = sqlite3.connect(DB_NAME)
            logs = pd.read_sql_query(f'''
                SELECT level, message, datetime(created_at, 'localtime') as timestamp,
                       (SELECT full_name FROM users WHERE id = user_id) as user
                FROM system_logs
                WHERE level >= ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', conn, params=(log_level, num_entries))
            
            if not logs.empty:
                st.dataframe(logs, use_container_width=True)
                
                if st.button("Clear Logs", type="secondary"):
                    _clear_logs(conn)
            else:
                st.info("No log entries found")
        except Exception as e:
            st.error(f"Error accessing logs: {str(e)}")
        finally:
            if conn:
                conn.close()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System Information
    with st.container():
        st.markdown("""
        <div class="system-card">
            <h3>System Information</h3>
        """, unsafe_allow_html=True)
        
        st.json(_get_system_info())
        
        st.markdown("### System Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache", help="Clear all cached data", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared")
        
        with col2:
            if st.button("Restart App", type="secondary", help="Restart the application", use_container_width=True):
                st.experimental_rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def _get_users_data():
    """Get users data from database"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        return pd.read_sql_query(
            "SELECT id, full_name, email, role, status FROM users WHERE id != ?",
            conn,
            params=(st.session_state.user_id,)
        )
    except Exception as e:
        st.error(f"Error fetching users: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def _update_user_status(user_id, status):
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
        conn.commit()
        st.success(f"User set to '{status}'")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Failed to update status: {e}")
    finally:
        conn.close()

def _reset_password_form(user):
    with st.form("reset_password_form"):
        new_pass = st.text_input("New Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        if st.form_submit_button("Confirm"):
            if new_pass == confirm:
                conn = sqlite3.connect(DB_NAME)
                try:
                    hashed = hash_password(new_pass)
                    conn.execute("UPDATE users SET password = ? WHERE id = ?", (hashed, user['id']))
                    conn.commit()
                    st.success("Password reset")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    conn.close()
            else:
                st.error("Passwords do not match")

def _delete_user(user):
    if user is None:
        st.error("No user selected for deletion.")
        return
    
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute("DELETE FROM users WHERE id = ?", (user['id'],))
        conn.commit()
        st.success(f"User '{user['email']}' deleted successfully!")
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Delete failed: {str(e)}")
    finally:
        if conn:
            conn.close()

def _display_add_user_form():
    """Display form to add new user"""
    with st.container():
        st.markdown("""
        <div class="admin-card">
            <h3>Add New User</h3>
        """, unsafe_allow_html=True)
        
        with st.form("add_user_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input("Full Name", key="new_user_name")
                new_email = st.text_input("Email", key="new_user_email")
            with col2:
                new_role = st.selectbox("Role", ["admin", "doctor", "assistant"], key="new_user_role")
                new_status = st.selectbox("Status", ["approved", "pending"], key="new_user_status")
            
            if st.form_submit_button("Add User"):
                if not new_name or not new_email:
                    st.error("Name and email are required")
                else:
                    _add_new_user(new_name, new_email, new_role, new_status)
        
        st.markdown("</div>", unsafe_allow_html=True)

def _add_new_user(name, email, role, status):
    """Add new user to database"""
    temp_password = "password123"  # In production, generate a random password
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute(
            "INSERT INTO users (full_name, email, password, role, status) VALUES (?, ?, ?, ?, ?)",
            (name, email.lower(), hash_password(temp_password), role, status)
        )
        conn.commit()
        st.success(f"User added successfully. Temporary password: {temp_password}")
        time.sleep(1)
        st.experimental_rerun()
    except sqlite3.IntegrityError:
        st.error("Email already exists")
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Error adding user: {str(e)}")
    finally:
        if conn:
            conn.close()

def _create_db_backup(conn):
    """Create database backup"""
    backup_dir = "backups"
    os.makedirs(backup_dir, exist_ok=True)
    backup_file = f"{backup_dir}/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    backup_conn = None
    try:
        backup_conn = sqlite3.connect(backup_file)
        with backup_conn:
            conn.backup(backup_conn)
        st.success(f"Backup created: {backup_file}")
    except Exception as e:
        st.error(f"Backup failed: {str(e)}")
    finally:
        if backup_conn:
            backup_conn.close()

def _get_backup_files():
    """Get list of available backup files"""
    return [f for f in os.listdir("backups") if f.endswith(".db")] if os.path.exists("backups") else []

def _restore_backup(backup_file):
    """Restore database from backup"""
    if st.checkbox("Confirm restore - THIS WILL OVERWRITE CURRENT DATA"):
        try:
            shutil.copy2(f"backups/{backup_file}", DB_NAME)
            st.success("Database restored successfully!")
            time.sleep(1)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Restore failed: {str(e)}")
            
def _clear_logs(conn):
    """Clear system logs"""
    try:
        conn.execute("DELETE FROM system_logs")
        conn.commit()
        st.success("Logs cleared")
        time.sleep(1)
        st.experimental_rerun()
    except Exception as e:
        conn.rollback()
        st.error(f"Error clearing logs: {str(e)}")

def _get_system_info():
    """Get system information"""
    return {
        "System": {
            "Python Version": sys.version.split()[0],
            "Streamlit Version": st.__version__,
            "TensorFlow Version": tf.__version__,
            "Pandas Version": pd.__version__,
            "SQLite Version": sqlite3.sqlite_version
        },
        "Paths": {
            "Working Directory": os.getcwd(),
            "Database Path": os.path.abspath(DB_NAME),
            "Model Path": os.path.abspath(MODEL_PATH)
        },
        "Session": {
            "User": st.session_state.user_email,
            "Role": st.session_state.user_role,
            "Session Start": st.session_state.get("session_start", "N/A")
        }
    }

# -------------------------------
# 🧭 Main Navigation
# -------------------------------
def main():
    """Main application entry point"""
    # Set page config (MUST BE FIRST STREAMLIT COMMAND)
    st.set_page_config(
        page_title="Munthandiz Cataract Detection",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Then load custom CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&family=Playfair+Display:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Open Sans', sans-serif;
        font-size: 14px;
        line-height: 1.5;
        font-weight: 400;
        color: #2d3748;
    }

    .main-title, .section-title, .hero-title, .step-title {
        font-family: 'Playfair Display', serif;
    }

    .main {
        background: linear-gradient(135deg, #98F5E1 0%, #B8F5D1 50%, #D1F5E8 100%);
        min-height: 100vh;
        background-image: url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-blend-mode: overlay;
    }
    .block-container {
        padding: 1.5rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    .block-container:hover {
        background: rgba(255, 255, 255, 0.85);
    }

    /* Titles */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3748;
        text-align: center;
        margin-bottom: 0.75rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 0.25px;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #B3B9B6FF;
        position: relative;
    }
    .section-title::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 40px;
        height: 2px;
        background: #38a169;
    }

    /* Card Styling */
    .feature-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin-bottom: 1.25rem;
        transition: all 0.3s ease;
        position: relative;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #48bb78, #38a169);
    }

    /* Metric Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(8px);
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #718096;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #48bb78, #38a169);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }

    /* Status Tags */
    .status-success, .status-error, .status-warning {
        font-family: 'Open Sans', sans-serif;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        text-align: center;
        margin: 0.75rem 0;
        color: white;
    }
    .status-success {
        background: linear-gradient(135deg, #48bb78, #38a169);
    }
    .status-error {
        background: linear-gradient(135deg, #f56565, #e53e3e);
    }
    .status-warning {
        background: linear-gradient(135deg, #ed8936, #dd6b20);
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        font-family: 'Open Sans', sans-serif !important;
        background: linear-gradient(135deg, #48bb78, #38a169) !important;
        color: white !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 2px 8px rgba(72, 187, 120, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(72, 187, 120, 0.4) !important;
    }

    /* Animations */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .slide-up {
        animation: slideUp 0.6s ease-out;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Verify and maintain session state
    if not verify_session():
        show_auth_page()
        return
    
    # Initialize navigation state
    if 'nav' not in st.session_state:
        st.session_state.nav = "Home"
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #2d3748;">Munthandiz</h1>
            <div style="font-size: 0.9rem; color: #4a5568;">Cataract Detection System</div>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #718096;">Welcome, {st.session_state.user_name}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        menu_options = {
            "🏠 Home": "Home",
            "👁️ Detection": "Detection",
            "📅 Appointments": "Appointments",
            "📊 Analytics": "Analytics",
            "📧 Messages": "Messages"
        }
        
        # Add admin panel if user is admin
        if st.session_state.user_role == "admin":
            menu_options["⚙️ Admin"] = "Admin"
        
        # Create navigation buttons
        for label, page in menu_options.items():
            if st.sidebar.button(label, use_container_width=True, key=f"nav_{page}"):
                st.session_state.nav = page
                st.experimental_rerun()
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.session_state.clear()
            st.experimental_rerun()
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="font-size: 0.75rem; color: #718096; text-align: center;">
            System Status: <span style="color: #38a169;">●</span> Operational
            <br>v1.0.0
        </div>
        """, unsafe_allow_html=True)
    
    # Page routing
    if st.session_state.nav == "Home":
        show_home_page()
    elif st.session_state.nav == "Detection":
        show_detection_page()
    elif st.session_state.nav == "Appointments":
        show_appointments_page()
    elif st.session_state.nav == "Analytics":
        show_analytics_page()
    elif st.session_state.nav == "Messages":
        show_messages_page()
    elif st.session_state.nav == "Admin" and st.session_state.user_role == "admin":
        show_admin_panel()
    else:
        st.warning("Page not found")
        st.session_state.nav = "Home"
        st.experimental_rerun()

if __name__ == "__main__":
    main()