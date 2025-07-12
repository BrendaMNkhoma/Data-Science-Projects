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
import string
import os
import random
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
DB_NAME = "cataract_detection_system.db"
MODELS_DIR = "models"  # Relative to REPO_ROOT
CLASS_NAMES = ['conjunctival_growth', 'mild', 'normal', 'severe']
SESSION_TIMEOUT_MINUTES = 60  # Increased timeout to 1 hour
REPO_ROOT = Path(__file__).parent  # Added for path resolution
MODEL_INPUT_SIZE = (224, 224)  # Expected input size for the model

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
    """Initialize the database with complete schema and migrations"""
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
                        registration_date DATE DEFAULT CURRENT_DATE,
                        contact_number TEXT,
                        last_visit_date DATE
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id INTEGER NOT NULL,
                        detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        result TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        attended_by TEXT NOT NULL,
                        notes TEXT,
                        image_path TEXT,
                        last_updated TIMESTAMP,
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
        
        # Migration 4: Add system_settings table (v4)
        if current_version < 4:
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                ''')
                
                # Initialize with empty active model setting
                cursor.execute('''
                    INSERT OR IGNORE INTO system_settings (key, value)
                    VALUES ('active_model', '')
                ''')
                
                cursor.execute("INSERT INTO migrations (version) VALUES (4)")
                conn.commit()
                current_version = 4
            except Exception as e:
                conn.rollback()
                raise Exception(f"Migration 4 failed: {str(e)}")
        
        # Migration 5: Enhance detections and add audit logs (v5)
        if current_version < 5:
            try:
                # Add columns to detections table
                cursor.execute('''
                    ALTER TABLE detections 
                    ADD COLUMN image_path TEXT
                ''')
                
                cursor.execute('''
                    ALTER TABLE detections 
                    ADD COLUMN last_updated TIMESTAMP
                ''')
                
                # Create detection logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detection_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        patient_id INTEGER,
                        detection_id INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id),
                        FOREIGN KEY (patient_id) REFERENCES patients(id),
                        FOREIGN KEY (detection_id) REFERENCES detections(id)
                    )
                ''')
                
                # Add indexes for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_detections_patient_id 
                    ON detections(patient_id)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_detections_date 
                    ON detections(detection_date)
                ''')
                
                cursor.execute("INSERT INTO migrations (version) VALUES (5)")
                conn.commit()
                current_version = 5
            except Exception as e:
                conn.rollback()
                raise Exception(f"Migration 5 failed: {str(e)}")
        
        # Migration 6: Create useful views (v6)
        if current_version < 6:
            try:
                # Create view for patient-detection join
                cursor.execute('''
                    CREATE VIEW IF NOT EXISTS patient_detections_view AS
                    SELECT 
                        d.id,
                        d.patient_id,
                        p.full_name,
                        p.gender,
                        p.age,
                        p.village,
                        p.district,
                        d.result,
                        d.confidence,
                        d.attended_by,
                        d.notes,
                        d.detection_date,
                        d.last_updated
                    FROM detections d
                    JOIN patients p ON d.patient_id = p.id
                ''')
                
                # Create view for detection statistics
                cursor.execute('''
                    CREATE VIEW IF NOT EXISTS detection_stats_view AS
                    SELECT 
                        result,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence,
                        MAX(detection_date) as last_detection
                    FROM detections
                    GROUP BY result
                ''')
                
                cursor.execute("INSERT INTO migrations (version) VALUES (6)")
                conn.commit()
                current_version = 6
            except Exception as e:
                conn.rollback()
                raise Exception(f"Migration 6 failed: {str(e)}")
        
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
    """Load and validate the active cataract detection model"""
    model_info = get_active_model_info()
    if not model_info or 'path' not in model_info:
        st.error("❌ No active model configured in system settings")
        return None
    
    try:
        # Resolve the model path (handle both relative and absolute paths)
        model_path = Path(model_info['path'])
        if not model_path.is_absolute():
            model_path = REPO_ROOT / model_path
        
        # Verify the model file exists
        if not model_path.exists():
            st.error(f"❌ Model file not found at: {model_path}")
            return None
        
        # Load the TensorFlow model
        model = load_model(model_path)
        
        # Verify the model has the expected structure
        if not hasattr(model, 'predict'):
            st.error("❌ Invalid model format - missing predict method")
            return None
            
        # Verify output shape matches expected classes
        if model.output_shape[1] != len(CLASS_NAMES):
            st.error(f"❌ Model expects {model.output_shape[1]} classes but we have {len(CLASS_NAMES)}")
            return None
            
        return model
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

@st.cache_resource
def get_active_model_info():
    """Get information about the currently active model"""
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT mv.*, u.full_name as uploaded_by
            FROM model_versions mv
            JOIN users u ON mv.uploaded_by = u.id
            WHERE mv.path = (SELECT value FROM system_settings WHERE key = 'active_model')
            ORDER BY mv.uploaded_at DESC
            LIMIT 1
        ''')
        
        model_info = cursor.fetchone()
        if model_info:
            return {
                "id": model_info[0],
                "version": model_info[1],
                "description": model_info[2],
                "path": model_info[4],
                "uploaded_at": model_info[6],
                "uploaded_by": model_info[9]
            }
        return {}
    except Exception as e:
        st.error(f"Error getting active model: {str(e)}")
        return {}
    finally:
        conn.close()

def delete_model_version(model_id):
    """Permanently delete a model version from system"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # First check if this is the active model
        cursor.execute('''
            SELECT path FROM model_versions WHERE id = ?
        ''', (model_id,))
        model_path = cursor.fetchone()
        
        if model_path:
            model_path = model_path[0]
            abs_path = os.path.join(REPO_ROOT, model_path)
            
            # Check if this is the active model
            cursor.execute('''
                SELECT value FROM system_settings WHERE key = 'active_model'
            ''')
            active_path = cursor.fetchone()
            
            if active_path and active_path[0] == model_path:
                raise ValueError("Cannot delete the active model")
            
            # Delete the file
            if os.path.exists(abs_path):
                os.remove(abs_path)
            
            # Delete the database record
            cursor.execute('''
                DELETE FROM model_versions WHERE id = ?
            ''', (model_id,))
            
            conn.commit()
            return True
        return False
        
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Failed to delete model: {str(e)}")
        return False
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
    
def save_detection(patient_id, result, confidence, attended_by, notes="", image_path=None):
    """Save detection results to database with robust error handling"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Start transaction
        cursor.execute("BEGIN TRANSACTION")
        
        # Insert detection record with image_path if provided
        if image_path:
            cursor.execute('''
                INSERT INTO detections 
                (patient_id, result, confidence, attended_by, notes, image_path, detection_date)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (patient_id, result, confidence, attended_by, notes, image_path))
        else:
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

def get_patient(patient_id):
    """Get single patient by ID with error handling"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM patients 
            WHERE id = ?
        ''', (patient_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
        return None
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def handle_new_patient():
    """Quick patient registration form"""
    with st.form("new_patient_form"):
        cols = st.columns(2)
        full_name = cols[0].text_input("Full Name*")
        age = cols[1].number_input("Age*", 1, 120)
        gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
        village = cols[0].text_input("Village*")
        district = cols[1].text_input("District*")
        
        if st.form_submit_button("Register"):
            if not all([full_name, age, village, district]):
                st.error("Missing required fields")
                return None, None
                
            conn = None
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO patients 
                    (full_name, age, gender, village, district)
                    VALUES (?, ?, ?, ?, ?)
                ''', (full_name, age, gender, village, district))
                patient_id = cursor.lastrowid
                conn.commit()
                st.success("Patient registered!")
                return patient_id, get_patient(patient_id)
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")
                return None, None
            finally:
                if conn:
                    conn.close()
    return None, None

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
                            st.rerun()
                        else:
                            st.error("Login failed - please try again", icon="🚨")

            # Registration switch
            st.markdown('<div class="auth-switch">Don\'t have an account?</div>', unsafe_allow_html=True)
            if st.button("Register here", key="to_register", use_container_width=True):
                st.session_state.auth_mode = "register"
                st.rerun()

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
                            st.rerun()
                        else:
                            st.error("Registration failed - please try again", icon="🚨")

            # Login switch
            st.markdown('<div class="auth-switch">Already have an account?</div>', unsafe_allow_html=True)
            if st.button("Sign in here", key="to_login", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()

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
            st.rerun()
    
    with action_col2:
        if st.button("📅 Schedule", use_container_width=True):
            st.session_state.nav = "Appointments"
            st.rerun()
    
    with action_col3:
        if st.button("✉️ Messages", use_container_width=True):
            st.session_state.nav = "Messages"
            st.rerun()
    
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
    """Show the fully functional cataract detection interface"""
    st.markdown('<h1 class="section-title">👁️ Cataract Detection</h1>', unsafe_allow_html=True)

    # Initialize session state for detection results
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = None

    # Display current model info
    model_info = get_active_model_info()
    with st.expander("ℹ️ Current Model Information", expanded=True):
        if model_info:
            st.markdown(f"""
            - **Version:** {model_info.get('version', 'N/A')}
            - **Description:** {model_info.get('description', 'N/A')}
            - **Uploaded by:** {model_info.get('uploaded_by', 'N/A')}
            """)
        else:
            st.warning("No active model configured")

    # Main tabs
    tab1, tab2 = st.tabs(["New Detection", "Manage Detections"])

    # TAB 1: New Detection
    with tab1:
        use_camera = st.checkbox("🎥 Use Camera", value=False, key="use_camera")

        # Patient Selection Section
        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
        
        # Using columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Existing Patient Selection
            with st.container():
                st.markdown("#### Select Existing Patient")
                patients = get_patients()
                if patients.empty:
                    st.info("No patients found")
                else:
                    patient_options = patients.apply(
                        lambda x: f"{x['full_name']} ({x['age']}y, {x['village']})", 
                        axis=1
                    )
                    selected_index = st.selectbox(
                        "Choose patient", 
                        range(len(patient_options)),
                        format_func=lambda x: patient_options[x],
                        key="patient_select"
                    )
                    if selected_index is not None:
                        st.session_state.selected_patient = patients.iloc[selected_index].to_dict()

        with col2:
            # New Patient Registration
            with st.container():
                st.markdown("#### Register New Patient")
                with st.form("patient_form", clear_on_submit=True):
                    full_name = st.text_input("Full Name*")
                    age = st.number_input("Age*", min_value=0, max_value=120, value=30)
                    gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
                    village = st.text_input("Village*")
                    district = st.text_input("District*")
                    
                    if st.form_submit_button("Register Patient"):
                        if not all([full_name, age, village, district]):
                            st.error("Please fill required fields (*)")
                        else:
                            patient_id = add_patient(
                                full_name=full_name,
                                gender=gender,
                                age=age,
                                village=village,
                                district=district,
                                traditional_authority="",
                                marital_status=""
                            )
                            if patient_id:
                                st.session_state.selected_patient = get_patient_by_id(patient_id)
                                st.success("Patient registered successfully!")
                                st.rerun()

        # Only show detection interface if patient is selected
        if st.session_state.selected_patient:
            patient = st.session_state.selected_patient
            st.markdown('<div class="section-title">Patient Details</div>', unsafe_allow_html=True)
            st.markdown(f"""
            - **Name:** {patient['full_name']}
            - **Age:** {patient['age']}
            - **Location:** {patient['village']}, {patient['district']}
            """)

            # Image Capture Section
            st.markdown('<div class="section-title">Eye Image Capture</div>', unsafe_allow_html=True)
            
            img = None
            if use_camera:
                img = st.camera_input("Capture eye image", key="camera_capture")
            else:
                img = st.file_uploader(
                    "Upload eye image", 
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=False,
                    key="file_upload"
                )

            if img:
                # Display preview
                st.image(img, caption="Eye Image Preview", use_column_width=True)

                # Analysis button
                if st.button("Analyze Image", key="analyze_btn", type="primary"):
                    with st.spinner("Analyzing..."):
                        try:
                            # Create temp file
                            temp_dir = "temp_images"
                            os.makedirs(temp_dir, exist_ok=True)
                            temp_file = os.path.join(temp_dir, f"eye_{patient['id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
                            
                            # Save image
                            with open(temp_file, "wb") as f:
                                f.write(img.getbuffer() if use_camera else img.getvalue())

                            # Load model and predict
                            model = load_detection_model()
                            if model:
                                predicted_class, confidence = predict_image(temp_file, model)
                                if predicted_class:
                                    st.session_state.detection_results = {
                                        "patient_id": patient['id'],
                                        "image_path": temp_file,
                                        "predicted_class": predicted_class,
                                        "confidence": confidence,
                                        "timestamp": datetime.now()
                                    }
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            if os.path.exists(temp_file):
                                os.remove(temp_file)

            # Show results if available
            if st.session_state.detection_results and st.session_state.detection_results['patient_id'] == patient['id']:
                results = st.session_state.detection_results
                
                st.markdown("### Analysis Results")
                
                # Color coding for results
                result_colors = {
                    'normal': 'green',
                    'mild': 'orange',
                    'severe': 'red',
                    'conjunctival_growth': 'purple'
                }
                result_color = result_colors.get(results['predicted_class'], 'blue')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **Patient:** {patient['full_name']}  
                    **Age:** {patient['age']}  
                    **Gender:** {patient['gender']}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Result:** <span style='color:{result_color}; font-weight:bold'>
                        {results['predicted_class'].replace('_', ' ').title()}
                    </span>  
                    **Confidence:** {results['confidence']:.2f}%
                    """, unsafe_allow_html=True)
                
                # Save form
                with st.form("save_detection_form"):
                    notes = st.text_area("Clinical Notes", height=100)
                    
                    if st.form_submit_button("Save Detection", type="primary"):
                        detection_id = save_detection(
                            patient_id=patient['id'],
                            result=results['predicted_class'],
                            confidence=results['confidence'],
                            attended_by=st.session_state.user_name,
                            notes=notes,
                            image_path=results['image_path']
                        )
                        
                        if detection_id:
                            # Clean up
                            del st.session_state.detection_results
                            st.session_state.selected_patient = None
                            st.success("Detection saved successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to save detection")

    # TAB 2: Manage Detections
    with tab2:
        st.markdown('<div class="section-title">Manage Detection Results</div>', unsafe_allow_html=True)
        
        # Add filters
        with st.expander("🔍 Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                date_filter = st.date_input("Filter by date")
            with col2:
                result_filter = st.multiselect(
                    "Filter by result",
                    options=CLASS_NAMES,
                    default=[]
                )
            with col3:
                min_confidence = st.slider(
                    "Minimum confidence", 
                    min_value=0, 
                    max_value=100, 
                    value=70
                )

        # Get filtered detections
        detections = get_detections()
        
        if not detections.empty:
            # Apply filters
            if date_filter:
                detections = detections[pd.to_datetime(detections['detection_date']).dt.date == date_filter]
            if result_filter:
                detections = detections[detections['result'].isin(result_filter)]
            detections = detections[detections['confidence'] >= min_confidence]

        if detections.empty:
            st.info("No detections found matching filters")
        else:
            # Configure interactive grid
            gb = GridOptionsBuilder.from_dataframe(detections)
            gb.configure_default_column(
                editable=False,
                filterable=True,
                sortable=True,
                resizable=True
            )
            
            # Configure columns
            gb.configure_column("result", editable=True, header_name="Result")
            gb.configure_column("confidence", 
                              editable=True, 
                              header_name="Confidence", 
                              type=["numericColumn", "numberColumnFilter"])
            gb.configure_column("notes", editable=True, header_name="Notes")
            
            # Configure selection
            gb.configure_selection(
                selection_mode="multiple",
                use_checkbox=True,
                groupSelectsChildren=True
            )
            
            grid_options = gb.build()

            # Display the grid
            grid_response = AgGrid(
                detections,
                gridOptions=grid_options,
                height=500,
                width='100%',
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                theme='streamlit',
                key='detections_grid'
            )

            # Action buttons
            selected_rows = grid_response.get("selected_rows", [])
            
            st.markdown("### Selected Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Changes", key="save_changes_btn"):
                    updated_data = grid_response['data']
                    success_count = 0
                    
                    for _, row in updated_data.iterrows():
                        if update_detection(
                            row['id'],
                            row['result'],
                            row['confidence'],
                            row['notes']
                        ):
                            success_count += 1
                    
                    st.success(f"Updated {success_count} records")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Refresh Data", key="refresh_data_btn"):
                    st.rerun()
            
            with col3:
                if selected_rows and st.button(
                    "🗑️ Delete Selected",
                    key="delete_selected_btn",
                    type="primary"
                ):
                    delete_count = 0
                    for row in selected_rows:
                        if delete_detection(row['id']):
                            delete_count += 1
                    
                    st.success(f"Deleted {delete_count} records")
                    st.rerun()

            # Detailed view for single selection
            if len(selected_rows) == 1:
                selected = selected_rows[0]
                st.markdown("---")
                st.markdown(f"### Detailed View: Detection #{selected['id']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Patient Information")
                    st.markdown(f"""
                    - **Name:** {selected['full_name']}
                    - **Age:** {selected['age']}
                    - **Gender:** {selected['gender']}
                    - **Location:** {selected['village']}, {selected['district']}
                    """)
                
                with col2:
                    st.markdown("#### Detection Details")
                    result_color = result_colors.get(selected['result'], 'blue')
                    st.markdown(f"""
                    - **Result:** <span style='color:{result_color}; font-weight:bold'>
                        {selected['result'].replace('_', ' ').title()}
                    </span>
                    - **Confidence:** {selected['confidence']:.2f}%
                    - **Date:** {selected['detection_date']}
                    - **Attended by:** {selected['attended_by']}
                    """, unsafe_allow_html=True)
                
                st.markdown("#### Clinical Notes")
                st.write(selected['notes'] if selected['notes'] else "No notes available")
                    
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
                        st.rerun()
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
                            st.rerun()
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
                    st.rerun()
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
            st.rerun()
    
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
            st.rerun()
    
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
                st.rerun()
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
                        st.rerun()
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
                                st.rerun()
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
    """Show comprehensive admin management interface with all fixes"""
    if st.session_state.user_role != 'admin':
        st.warning("⛔ Admin access only")
        return
    
    # Add delete functionality JS
    st.components.v1.html("""
    <script>
    function deleteModel(modelId) {
        Streamlit.setComponentValue([modelId]);
    }
    </script>
    """)
    
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
    
    with tab1:
        _display_pending_approvals()
    
    with tab2:
        _display_user_management()
    
    with tab3:
        _display_model_management()
    
    with tab4:
        _display_system_settings()

# -------------------------------
# 🧑‍💼 User Management Functions
# -------------------------------
def _display_pending_approvals():
    """Display pending user approvals"""
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
                            st.rerun()
                with col2:
                    if st.button(f"❌ Reject {user['id']}", key=f"reject_{user['id']}"):
                        if reject_user(user['id'], st.session_state.user_id):
                            st.success("User rejected")
                            st.rerun()
    else:
        st.info("No pending registrations found")

def _display_add_user_form():
    """Display form to add new users"""
    with st.expander("➕ Add New User", expanded=False):
        with st.form("add_user_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Name*", key="new_user_name")
                email = st.text_input("Email*", key="new_user_email")
            with col2:
                role = st.selectbox(
                    "Role*",
                    ["admin", "doctor", "assistant"],
                    index=2,
                    key="new_user_role"
                )
                status = st.selectbox(
                    "Status*",
                    ["approved", "pending"],
                    index=0,
                    key="new_user_status"
                )
            
            temp_password = st.text_input(
                "Password1234*",
                value="".join(random.choices(string.ascii_letters + string.digits, k=12)),
                key="new_user_password"
            )
            
            if st.form_submit_button("Create User"):
                if not all([full_name, email, temp_password]):
                    st.error("Please fill all required fields (*)")
                else:
                    try:
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        cursor.execute(
                            """INSERT INTO users 
                            (full_name, email, password, role, status) 
                            VALUES (?, ?, ?, ?, ?)""",
                            (
                                full_name.strip(),
                                email.lower().strip(),
                                hash_password(temp_password),
                                role,
                                status
                            )
                        )
                        conn.commit()
                        st.success(f"User {email} created successfully!")
                        st.info(f"password1234: {temp_password}")
                        time.sleep(2)
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Email already exists")
                    except Exception as e:
                        st.error(f"Error creating user: {str(e)}")
                    finally:
                        if conn:
                            conn.close()

def _get_users_data():
    """Get users data from database, excluding current user"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        query = """
        SELECT id, full_name, email, role, status 
        FROM users 
        WHERE id != ?
        ORDER BY status DESC, full_name ASC
        """
        return pd.read_sql_query(query, conn, params=(st.session_state.user_id,))
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
            
def _display_user_management():
    """Display user management interface with immediate hard actions"""
    st.markdown('<div class="section-title">User Accounts</div>', unsafe_allow_html=True)

    users_df = _get_users_data()
    if users_df.empty:
        st.info("No users found in the database")
        _display_add_user_form()
        return

    # Configure AgGrid with selection
    gb = GridOptionsBuilder.from_dataframe(users_df)
    gb.configure_default_column(editable=False, filterable=True, sortable=True)
    gb.configure_selection('single', use_checkbox=True, pre_selected_rows=[])
    grid_options = gb.build()

    # Display the grid
    grid_response = AgGrid(
        users_df,
        gridOptions=grid_options,
        height=400,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        theme='streamlit',
        fit_columns_on_grid_load=True,
        key='users_grid'
    )

    # Get selected user (safe handling)
    selected_rows = grid_response['selected_rows']
    selected_user = None
    
    if isinstance(selected_rows, list) and len(selected_rows) > 0:
        selected_user = selected_rows[0]
    elif isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
        selected_user = selected_rows.iloc[0].to_dict()

    # Action buttons
    st.markdown("### User Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ Activate", disabled=not selected_user, key="activate_btn"):
            if selected_user:
                if _update_user_status(selected_user['id'], 'approved'):
                    st.success("User activated")
                    time.sleep(0.5)
                    st.rerun()  # Updated from experimental_rerun()

    with col2:
        if st.button("🚫 Deactivate", disabled=not selected_user, key="deactivate_btn"):
            if selected_user:
                if _update_user_status(selected_user['id'], 'suspended'):
                    st.success("User deactivated")
                    time.sleep(0.5)
                    st.rerun()

    with col3:
        if st.button("🔒 Reset Password", disabled=not selected_user, key="reset_btn"):
            if selected_user:
                new_pass = _reset_password(selected_user['id'])
                if new_pass:
                    st.success(f"Password reset to: {new_pass}")
                    time.sleep(0.5)
                    st.rerun()

    with col4:
        if st.button("🗑️ Delete", disabled=not selected_user, key="delete_btn"):
            if selected_user:
                if _delete_user(selected_user['id']):
                    st.success("User deleted")
                    time.sleep(0.5)
                    st.rerun()

    _display_add_user_form()

def _update_user_status(user_id, status):
    """Update user status and return success boolean"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        st.error(f"Error updating status: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def _reset_password(user_id):
    """Reset user password and return new password if successful"""
    new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password = ? WHERE id = ?",
            (hash_password(new_password), user_id))
        conn.commit()
        return new_password if cursor.rowcount > 0 else None
    except Exception as e:
        st.error(f"Error resetting password: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def _delete_user(user_id):
    """Delete user and return success boolean"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        st.error(f"Error deleting user: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()
# -------------------------------
# 🤖 Model Management Functions
# -------------------------------
def _display_model_management():
    """Display model management interface"""
    st.markdown("## Model Management")
    
    # Current active model
    current_model = get_active_model_info()
    if current_model:
        st.markdown(f"""
        **Active Model:**  
        Version: {current_model['version']}  
        Description: {current_model['description']}  
        Uploaded: {current_model['upload_date']} by {current_model['uploaded_by']}
        """)
    
    # Model upload form
    with st.expander("Upload New Model", expanded=False):
        with st.form("model_upload_form"):
            new_model = st.file_uploader("Model file (.h5)", type=["h5"])
            version = st.text_input("Version* (e.g., 1.0.0)")
            description = st.text_area("Description")
            release_notes = st.text_area("Release Notes")
            
            if st.form_submit_button("Deploy Model"):
                if new_model and version:
                    _handle_model_upload(new_model, version, description, release_notes)
                else:
                    st.warning("Please provide both model file and version number")
    
    # Model history with delete
    st.markdown("## Model Versions")
    models = _get_model_history()
    
    if not models.empty:
        # Add delete action column
        models['Delete'] = "Delete"
        
        gb = GridOptionsBuilder.from_dataframe(models)
        gb.configure_default_column(filterable=True, sortable=True)
        gb.configure_column("is_active", header_name="Active", width=100)
        gb.configure_column("Delete", 
                          cellRenderer="deleteButton",
                          width=120,
                          pinned="right")
        
        grid_options = gb.build()
        grid_options['components'] = {
            'deleteButton': {
                'template': '''
                    <button onclick="deleteModel(params.data.id)" 
                            style="color: white; background-color: #ff4b4b; border: none; border-radius: 3px; padding: 2px 8px;">
                        Delete
                    </button>
                '''
            }
        }
        
        grid_response = AgGrid(
            models,
            gridOptions=grid_options,
            height=400,
            width='100%',
            theme='streamlit',
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True
        )
        
        # Handle delete actions
        if 'clicks' in st.session_state and st.session_state.clicks:
            model_id = st.session_state.clicks[0]
            if _confirm_delete_model(model_id):
                if delete_model_version(model_id):
                    st.success("Model version deleted")
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("No model versions found")

def _handle_model_upload(model_file, version, description, release_notes):
    """Process model upload"""
    try:
        # Validate version
        if not re.match(r'^\d+\.\d+\.\d+$', version):
            raise ValueError("Use semantic versioning (e.g., 1.0.0)")
        
        # Create models directory
        models_dir = REPO_ROOT / MODELS_DIR
        models_dir.mkdir(exist_ok=True)
        
        # Save model file
        model_filename = f"model_v{version.replace('.', '_')}.h5"
        model_path = models_dir / model_filename
        
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())
        
        # Validate model
        try:
            test_model = load_model(model_path)
            input_shape = test_model.input_shape
            del test_model
        except Exception as e:
            os.remove(model_path)
            raise ValueError(f"Invalid model: {str(e)}")
        
        # Store in database
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_versions 
            (version, description, release_notes, path, uploaded_by, performance_metrics)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            version,
            description,
            release_notes,
            str(model_path.relative_to(REPO_ROOT)),
            st.session_state.user_id,
            json.dumps({"input_shape": str(input_shape)})
        ))
        
        # Set as active
        cursor.execute('''
            INSERT OR REPLACE INTO system_settings 
            (key, value) VALUES ('active_model', ?)
        ''', (str(model_path.relative_to(REPO_ROOT)),))
        
        conn.commit()
        st.success(f"Model v{version} deployed!")
        st.balloons()
        st.rerun()
        
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        if 'model_path' in locals() and model_path.exists():
            model_path.unlink()

def delete_model_version(model_id):
    """Hard delete a model version"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Get model path first
        cursor.execute("SELECT path FROM model_versions WHERE id = ?", (model_id,))
        model_path = cursor.fetchone()[0]
        abs_path = REPO_ROOT / model_path
        
        # Delete from database
        cursor.execute("DELETE FROM model_versions WHERE id = ?", (model_id,))
        
        # Delete file
        if abs_path.exists():
            abs_path.unlink()
        
        conn.commit()
        return True
        
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Delete failed: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

# -------------------------------
# ⚙️ System Settings Functions
# -------------------------------
def _display_system_settings():
    """Display system settings"""
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
            
            # Backup/Restore
            st.markdown("### Backup & Restore")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Create Backup", use_container_width=True):
                    _create_db_backup(conn)
            
            with col2:
                backup_files = _get_backup_files()
                selected_backup = st.selectbox("Select backup", backup_files)
                
                if selected_backup and st.button("🔄 Restore Backup", use_container_width=True):
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
        
        log_level = st.selectbox("Log level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        num_entries = st.slider("Entries to show", 10, 500, 100)
        
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
            st.error(f"Error loading logs: {str(e)}")
        finally:
            if conn:
                conn.close()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System Info
    with st.container():
        st.markdown("""
        <div class="system-card">
            <h3>System Information</h3>
        """, unsafe_allow_html=True)
        
        st.json(_get_system_info())
        
        st.markdown("### System Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared")
        
        with col2:
            if st.button("Restart App", type="secondary", use_container_width=True):
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

def _get_system_info():
    """Get system information without MODEL_PATH dependency"""
    active_model = get_active_model_info()
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
            "Active Model": active_model.get('path', 'None')
        },
        "Session": {
            "User": st.session_state.user_email,
            "Role": st.session_state.user_role,
            "Session Start": st.session_state.get("session_start", "N/A")
        }
    }

# -------------------------------
# 🛠️ Helper Functions
# -------------------------------
def get_active_model_info():
    """Get active model info from database"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT mv.version, mv.description, mv.path, 
                   datetime(mv.uploaded_at, 'localtime') as uploaded_at,
                   u.full_name as uploaded_by
            FROM model_versions mv
            JOIN users u ON mv.uploaded_by = u.id
            WHERE mv.path = (SELECT value FROM system_settings WHERE key = 'active_model')
            ORDER BY mv.uploaded_at DESC
            LIMIT 1
        ''')
        model = cursor.fetchone()
        
        if model:
            return {
                "version": model[0],
                "description": model[1],
                "path": model[2],
                "upload_date": model[3],
                "uploaded_by": model[4]
            }
        return {}
    except Exception as e:
        st.error(f"Error getting active model: {str(e)}")
        return {}
    finally:
        if conn:
            conn.close()

def _get_model_history():
    """Get model history from database"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        models = pd.read_sql_query('''
            SELECT 
                mv.id,
                mv.version, 
                mv.description,
                datetime(mv.uploaded_at, 'localtime') as uploaded_at,
                mv.path,
                (SELECT full_name FROM users WHERE id = mv.uploaded_by) as uploaded_by,
                CASE 
                    WHEN mv.path = (SELECT value FROM system_settings WHERE key = 'active_model') 
                    THEN '✅' 
                    ELSE '' 
                END as is_active
            FROM model_versions mv
            ORDER BY mv.uploaded_at DESC
        ''', conn)
        
        if not models.empty:
            models['full_path'] = models['path'].apply(
                lambda x: str(REPO_ROOT / x) if x and not os.path.isabs(x) else x
            )
            models['exists'] = models['full_path'].apply(
                lambda x: os.path.exists(x) if x else False
            )
        return models
        
    except Exception as e:
        st.error(f"Error loading model history: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def _confirm_delete_model(model_id):
    """Show delete confirmation dialog"""
    with st.container():
        st.warning("Permanently delete this model version?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Delete", type="primary"):
                return True
        with col2:
            if st.button("Cancel"):
                return False
    return False

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
                st.rerun()
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.session_state.clear()
            st.rerun()
        
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
        st.rerun()

if __name__ == "__main__":
    main()
