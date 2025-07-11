# -------------------------------
# 📦 Package Imports
# -------------------------------
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
from datetime import date, datetime, timedelta, time
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
# Get the root directory of your project (where the Git repo is)
REPO_ROOT = Path(__file__).resolve().parent.parent  # Goes up two levels from App/ to reach repo root

# Model paths
MODEL_DIR = REPO_ROOT / "SAVED MODELS"
MODEL_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist
DEFAULT_MODEL = "MobileNetV2.h5"  # Your default model filename
MODEL_PATH = str(MODEL_DIR / DEFAULT_MODEL)  # Full path to default model

DB_NAME = "cataract_system.db"
CLASS_NAMES = ['conjunctival_growth', 'mild', 'normal', 'severe']
SESSION_TIMEOUT_MINUTES = 60  # Increased timeout to 1 hour

# -------------------------------
# 🔐 Enhanced Authentication Functions
# -------------------------------
def hash_password(password):
    """Hash password using SHA256 with salt for better security"""
    salt = "zambica_salt"  # In production, use a unique salt per user
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
                    CREATE TABLE IF NOT EXISTS deleted_appointments (
                        id INTEGER PRIMARY KEY,
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
                        status TEXT,
                        created_at TIMESTAMP,
                        deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        deleted_by TEXT
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
                    'admin@cataract.com', 
                    hash_password("admin.com"),
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
                cursor.execute("SELECT id, password FROM users WHERE email = 'zambica360@gmail.com'")
                admin = cursor.fetchone()
                
                if admin:
                    admin_id, current_password = admin
                    if not verify_password("admin.com", current_password):
                        cursor.execute('''
                            UPDATE users SET password = ? WHERE id = ?
                        ''', (hash_password("admin.com"), admin_id))
                        conn.commit()
                
                cursor.execute("INSERT INTO migrations (version) VALUES (3)")
                conn.commit()
                current_version = 3
            except Exception as e:
                conn.rollback()
                raise Exception(f"Migration 3 failed: {str(e)}")
        
        # Migration 4: Create model directory if it doesn't exist (v4)
        if current_version < 4:
            try:
                # Create model directory if it doesn't exist
                MODEL_DIR.mkdir(exist_ok=True)
                cursor.execute("INSERT INTO migrations (version) VALUES (4)")
                conn.commit()
                current_version = 4
            except Exception as e:
                conn.rollback()
                raise Exception(f"Migration 4 failed: {str(e)}")
        
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
    
# -------------------------------
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
def load_detection_model():
    """Load the cataract detection model from the active model in database"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Get the most recent active model from database
        cursor.execute('''
            SELECT path FROM model_versions 
            ORDER BY uploaded_at DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        
        if result:
            model_path = result[0]
            # Convert to absolute path if stored as relative
            if not os.path.isabs(model_path):
                model_path = str(REPO_ROOT / model_path)
            
            if os.path.exists(model_path):
                model = load_model(model_path)
                st.session_state.current_model = model_path
                return model
            else:
                st.warning(f"Model file not found at {model_path}. Using default model.")
        
        # Fallback to default model if no model in database or file missing
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            st.session_state.current_model = MODEL_PATH
            return model
        else:
            st.error("No valid model found in database or default location")
            return None
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Try to load default model as last resort
        try:
            if os.path.exists(MODEL_PATH):
                model = load_model(MODEL_PATH)
                st.session_state.current_model = MODEL_PATH
                return model
        except:
            pass
        return None
    finally:
        if conn:
            conn.close()

def get_active_model_info():
    """Get information about the currently active model"""
    active_model = st.session_state.get('current_model', MODEL_PATH)
    
    # Get model version info from database
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        
        # Find the model in database by path (either absolute or relative)
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
        ))
        
        model_info = cursor.fetchone()
        
        if model_info:
            return {
                "path": active_model,
                "version": model_info[0],
                "description": model_info[1],
                "upload_date": model_info[2],
                "uploaded_by": model_info[3]
            }
        
        # If not found in database, return basic info
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
    """
    Enhance image quality through multiple processing steps.
    Args:
        img_pil: PIL Image object
    Returns:
        PIL Image: Enhanced image or original if enhancement fails
    """
    try:
        # Convert to OpenCV format (BGR)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Resize to model input size
        img_cv = cv2.resize(img_cv, (224, 224))

        # Apply enhancement pipeline
        img_cv = cv2.detailEnhance(img_cv, sigma_s=10, sigma_r=0.15)
        img_cv = cv2.medianBlur(img_cv, 3)  # Reduce noise
        
        # Convert back to PIL format (RGB)
        enhanced_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        # Apply brightness, contrast, sharpness adjustments
        enhancer = ImageEnhance.Brightness(enhanced_pil)
        enhanced_pil = enhancer.enhance(1.2)

        enhancer = ImageEnhance.Contrast(enhanced_pil)
        enhanced_pil = enhancer.enhance(1.3)

        enhancer = ImageEnhance.Sharpness(enhanced_pil)
        enhanced_pil = enhancer.enhance(2.0)

        return enhanced_pil
    except Exception as e:
        st.warning(f"Image enhancement failed, using original: {str(e)}")
        return img_pil

def preprocess_image(img_pil):
    """
    Prepare image for model prediction with normalization.
    Args:
        img_pil: PIL Image object
    Returns:
        numpy array: Preprocessed image array or None if failed
    """
    try:
        enhanced_img = enhance_image_quality(img_pil)
        img = enhanced_img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize to [0,1] range
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing failed: {str(e)}")
        return None

def predict_image(img_path, model):
    """
    Make cataract prediction on an image.
    Args:
        img_path: Path to image file
        model: Loaded TensorFlow model
    Returns:
        tuple: (predicted_class, confidence) or (None, None) if failed
    """
    try:
        img = Image.open(img_path)
        img_array = preprocess_image(img)
        
        if img_array is None:
            return None, None

        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = np.max(predictions[0]) * 100
        
        return predicted_class, confidence
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

def save_detection(patient_id, result, confidence, attended_by, notes=""):
    """
    Save detection results to database with validation.
    Args:
        patient_id: ID of patient
        result: Detection result class
        confidence: Prediction confidence (0-100)
        attended_by: Staff member who performed detection
        notes: Optional additional notes
    Returns:
        int: Detection ID if successful, None otherwise
    """
    conn = None
    try:
        # Validate inputs
        if not all([patient_id, result, attended_by]):
            raise ValueError("Missing required fields")
            
        if not 0 <= confidence <= 100:
            raise ValueError("Confidence must be 0-100")

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections 
            (patient_id, result, confidence, attended_by, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            int(patient_id),
            str(result),
            float(confidence),
            str(attended_by),
            str(notes) if notes else None
        ))
        
        detection_id = cursor.lastrowid
        conn.commit()
        return detection_id
        
    except sqlite3.Error as e:
        st.error(f"Database error saving detection: {str(e)}")
        if conn:
            conn.rollback()
        return None
    except Exception as e:
        st.error(f"Error saving detection: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def get_detections(limit=None, patient_id=None):
    """
    Get detection records with optional filtering.
    Args:
        limit: Maximum number of records to return
        patient_id: Filter by specific patient
    Returns:
        DataFrame: Detection records with patient info
    """
    conn = None
    try:
        query = '''
            SELECT 
                d.id, d.result, d.confidence, 
                datetime(d.detection_date) as detection_date,
                d.attended_by, d.notes,
                p.full_name, p.gender, p.age
            FROM detections d
            JOIN patients p ON d.patient_id = p.id
        '''
        
        params = []
        if patient_id:
            query += " WHERE d.patient_id = ?"
            params.append(int(patient_id))
            
        query += " ORDER BY d.detection_date DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
            
        conn = sqlite3.connect(DB_NAME)
        return pd.read_sql_query(query, conn, params=params)
        
    except Exception as e:
        st.error(f"Error getting detections: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_detection_by_id(detection_id):
    """
    Get a specific detection record by ID.
    Args:
        detection_id: ID of detection record
    Returns:
        dict: Detection details or None if not found
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                d.*, 
                p.full_name,
                p.gender,
                p.age
            FROM detections d
            JOIN patients p ON d.patient_id = p.id
            WHERE d.id = ?
        ''', (int(detection_id),))
        
        row = cursor.fetchone()
        if row:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
        return None
        
    except Exception as e:
        st.error(f"Error getting detection: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def update_detection(detection_id, new_result=None, new_confidence=None, new_notes=None):
    """
    Update detection record with new values.
    Args:
        detection_id: ID of detection to update
        new_result: Updated result class (optional)
        new_confidence: Updated confidence (optional)
        new_notes: Updated notes (optional)
    Returns:
        bool: True if update successful, False otherwise
    """
    conn = None
    try:
        if not any([new_result, new_confidence, new_notes]):
            raise ValueError("No update values provided")

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Build dynamic update query
        updates = []
        params = []
        
        if new_result is not None:
            updates.append("result = ?")
            params.append(str(new_result))
            
        if new_confidence is not None:
            if not 0 <= new_confidence <= 100:
                raise ValueError("Confidence must be 0-100")
            updates.append("confidence = ?")
            params.append(float(new_confidence))
            
        if new_notes is not None:
            updates.append("notes = ?")
            params.append(str(new_notes) if new_notes else None)
        
        params.append(int(detection_id))
        
        query = f'''
            UPDATE detections 
            SET {', '.join(updates)}
            WHERE id = ?
        '''
        
        cursor.execute(query, params)
        conn.commit()
        
        return cursor.rowcount > 0
        
    except sqlite3.Error as e:
        st.error(f"Database error updating detection: {str(e)}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        st.error(f"Error updating detection: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def delete_detection(detection_id):
    """
    Delete a detection record.
    Args:
        detection_id: ID of detection to delete
    Returns:
        bool: True if deletion successful, False otherwise
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM detections 
            WHERE id = ?
        ''', (int(detection_id),))
        
        conn.commit()
        return cursor.rowcount > 0
        
    except sqlite3.Error as e:
        st.error(f"Database error deleting detection: {str(e)}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
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
    """Add new appointment with full patient details and validation"""
    conn = None
    try:
        # Validate required fields
        if not all([patient_id, patient_name, appointment_date, appointment_time, doctor_email]):
            st.error("Missing required appointment fields")
            return None

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Check if doctor exists
        if not get_doctor_by_email(doctor_email):
            st.error("Specified doctor not found")
            return None

        cursor.execute('''
            INSERT INTO appointments (
                patient_id, patient_name, gender, age, village, traditional_authority,
                district, marital_status, appointment_date, appointment_time, booked_by,
                doctor_email, notes, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, patient_name, gender, age, village, traditional_authority,
            district, marital_status, appointment_date, appointment_time.strftime("%H:%M") if isinstance(appointment_time, time) else appointment_time, 
            booked_by, doctor_email.lower(), notes, status
        ))
        
        appointment_id = cursor.lastrowid
        conn.commit()
        st.success(f"Appointment {appointment_id} created successfully")
        return appointment_id
    except sqlite3.IntegrityError as e:
        st.error(f"Database integrity error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error adding appointment: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def get_appointments(status_filter=None, doctor_filter=None, date_range=None):
    """Get appointments with optional filtering"""
    conn = None
    try:
        base_query = '''
            SELECT a.*, p.registration_date as patient_reg_date 
            FROM appointments a
            LEFT JOIN patients p ON a.patient_id = p.id
        '''
        conditions = []
        params = []
        
        if status_filter:
            conditions.append("a.status = ?")
            params.append(status_filter)
        if doctor_filter:
            conditions.append("a.doctor_email = ?")
            params.append(doctor_filter.lower())
        if date_range:
            start_date, end_date = date_range
            conditions.append("a.appointment_date BETWEEN ? AND ?")
            params.extend([start_date, end_date])
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        base_query += " ORDER BY a.appointment_date DESC, a.appointment_time DESC"
        
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(base_query, conn, params=params if params else None)
        return df
    except Exception as e:
        st.error(f"Error retrieving appointments: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_doctor_appointments(doctor_email, upcoming_only=True):
    """Get appointments for a specific doctor with enhanced filtering"""
    conn = None
    try:
        if not doctor_email:
            st.error("Doctor email required")
            return pd.DataFrame()

        query = '''
            SELECT * FROM appointments 
            WHERE doctor_email = ?
        '''
        params = [doctor_email.lower()]
        
        if upcoming_only:
            query += " AND appointment_date >= date('now')"
        
        query += " ORDER BY appointment_date ASC, appointment_time ASC"  # Chronological order for doctors
        
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        st.error(f"Error getting doctor appointments: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def update_appointment_status(appointment_id, status, doctor_email=None):
    """Update appointment status with comprehensive validation"""
    conn = None
    try:
        if not get_appointment_by_id(appointment_id):
            st.error(f"Appointment {appointment_id} not found")
            return False

        valid_statuses = ["Pending", "Confirmed", "Completed", "Cancelled"]
        if status not in valid_statuses:
            st.error(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
            return False

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        if doctor_email:
            if not get_doctor_by_email(doctor_email):
                st.error("Specified doctor not found")
                return False
                
            cursor.execute('''
                UPDATE appointments 
                SET status = ?,
                    doctor_email = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, doctor_email.lower(), appointment_id))
        else:
            cursor.execute('''
                UPDATE appointments 
                SET status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, appointment_id))
        
        conn.commit()
        
        if cursor.rowcount == 0:
            st.warning("No appointments were updated")
            return False
            
        return True
    except Exception as e:
        st.error(f"Error updating appointment status: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def reschedule_appointment(appointment_id, new_date, new_time, new_doctor_email=None):
    """Reschedule an appointment with full validation"""
    conn = None
    try:
        appointment = get_appointment_by_id(appointment_id)
        if not appointment:
            st.error(f"Appointment {appointment_id} not found")
            return False

        # Validate new date is not in the past
        if new_date < date.today():
            st.error("Cannot schedule appointments in the past")
            return False

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        update_data = {
            'appointment_date': new_date,
            'appointment_time': new_time.strftime("%H:%M") if isinstance(new_time, time) else new_time,
            'status': 'Pending',  # Reset status when rescheduling
            'updated_at': datetime.now()
        }
        
        if new_doctor_email:
            if not get_doctor_by_email(new_doctor_email):
                st.error("Specified doctor not found")
                return False
            update_data['doctor_email'] = new_doctor_email.lower()

        set_clause = ", ".join(f"{k} = ?" for k in update_data.keys())
        params = list(update_data.values()) + [appointment_id]
        
        cursor.execute(f'''
            UPDATE appointments 
            SET {set_clause}
            WHERE id = ?
        ''', params)
        
        conn.commit()
        
        if cursor.rowcount == 0:
            st.warning("No appointments were rescheduled")
            return False
            
        st.success(f"Appointment {appointment_id} rescheduled successfully")
        return True
    except Exception as e:
        st.error(f"Error rescheduling appointment: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_doctors(include_inactive=False):
    """Get doctors with optional inactive ones"""
    conn = None
    try:
        query = '''
            SELECT id, full_name, email, status
            FROM users 
            WHERE role = 'doctor'
        '''
        
        if not include_inactive:
            query += " AND status = 'approved'"
            
        query += " ORDER BY full_name"
        
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error getting doctors: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_doctor_by_email(email, return_dict=False):
    """Get doctor details by email with enhanced return options"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, full_name, email, status
            FROM users 
            WHERE email = ? AND role = 'doctor'
        ''', (email.lower(),))
        
        doctor = cursor.fetchone()
        
        if not doctor:
            return None
            
        if return_dict:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, doctor))
        return doctor
    except Exception as e:
        st.error(f"Error getting doctor by email: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def get_patient_appointments(patient_id, include_past=False):
    """Get appointments for a patient with date filtering"""
    conn = None
    try:
        query = '''
            SELECT a.*, d.full_name as doctor_name
            FROM appointments a
            LEFT JOIN users d ON a.doctor_email = d.email
            WHERE a.patient_id = ?
        '''
        
        params = [patient_id]
        
        if not include_past:
            query += " AND a.appointment_date >= date('now')"
        
        query += " ORDER BY a.appointment_date DESC, a.appointment_time DESC"
        
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        st.error(f"Error getting patient appointments: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def delete_appointment(appointment_id):
    """Delete appointment with confirmation and logging"""
    conn = None
    try:
        appointment = get_appointment_by_id(appointment_id, detailed=True)
        if not appointment:
            st.error(f"Appointment {appointment_id} not found")
            return False

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # First archive the appointment to deleted_appointments table
        cursor.execute('''
            INSERT INTO deleted_appointments (
                id, patient_id, patient_name, gender, age, village, traditional_authority,
                district, marital_status, appointment_date, appointment_time, booked_by,
                doctor_email, notes, status, created_at, deleted_by
            )
            SELECT 
                id, patient_id, patient_name, gender, age, village, traditional_authority,
                district, marital_status, appointment_date, appointment_time, booked_by,
                doctor_email, notes, status, created_at, ?
            FROM appointments 
            WHERE id = ?
        ''', (st.session_state.user_email, appointment_id))
        
        # Then delete from active appointments
        cursor.execute('DELETE FROM appointments WHERE id = ?', (appointment_id,))
        
        conn.commit()
        
        if cursor.rowcount > 0:
            st.success(f"Appointment {appointment_id} deleted successfully")
            return True
        else:
            st.warning("No appointments were deleted")
            return False
            
    except Exception as e:
        st.error(f"Error deleting appointment: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_appointment_by_id(appointment_id, detailed=False):
    """Get appointment by ID with detailed option"""
    conn = None
    try:
        query = '''
            SELECT a.*, p.registration_date as patient_reg_date
            FROM appointments a
            LEFT JOIN patients p ON a.patient_id = p.id
            WHERE a.id = ?
        ''' if detailed else 'SELECT * FROM appointments WHERE id = ?'
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(query, (appointment_id,))
        
        result = cursor.fetchone()
        if not result:
            return None
            
        if detailed:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, result))
        return result
    except Exception as e:
        st.error(f"Error getting appointment: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def get_upcoming_appointments(days=7, doctor_email=None):
    """Get upcoming appointments with doctor filtering"""
    conn = None
    try:
        query = '''
            SELECT a.*, p.full_name as patient_name, u.full_name as doctor_name
            FROM appointments a
            JOIN patients p ON a.patient_id = p.id
            JOIN users u ON a.doctor_email = u.email
            WHERE a.appointment_date BETWEEN date('now') AND date('now', ?)
        '''
        
        params = [f'+{days} days']
        
        if doctor_email:
            query += " AND a.doctor_email = ?"
            params.append(doctor_email.lower())
        
        query += " ORDER BY a.appointment_date, a.appointment_time"
        
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        st.error(f"Error getting upcoming appointments: {str(e)}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_appointment_stats(time_period='30 days'):
    """Get comprehensive appointment statistics"""
    conn = None
    try:
        stats = {}
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Basic counts
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) as cancelled
            FROM appointments
            WHERE appointment_date >= date('now', ?)
        ''', (f'-{time_period}',))
        
        counts = cursor.fetchone()
        stats['total'] = counts[0]
        stats['completed'] = counts[1]
        stats['pending'] = counts[2]
        stats['cancelled'] = counts[3]
        
        # Status distribution
        cursor.execute('''
            SELECT status, COUNT(*) as count
            FROM appointments
            WHERE appointment_date >= date('now', ?)
            GROUP BY status
        ''', (f'-{time_period}',))
        stats['status_distribution'] = dict(cursor.fetchall())
        
        # Doctor workload
        cursor.execute('''
            SELECT u.full_name, COUNT(*) as appointment_count
            FROM appointments a
            JOIN users u ON a.doctor_email = u.email
            WHERE a.appointment_date >= date('now', ?)
            GROUP BY u.full_name
            ORDER BY appointment_count DESC
        ''', (f'-{time_period}',))
        stats['doctor_workload'] = cursor.fetchall()
        
        # Daily trend
        daily_trend = pd.read_sql_query('''
            SELECT date(appointment_date) as day, 
                   COUNT(*) as count,
                   SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed
            FROM appointments
            WHERE appointment_date >= date('now', ?)
            GROUP BY day
            ORDER BY day
        ''', conn, params=(f'-{time_period}',))
        stats['daily_trend'] = daily_trend
        
        return stats
    except Exception as e:
        st.error(f"Error getting appointment stats: {str(e)}")
        return None
    finally:
        if conn:
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
    .hero {
        background: linear-gradient(135deg, #48bb78, #38a169);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        transition: all 0.5s ease;
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
    """Show the cataract detection interface with edit/delete functionality"""
    st.markdown('<h1 class="section-title">👁️ Cataract Detection</h1>', unsafe_allow_html=True)
    
    # Display active model information at the top
    model_info = get_active_model_info()
    with st.expander("ℹ️ Current Model Information", expanded=True):
        st.markdown(f"""
        - **Version:** {model_info['version']}
        - **Description:** {model_info['description']}
        - **Uploaded by:** {model_info['uploaded_by']}
        - **Path:** `{model_info['path']}`
        - **Accuracy:** {model_info.get('metrics', {}).get('accuracy', 'N/A')}
        """)
    
    # Tab interface for different functions
    tab1, tab2 = st.tabs(["New Detection", "Manage Detections"])
    
    with tab1:
        _show_new_detection_interface()
    
    with tab2:
        _show_manage_detections_interface()

def _show_new_detection_interface():
    """Show interface for new cataract detection"""
    # Backward-compatible camera toggle
    use_camera = st.checkbox("🎥 Use Camera", value=False)
    
    # Step 1: Select or register patient
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
    
    patient_id = _get_or_register_patient()
    
    # Step 2: Image capture/upload and prediction
    if patient_id:
        _process_eye_image(patient_id, use_camera)

def _get_or_register_patient():
    """Handle patient selection or registration"""
    subtab1, subtab2 = st.tabs(["Existing Patient", "New Patient"])
    patient_id = None
    
    with subtab1:
        patients = get_patients()
        if patients.empty:
            st.info("No patients found. Please register a new patient.")
        else:
            patient_options = patients.apply(
                lambda x: f"{x['full_name']} | {x['village']} | {x['district']} | Age: {x['age']}", 
                axis=1
            )
            selected_patient = st.selectbox(
                "Select Patient", 
                options=patient_options,
                key="patient_select"
            )
            patient_id = patients.iloc[patient_options.tolist().index(selected_patient)]['id']
    
    with subtab2:
        st.markdown("### Register New Patient")
        with st.form("patient_form", clear_on_submit=True):
            cols = st.columns(2)
            with cols[0]:
                full_name = st.text_input("Full Name*", key="patient_name")
                age = st.number_input("Age*", min_value=0, max_value=120, key="patient_age")
                village = st.text_input("Village*", key="patient_village")
            with cols[1]:
                gender = st.selectbox(
                    "Gender*", 
                    options=["Male", "Female", "Other"], 
                    key="patient_gender"
                )
                district = st.text_input("District*", key="patient_district")
                marital_status = st.selectbox(
                    "Marital Status", 
                    options=["Single", "Married", "Divorced", "Widowed"],
                    key="patient_marital"
                )
            
            if st.form_submit_button("Register Patient"):
                if not all([full_name, age, village, gender, district]):
                    st.error("Please fill all required fields (*)")
                else:
                    patient_id = add_patient(
                        full_name=full_name,
                        gender=gender,
                        age=age,
                        village=village,
                        traditional_authority="",  # Optional field
                        district=district,
                        marital_status=marital_status
                    )
                    if patient_id:
                        st.success(f"Patient {full_name} registered successfully!")
                        st.session_state.patient_id = patient_id
                        st.rerun()
    
    return patient_id or st.session_state.get('patient_id')

def _process_eye_image(patient_id, use_camera):
    """Handle image capture/upload and prediction"""
    st.markdown('<div class="section-title">Capture Eye Image</div>', unsafe_allow_html=True)
    
    img = None
    if use_camera:
        img = st.camera_input("Take an eye photo", key="camera_input")
    else:
        img = st.file_uploader(
            "Or upload an eye image...", 
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
    
    if img is not None:
        # Display the captured/uploaded image
        st.image(img, caption="Eye Image for Analysis", use_column_width=True)
        
        # Save to temp file for prediction
        temp_file = f"temp_eye_{patient_id}_{datetime.now().timestamp()}.jpg"
        try:
            with open(temp_file, "wb") as f:
                f.write(img.getbuffer() if use_camera else img.getvalue())
            
            # Load model and predict
            model = load_detection_model()
            if model and st.button("Analyze Image", key="analyze_btn"):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence = predict_image(temp_file, model)
                    
                    if predicted_class:
                        _display_prediction_results(
                            patient_id=patient_id,
                            predicted_class=predicted_class,
                            confidence=confidence
                        )
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

def _display_prediction_results(patient_id, predicted_class, confidence):
    """Display and save prediction results"""
    # Display results in a styled card
    st.markdown(f'''
    <div class="feature-card">
        <h3>Analysis Results</h3>
        <p><strong>Prediction:</strong> {predicted_class}</p>
        <p><strong>Confidence:</strong> {confidence:.2f}%</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Step 3: Save results
    st.markdown('<div class="section-title">Save Results</div>', unsafe_allow_html=True)
    
    with st.form("save_results_form"):
        notes = st.text_area(
            "Additional Notes",
            placeholder="Enter any additional observations...",
            key="detection_notes"
        )
        
        if st.form_submit_button("Save Detection Results"):
            detection_id = save_detection(
                patient_id=patient_id,
                result=predicted_class,
                confidence=confidence,
                attended_by=st.session_state.user_name,
                notes=notes
            )
            
            if detection_id:
                st.success("Detection results saved successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to save detection results")

def _show_manage_detections_interface():
    """Show interface for managing existing detections"""
    st.markdown('<div class="section-title">Manage Detection Results</div>', unsafe_allow_html=True)
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        patient_filter = st.text_input("Filter by Patient Name", key="patient_filter")
    with col2:
        result_filter = st.multiselect(
            "Filter by Result",
            options=CLASS_NAMES,
            default=CLASS_NAMES,
            key="result_filter"
        )
    
    # Get filtered detections
    detections = get_detections()
    if not detections.empty:
        if patient_filter:
            detections = detections[detections['full_name'].str.contains(patient_filter, case=False)]
        if result_filter:
            detections = detections[detections['result'].isin(result_filter)]
    
    if not detections.empty:
        # Configure editable grid
        gb = GridOptionsBuilder.from_dataframe(detections)
        gb.configure_default_column(
            editable=False,
            filterable=True,
            sortable=True,
            resizable=True
        )
        
        # Configure editable columns
        gb.configure_column("result", 
                           header_name="Result", 
                           editable=True,
                           cellEditor='agSelectCellEditor',
                           cellEditorParams={'values': CLASS_NAMES})
        
        gb.configure_column("confidence", 
                           header_name="Confidence", 
                           type=["numericColumn"],
                           editable=True,
                           valueFormatter="value.toFixed(2) + '%'")
        
        gb.configure_column("notes", 
                           header_name="Notes", 
                           editable=True)
        
        # Configure selection
        gb.configure_selection(
            selection_mode="multiple",
            use_checkbox=True,
            rowMultiSelectWithClick=True,
            suppressRowDeselection=False
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
            fit_columns_on_grid_load=True,
            theme='streamlit',
            allow_unsafe_jscode=True,
            key="detections_grid"
        )
        
        # Handle grid actions
        _handle_grid_actions(grid_response, detections)
    else:
        st.info("No detection records found matching filters")

def _handle_grid_actions(grid_response, original_data):
    """Handle actions on the detections grid with proper database updates"""
    selected_rows = grid_response.get('selected_rows', [])
    updated_data = grid_response.get('data', pd.DataFrame())
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Changes", key="save_changes"):
            if not updated_data.empty:
                success_count = 0
                failed_count = 0
                
                # Create a progress bar
                progress_bar = st.progress(0)
                total_rows = len(updated_data)
                
                for i, (_, row) in enumerate(updated_data.iterrows()):
                    try:
                        # Find the original row to compare changes
                        original_row = original_data[original_data['id'] == row['id']].iloc[0]
                        
                        # Check if any editable fields were changed
                        changes = {}
                        if row['result'] != original_row['result']:
                            changes['new_result'] = row['result']
                        if row['confidence'] != original_row['confidence']:
                            changes['new_confidence'] = row['confidence']
                        if row['notes'] != original_row['notes']:
                            changes['new_notes'] = row['notes']
                        
                        # Only update if there are changes
                        if changes:
                            if update_detection(
                                detection_id=row['id'],
                                new_result=changes.get('new_result'),
                                new_confidence=changes.get('new_confidence'),
                                new_notes=changes.get('new_notes')
                            ):
                                success_count += 1
                            else:
                                failed_count += 1
                    
                    except Exception as e:
                        st.error(f"Error updating record ID {row['id']}: {str(e)}")
                        failed_count += 1
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_rows)
                
                # Show results
                if success_count > 0:
                    st.success(f"Successfully updated {success_count} records")
                if failed_count > 0:
                    st.error(f"Failed to update {failed_count} records")
                
                # Rerun to refresh the data
                time.sleep(1)
                st.rerun()
            else:
                st.warning("No data to save")
    
    with col2:
        if st.button("🔄 Refresh Data", key="refresh_data"):
            st.rerun()
    
    with col3:
        if selected_rows and st.button("🗑️ Delete Selected", type="secondary", key="delete_selected"):
            with st.expander("⚠️ Confirm Deletion", expanded=True):
                st.warning(f"You are about to delete {len(selected_rows)} records. This action cannot be undone!")
                
                # Show summary of selected records
                for i, row in enumerate(selected_rows[:5]):  # Show first 5 as sample
                    st.write(f"{i+1}. {row['full_name']} - {row['result']} ({row['confidence']}%)")
                if len(selected_rows) > 5:
                    st.write(f"...and {len(selected_rows)-5} more")
                
                # Double confirmation
                if st.checkbox("I understand this will permanently delete the selected records", key="confirm_delete_check"):
                    if st.button("✅ CONFIRM DELETE", type="primary", key="final_confirm_delete"):
                        success_count = 0
                        failed_count = 0
                        progress_bar = st.progress(0)
                        
                        for i, row in enumerate(selected_rows):
                            try:
                                if delete_detection(row['id']):
                                    success_count += 1
                                else:
                                    failed_count += 1
                            except Exception as e:
                                st.error(f"Error deleting record ID {row['id']}: {str(e)}")
                                failed_count += 1
                            
                            progress_bar.progress((i + 1) / len(selected_rows))
                        
                        if success_count > 0:
                            st.success(f"Successfully deleted {success_count} records")
                        if failed_count > 0:
                            st.error(f"Failed to delete {failed_count} records")
                        
                        time.sleep(1)
                        st.rerun()
    
    # Show detailed view for single selection
    if len(selected_rows) == 1:
        _show_detection_details(selected_rows[0])
      
def _show_detection_details(detection):
    """Show detailed view of a single detection"""
    st.markdown("---")
    st.markdown("### Detailed View")
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**Patient:** {detection['full_name']}")
        st.markdown(f"**Gender:** {detection['gender']}")
        st.markdown(f"**Age:** {detection['age']}")
    with cols[1]:
        st.markdown(f"**Result:** {detection['result']}")
        st.markdown(f"**Confidence:** {detection['confidence']:.2f}%")
        st.markdown(f"**Date:** {detection['detection_date']}")
    
    st.markdown("**Notes:**")
    st.write(detection['notes'])
    
    # Show patient history
    st.markdown("### Patient History")
    try:
        history = get_detections(patient_id=detection['patient_id'])
        if not history.empty:
            st.dataframe(
                history[['detection_date', 'result', 'confidence', 'attended_by']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No additional records for this patient")
    except Exception as e:
        st.error(f"Error loading patient history: {str(e)}")

# -------------------------------
# 📅 Appointments Page 
# -------------------------------
def show_appointments_page():
    """Main appointments page with role-based views"""
    st.markdown('<h1 class="section-title">📅 Appointments Management</h1>', unsafe_allow_html=True)
    
    # Role-based view selection
    if st.session_state.user_role == 'doctor':
        display_doctor_appointments()
    else:
        display_admin_appointments()

def display_doctor_appointments():
    """View for doctors to manage their appointments with enhanced features"""
    st.markdown('<div class="section-title">My Appointments</div>', unsafe_allow_html=True)
    
    # Get doctor's upcoming appointments
    appointments = get_doctor_appointments(st.session_state.user_email, upcoming_only=True)
    
    if not appointments.empty:
        # Enhanced filtering
        col1, col2 = st.columns(2)
        with col1:
            status_options = ["All", "Pending", "Confirmed", "Completed", "Cancelled"]
            selected_status = st.selectbox("Filter by status:", status_options)
        with col2:
            date_options = ["All", "Today", "Tomorrow", "This Week", "Next 7 Days", "Next 30 Days"]
            selected_date = st.selectbox("Filter by date:", date_options)
        
        # Apply filters
        if selected_status != "All":
            appointments = appointments[appointments['status'] == selected_status]
        
        today = date.today()
        if selected_date != "All":
            if selected_date == "Today":
                appointments = appointments[appointments['appointment_date'] == today]
            elif selected_date == "Tomorrow":
                tomorrow = today + timedelta(days=1)
                appointments = appointments[appointments['appointment_date'] == tomorrow]
            elif selected_date == "This Week":
                start_date = today - timedelta(days=today.weekday())
                end_date = start_date + timedelta(days=6)
                appointments = appointments[
                    (appointments['appointment_date'] >= start_date) & 
                    (appointments['appointment_date'] <= end_date)
                ]
            elif selected_date == "Next 7 Days":
                end_date = today + timedelta(days=7)
                appointments = appointments[
                    (appointments['appointment_date'] >= today) & 
                    (appointments['appointment_date'] <= end_date)
                ]
            elif selected_date == "Next 30 Days":
                end_date = today + timedelta(days=30)
                appointments = appointments[
                    (appointments['appointment_date'] >= today) & 
                    (appointments['appointment_date'] <= end_date)
                ]
        
        # Display appointments with enhanced cards
        for _, appt in appointments.iterrows():
            with st.container():
                card = st.container()
                with card:
                    cols = st.columns([4, 1])
                    with cols[0]:
                        # Enhanced appointment card
                        st.markdown(f"""
                        **Patient:** {appt['patient_name']}  
                        **Age/Gender:** {appt['age']}/{appt['gender']}  
                        **Location:** {appt['village']}, {appt['district']}  
                        **Date/Time:** {appt['appointment_date']} {appt['appointment_time']}  
                        **Status:** <span class="status-{appt['status'].lower()}">{appt['status']}</span>  
                        **Notes:** {appt['notes']}
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        # Doctor actions
                        if appt['status'] == 'Pending':
                            if st.button("✅ Accept", key=f"accept_{appt['id']}"):
                                if update_appointment_status(appt['id'], 'Confirmed', st.session_state.user_email):
                                    st.success("Appointment confirmed!")
                                    time.sleep(1)
                                    st.rerun()
                            
                            if st.button("🔄 Reschedule", key=f"reschedule_{appt['id']}"):
                                show_reschedule_form(appt['id'])
                            
                            if st.button("❌ Decline", key=f"decline_{appt['id']}"):
                                if update_appointment_status(appt['id'], 'Cancelled'):
                                    st.success("Appointment declined")
                                    time.sleep(1)
                                    st.rerun()
                        
                        elif appt['status'] == 'Confirmed':
                            if st.button("✔️ Mark Complete", key=f"complete_{appt['id']}"):
                                if update_appointment_status(appt['id'], 'Completed'):
                                    st.success("Appointment marked complete!")
                                    time.sleep(1)
                                    st.rerun()
                    
                    # Add some visual separation
                    st.markdown("---")
    else:
        st.info("No upcoming appointments scheduled for you")

def display_admin_appointments():
    """Enhanced admin view for managing all appointments"""
    tab1, tab2, tab3 = st.tabs(["📝 New Appointment", "🛠️ Manage Appointments", "📊 Reports"])
    
    with tab1:
        st.markdown('<div class="section-title">Schedule New Appointment</div>', unsafe_allow_html=True)
        
        # Enhanced patient selection with search
        patients = get_patients()
        if patients.empty:
            st.warning("No patients available. Please register patients first.")
        else:
            patient_options = patients['full_name'] + " | " + patients['village'] + " | " + patients['district']
            selected_patient = st.selectbox(
                "Select Patient:", 
                options=patient_options,
                help="Search for patient by name or location"
            )
            patient_id = patients.iloc[patient_options.tolist().index(selected_patient)]['id']
            patient_data = patients[patients['id'] == patient_id].iloc[0]
            
            # Enhanced doctor selection with availability indicators
            doctors = get_doctors()
            if doctors.empty:
                st.warning("No doctors available. Please have an admin register doctors.")
            else:
                # Show doctor workload
                doctor_stats = get_appointment_stats(time_period='7 days')['doctor_workload']
                workload_map = {doc[0]: doc[1] for doc in doctor_stats}
                
                doctor_options = [
                    f"{doc['full_name']} ({doc['email']}) | 🏥 {workload_map.get(doc['full_name'], 0)} appts this week"
                    for _, doc in doctors.iterrows()
                ]
                
                selected_doctor = st.selectbox(
                    "Select Doctor:", 
                    options=doctor_options,
                    help="Doctor availability based on recent appointments"
                )
                doctor_email = doctors.iloc[doctor_options.index(selected_doctor)]['email']
                
                # Appointment details with validation
                col1, col2 = st.columns(2)
                with col1:
                    appointment_date = st.date_input(
                        "Date:", 
                        min_value=date.today(),
                        help="Appointments cannot be scheduled in the past"
                    )
                with col2:
                    appointment_time = st.time_input(
                        "Time:",
                        help="Select between 8:00 AM and 5:00 PM"
                    )
                
                notes = st.text_area(
                    "Notes:", 
                    placeholder="Additional information about the appointment",
                    max_chars=500
                )
                
                if st.button("Schedule Appointment", type="primary"):
                    # Validate time
                    if not (time(8, 0) <= appointment_time <= time(17, 0)):
                        st.error("Appointments must be between 8:00 AM and 5:00 PM")
                    else:
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
                            appointment_time=appointment_time,
                            booked_by=st.session_state.user_name,
                            doctor_email=doctor_email,
                            notes=notes
                        )
                        
                        if appointment_id:
                            st.success(f"Appointment scheduled successfully (ID: {appointment_id})")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
    
    with tab2:
        st.markdown('<div class="section-title">Manage Appointments</div>', unsafe_allow_html=True)
        
        # Enhanced filtering with date ranges
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status_filter = st.selectbox(
                "Status:",
                options=["All", "Pending", "Confirmed", "Completed", "Cancelled"],
                index=0
            )
        with col2:
            doctor_filter = st.selectbox(
                "Doctor:",
                options=["All"] + sorted(get_doctors()['email'].unique().tolist()),
                index=0
            )
        with col3:
            date_filter = st.selectbox(
                "Date Range:",
                options=["All", "Today", "Yesterday", "This Week", "Last Week", "This Month", "Custom"],
                index=0
            )
        with col4:
            if date_filter == "Custom":
                date_range = st.date_input(
                    "Select date range:",
                    value=(date.today(), date.today() + timedelta(days=7)),
                    min_value=date.today() - timedelta(days=365),
                    max_value=date.today() + timedelta(days=365)
                )
        
        # Get filtered appointments
        filter_params = {}
        if status_filter != "All":
            filter_params['status_filter'] = status_filter
        if doctor_filter != "All":
            filter_params['doctor_filter'] = doctor_filter
        
        if date_filter != "All":
            today = date.today()
            if date_filter == "Today":
                filter_params['date_range'] = (today, today)
            elif date_filter == "Yesterday":
                yesterday = today - timedelta(days=1)
                filter_params['date_range'] = (yesterday, yesterday)
            elif date_filter == "This Week":
                start = today - timedelta(days=today.weekday())
                end = start + timedelta(days=6)
                filter_params['date_range'] = (start, end)
            elif date_filter == "Last Week":
                start = today - timedelta(days=today.weekday() + 7)
                end = start + timedelta(days=6)
                filter_params['date_range'] = (start, end)
            elif date_filter == "This Month":
                start = date(today.year, today.month, 1)
                end = date(today.year, today.month + 1, 1) - timedelta(days=1)
                filter_params['date_range'] = (start, end)
            elif date_filter == "Custom" and len(date_range) == 2:
                filter_params['date_range'] = date_range
        
        appointments = get_appointments(**filter_params)
        
        if not appointments.empty:
            # Enhanced display with sorting
            sort_col, _ = st.columns([1, 3])
            with sort_col:
                sort_by = st.selectbox(
                    "Sort by:",
                    options=["Date (Newest)", "Date (Oldest)", "Patient Name", "Doctor", "Status"]
                )
            
            if sort_by == "Date (Newest)":
                appointments = appointments.sort_values(['appointment_date', 'appointment_time'], ascending=[False, False])
            elif sort_by == "Date (Oldest)":
                appointments = appointments.sort_values(['appointment_date', 'appointment_time'], ascending=[True, True])
            elif sort_by == "Patient Name":
                appointments = appointments.sort_values('patient_name')
            elif sort_by == "Doctor":
                appointments = appointments.sort_values('doctor_email')
            elif sort_by == "Status":
                appointments = appointments.sort_values('status')
            
            # Display with interactive elements
            st.dataframe(
                appointments[[
                    'id', 'patient_name', 'doctor_email', 'appointment_date', 
                    'appointment_time', 'status', 'notes'
                ]],
                use_container_width=True,
                hide_index=True
            )
            
            # Enhanced appointment actions
            selected_id = st.text_input(
                "Enter Appointment ID to manage:",
                help="Find ID in the table above"
            )
            
            if selected_id:
                try:
                    selected_id = int(selected_id)
                    appointment = get_appointment_by_id(selected_id, detailed=True)
                    
                    if appointment:
                        with st.expander(f"Manage Appointment {selected_id}", expanded=True):
                            st.markdown(f"""
                            **Patient:** {appointment['patient_name']}  
                            **Doctor:** {appointment['doctor_email']}  
                            **Scheduled:** {appointment['appointment_date']} {appointment['appointment_time']}  
                            **Current Status:** {appointment['status']}
                            """)
                            
                            cols = st.columns(3)
                            with cols[0]:
                                new_status = st.selectbox(
                                    "Update Status:",
                                    options=["Pending", "Confirmed", "Completed", "Cancelled"],
                                    index=["Pending", "Confirmed", "Completed", "Cancelled"].index(appointment['status'])
                                )
                                if st.button("Update Status"):
                                    if update_appointment_status(selected_id, new_status):
                                        st.success("Status updated!")
                                        time.sleep(1)
                                        st.rerun()
                            
                            with cols[1]:
                                if st.button("Reschedule Appointment"):
                                    show_reschedule_form(selected_id)
                            
                            with cols[2]:
                                if st.button("Delete Appointment", type="secondary"):
                                    if st.checkbox("Confirm permanent deletion"):
                                        if delete_appointment(selected_id):
                                            st.success("Appointment deleted!")
                                            time.sleep(1)
                                            st.rerun()
                    else:
                        st.error(f"No appointment found with ID {selected_id}")
                except ValueError:
                    st.error("Please enter a valid numeric appointment ID")
        else:
            st.info("No appointments match the selected filters")
    
    with tab3:
        st.markdown('<div class="section-title">Appointment Analytics</div>', unsafe_allow_html=True)
        
        # Time period selection
        time_period = st.selectbox(
            "Report Time Period:",
            options=["Last 7 Days", "Last 30 Days", "Last 90 Days", "This Year", "All Time"],
            index=1
        )
        
        period_map = {
            "Last 7 Days": "7 days",
            "Last 30 Days": "30 days",
            "Last 90 Days": "90 days",
            "This Year": f"{date.today().year}-01-01",
            "All Time": "1000 days"  # Effectively all time
        }
        
        stats = get_appointment_stats(period_map[time_period])
        
        if stats:
            # Key metrics
            st.markdown("### Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Appointments", stats['total'])
            with col2:
                st.metric("Completed", stats['completed'], 
                         delta=f"{stats['completed']/max(1, stats['total'])*100:.1f}%")
            with col3:
                st.metric("Pending", stats['pending'])
            with col4:
                st.metric("Cancelled", stats['cancelled'])
            
            # Visualizations
            st.markdown("### Status Distribution")
            status_df = pd.DataFrame.from_dict(stats['status_distribution'], orient='index', columns=['Count'])
            fig1 = px.pie(
                status_df, 
                values='Count', 
                names=status_df.index,
                title=f"Appointment Status Distribution ({time_period})",
                hole=0.3
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("### Daily Appointment Trend")
            if not stats['daily_trend'].empty:
                fig2 = px.line(
                    stats['daily_trend'], 
                    x='day', 
                    y=['count', 'completed'],
                    title=f"Daily Appointments ({time_period})",
                    labels={'value': 'Appointments', 'day': 'Date'},
                    hover_data={'day': '|%B %d, %Y'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("### Doctor Workload")
            doctor_df = pd.DataFrame(stats['doctor_workload'], columns=['Doctor', 'Appointments'])
            fig3 = px.bar(
                doctor_df,
                x='Doctor',
                y='Appointments',
                title=f"Doctor Workload ({time_period})",
                color='Appointments',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.error("Could not load appointment statistics")

def show_reschedule_form(appointment_id):
    """Enhanced reschedule form with validation"""
    appointment = get_appointment_by_id(appointment_id, detailed=True)
    if not appointment:
        st.error(f"Appointment ID {appointment_id} not found")
        return
    
    with st.form(f"reschedule_form_{appointment_id}", clear_on_submit=False):
        st.markdown(f"""
        ### Reschedule Appointment {appointment_id}
        **Patient:** {appointment['patient_name']}  
        **Current Date/Time:** {appointment['appointment_date']} {appointment['appointment_time']}  
        **Doctor:** {appointment['doctor_email']}
        """)
        
        # Doctor selection for reassignment
        doctors = get_doctors()
        current_doctor_idx = doctors[doctors['email'] == appointment['doctor_email']].index[0]
        new_doctor = st.selectbox(
            "Doctor (optional change):",
            options=doctors['full_name'] + " (" + doctors['email'] + ")",
            index=current_doctor_idx,
            help="Keep current doctor or select a different one"
        )
        new_doctor_email = doctors.iloc[doctors['full_name'].str.cat(doctors['email'], sep=" (").str.cat([")"]*len(doctors)).tolist().index(new_doctor)]['email']
        
        # Date/time selection
        col1, col2 = st.columns(2)
        with col1:
            new_date = st.date_input(
                "New Date:", 
                min_value=date.today(),
                value=appointment['appointment_date']
            )
        with col2:
            new_time = st.time_input(
                "New Time:",
                value=datetime.strptime(appointment['appointment_time'], "%H:%M").time()
            )
        
        # Confirmation
        if st.form_submit_button("Confirm Reschedule"):
            if new_date < date.today():
                st.error("Cannot schedule appointments in the past")
            elif not (time(8, 0) <= new_time <= time(17, 0)):
                st.error("Appointments must be between 8:00 AM and 5:00 PM")
            else:
                if reschedule_appointment(
                    appointment_id, 
                    new_date, 
                    new_time,
                    new_doctor_email if new_doctor_email != appointment['doctor_email'] else None
                ):
                    st.success("Appointment rescheduled successfully!")
                    time.sleep(1)
                    st.rerun()
        
        if st.form_submit_button("Cancel", type="secondary"):
            st.rerun()
            
# -------------------------------
# 📊 Analytics Page
# -------------------------------
def show_analytics_page():
    """Show analytics and statistics in a KPI dashboard format"""
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
        
        # Calculate KPIs with safe defaults for empty data
        total_patients = len(patients)
        total_detections = len(detections)
        positive_cases = len(detections[detections['result'] != 'normal']) if not detections.empty else 0
        avg_confidence = detections['confidence'].mean() if not detections.empty else 0
        upcoming_appointments = len(appointments[appointments['status'] == 'Pending']) if not appointments.empty else 0
        
        # Create a grid layout for KPIs (3 columns)
        st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)
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
                <div class="metric-title">Upcoming Appointments</div>
                <div class="metric-value">{upcoming_appointments}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Positive Cases</div>
                <div class="metric-value">{positive_cases}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Detection Rate</div>
                <div class="metric-value">{total_detections/max(total_patients,1):.1f}/patient</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Interactive filters
        st.markdown('<div class="section-title">Data Filters</div>', unsafe_allow_html=True)
        with st.expander("Filter Options", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_confidence = st.slider("Minimum Confidence", 0, 100, 70)
                result_filter = st.multiselect(
                    "Filter Results",
                    options=detections['result'].unique() if not detections.empty else [],
                    default=detections['result'].unique() if not detections.empty else []
                )
            
            with col2:
                age_range = st.slider(
                    "Age Range",
                    min_value=0,
                    max_value=100,
                    value=(0, 100)
                )
                gender_filter = st.multiselect(
                    "Filter Gender",
                    options=detections['gender'].unique() if not detections.empty else [],
                    default=detections['gender'].unique() if not detections.empty else []
                )
            
            with col3:
                district_filter = st.multiselect(
                    "Filter District",
                    options=detections['district'].unique() if not detections.empty else [],
                    default=detections['district'].unique() if not detections.empty else []
                )
                
                # Safe date range input handling
                if not detections.empty and 'detection_date' in detections.columns:
                    min_date = detections['detection_date'].min()
                    max_date = detections['detection_date'].max()
                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                else:
                    # Default to current date if no data
                    today = date.today()
                    date_range = st.date_input(
                        "Date Range",
                        value=(today, today)
                    )
        
        # Apply filters with safe handling
        if not detections.empty:
            filtered_data = detections[
                (detections['confidence'] >= min_confidence) &
                (detections['result'].isin(result_filter)) &
                (detections['age'] >= age_range[0]) &
                (detections['age'] <= age_range[1]) &
                (detections['gender'].isin(gender_filter)) &
                (detections['district'].isin(district_filter))
            ]
            
            # Apply date filter if we have valid dates
            if len(date_range) == 2 and 'detection_date' in detections.columns:
                filtered_data = filtered_data[
                    (filtered_data['detection_date'] >= date_range[0]) &
                    (filtered_data['detection_date'] <= date_range[1])
                ]
        else:
            filtered_data = pd.DataFrame()
        
        # Visualization Section (2 columns)
        st.markdown('<div class="section-title">Trends & Distributions</div>', unsafe_allow_html=True)
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Patient distribution by district
            if not filtered_data.empty:
                district_counts = filtered_data['district'].value_counts().reset_index()
                district_counts.columns = ['District', 'Count']
                fig = px.bar(district_counts, x='District', y='Count', 
                             title="Patient Distribution by District",
                             color='District',
                             height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for visualization")
            
            # Detection confidence distribution
            if not filtered_data.empty:
                fig = px.histogram(filtered_data, x='confidence', 
                                   title="Confidence Distribution",
                                   nbins=20,
                                   height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_col2:
            # Detection results breakdown
            if not filtered_data.empty:
                result_counts = filtered_data['result'].value_counts().reset_index()
                result_counts.columns = ['Result', 'Count']
                fig = px.pie(result_counts, values='Count', names='Result', 
                             title="Detection Results Breakdown",
                             height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Monthly detection trend
            if not filtered_data.empty and 'detection_date' in filtered_data.columns:
                filtered_data['month'] = pd.to_datetime(filtered_data['detection_date']).dt.to_period('M')
                monthly_counts = filtered_data.groupby('month').size().reset_index()
                monthly_counts.columns = ['Month', 'Count']
                monthly_counts['Month'] = monthly_counts['Month'].astype(str)
                fig = px.line(monthly_counts, x='Month', y='Count',
                              title="Monthly Detection Trend",
                              height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Data Tables
        st.markdown('<div class="section-title">Detailed Records</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Patients", "Detections", "Appointments"])
        
        with tab1:
            if not patients.empty:
                st.dataframe(patients, use_container_width=True)
            else:
                st.info("No patient data available")
        
        with tab2:
            if not filtered_data.empty:
                st.dataframe(filtered_data, use_container_width=True)
            else:
                st.info("No detection data available")
        
        with tab3:
            if not appointments.empty:
                st.dataframe(appointments, use_container_width=True)
            else:
                st.info("No appointment data available")
    except Exception as e:
        st.error(f"Error loading analytics data: {str(e)}")
    finally:
        conn.close()

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
                    st.rerun()
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
    """Display pending user approvals with error handling"""
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
                    <div class="action-buttons">
                        <button class="stButton" onclick="approveUser('{user['id']}')">✅ Approve</button>
                        <button class="stButton" onclick="rejectUser('{user['id']}')">❌ Reject</button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No pending registrations found")

def _display_user_management():
    """Display user management interface with robust selection handling"""
    st.markdown('<div class="section-title">User Accounts</div>', unsafe_allow_html=True)
    
    # Get user data with error handling
    try:
        users_df = _get_users_data()
        if users_df is None:
            st.error("Failed to load user data")
            return
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")
        return
    
    if not users_df.empty:
        # Configure AgGrid with safe defaults
        try:
            gb = GridOptionsBuilder.from_dataframe(users_df)
            gb.configure_default_column(
                editable=False,
                filterable=True,
                sortable=True,
                suppressMenu=True
            )
            gb.configure_column(
                "status",
                editable=True,
                cellEditor='agSelectCellEditor',
                cellEditorParams={
                    'values': ['approved', 'pending', 'rejected', 'suspended']
                },
                suppressMenu=True
            )
            gb.configure_selection(
                selection_mode="single",
                use_checkbox=True,
                suppressRowDeselection=False  # Allow unselecting
            )
            grid_options = gb.build()
            
            # Display AgGrid with error handling
            grid_response = AgGrid(
                users_df,
                gridOptions=grid_options,
                height=400,
                width='100%',
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                fit_columns_on_grid_load=True,
                theme='streamlit',
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False
            )
            
            # Robust selection handling
            selected_rows = grid_response.get('selected_rows', None)
            
            # Handle all possible selection states
            if selected_rows is None:
                has_selection = False
                selected_rows = pd.DataFrame()  # Empty DataFrame as fallback
            elif isinstance(selected_rows, pd.DataFrame):
                has_selection = not selected_rows.empty
            else:
                # Handle unexpected types (shouldn't happen but just in case)
                try:
                    selected_rows = pd.DataFrame(selected_rows)
                    has_selection = not selected_rows.empty
                except Exception:
                    has_selection = False
                    selected_rows = pd.DataFrame()

            # User action buttons with additional safety checks
            st.markdown("### User Actions")
            col1, col2, col3, col4 = st.columns(4)
            
            # Helper function for button actions
            def handle_user_action(action_func, user_id=None, user_data=None):
                try:
                    if user_id is not None:
                        action_func(user_id)
                    elif user_data is not None and 'id' in user_data:
                        action_func(user_data)
                    else:
                        st.error("Invalid user selection")
                except Exception as e:
                    st.error(f"Action failed: {str(e)}")
            
            with col1:
                if st.button("✅ Activate", 
                           disabled=not has_selection,
                           help="Activate selected user",
                           use_container_width=True):
                    if has_selection:
                        handle_user_action(
                            lambda user_id: _update_user_status(user_id, 'approved'),
                            user_id=selected_rows.iloc[0]['id']
                        )
            
            with col2:
                if st.button("🚫 Deactivate", 
                           disabled=not has_selection,
                           help="Deactivate selected user",
                           use_container_width=True,
                           type="secondary"):
                    if has_selection:
                        handle_user_action(
                            lambda user_id: _update_user_status(user_id, 'suspended'),
                            user_id=selected_rows.iloc[0]['id']
                        )
            
            with col3:
                if st.button("🔒 Reset Password", 
                           disabled=not has_selection,
                           help="Reset password for selected user",
                           use_container_width=True,
                           type="secondary"):
                    if has_selection:
                        handle_user_action(
                            _reset_password_form,
                            user_data=selected_rows.iloc[0]
                        )
            
            with col4:
                if st.button("🗑️ Delete", 
                           disabled=not has_selection,
                           help="Delete selected user",
                           use_container_width=True,
                           type="secondary"):
                    if has_selection:
                        handle_user_action(
                            _delete_user,
                            user_data=selected_rows.iloc[0]
                        )
            
            # Add new user form
            _display_add_user_form()
            
        except Exception as e:
            st.error(f"Error configuring user table: {str(e)}")
            _display_add_user_form()
    else:
        st.info("No users found in the database")
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
                        MODEL_DIR.mkdir(exist_ok=True)
                        
                        # Save the new model file
                        model_filename = f"model_v{version.replace('.', '_')}.h5"
                        model_path = str(MODEL_DIR / model_filename)
                        
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
                        st.rerun()
                        
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
                                        st.rerun()
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
                st.rerun()
        
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
    """Update user status in database"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute(
            "UPDATE users SET status = ? WHERE id = ?",
            (status, user_id)
        )
        conn.commit()
        st.success(f"User status updated to {status}!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Error updating user status: {str(e)}")
    finally:
        if conn:
            conn.close()

def _reset_password_form(user):
    """Display password reset form"""
    with st.form("reset_password_form"):
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.form_submit_button("Confirm Reset"):
            if new_password == confirm_password:
                conn = None
                try:
                    conn = sqlite3.connect(DB_NAME)
                    conn.execute(
                        "UPDATE users SET password = ? WHERE id = ?",
                        (hash_password(new_password), user['id'])
                    )
                    conn.commit()
                    st.success("Password reset successfully")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    if conn:
                        conn.rollback()
                    st.error(f"Error resetting password: {str(e)}")
                finally:
                    if conn:
                        conn.close()
            else:
                st.error("Passwords don't match")

def _delete_user(user):
    """Delete user from database"""
    if st.checkbox(f"Confirm deletion of {user['email']}", key="confirm_delete"):
        conn = None
        try:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("DELETE FROM users WHERE id = ?", (user['id'],))
            conn.commit()
            st.success("User deleted successfully")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            if conn:
                conn.rollback()
            st.error(f"Error deleting user: {str(e)}")
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
    temp_password = "Temp123!"  # In production, generate a random password
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

def _update_model(new_model, version, description, release_notes):
    """Update the AI model"""
    try:
        # Create backup directory if it doesn't exist
        os.makedirs("model_backups", exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"model_backups/model_{timestamp}.h5"
        
        # Backup current model if exists
        if os.path.exists(MODEL_PATH):
            shutil.copy2(MODEL_PATH, backup_path)
        
        # Save new model
        with open(MODEL_PATH, "wb") as f:
            f.write(new_model.getbuffer())
        
        # Update model version in database
        conn = None
        try:
            conn = sqlite3.connect(DB_NAME)
            conn.execute('''
                INSERT INTO model_versions 
                (version, description, release_notes, path, uploaded_by)
                VALUES (?, ?, ?, ?, ?)
            ''', (version, description, release_notes, MODEL_PATH, st.session_state.user_id))
            conn.commit()
            
            st.success(f"""
            Model updated successfully to version {version}!
            Backup saved to: {backup_path}
            """)
            time.sleep(1)
            st.rerun()
        except Exception as e:
            if conn:
                conn.rollback()
            # Restore backup if DB update failed
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, MODEL_PATH)
            st.error(f"Database error: {str(e)}")
        finally:
            if conn:
                conn.close()
    except Exception as e:
        st.error(f"Model update failed: {str(e)}")

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
            st.rerun()
        except Exception as e:
            st.error(f"Restore failed: {str(e)}")

def _clear_logs(conn):
    """Clear system logs"""
    try:
        conn.execute("DELETE FROM system_logs")
        conn.commit()
        st.success("Logs cleared")
        time.sleep(1)
        st.rerun()
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
