import sqlite3
import pandas as pd
import hashlib
from datetime import datetime

# --- 1. SECURITY UTILS ---
def make_hashes(password):
    """Encodes password for secure storage."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    """Verifies a password against its stored hash."""
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# --- 2. DATABASE INITIALIZATION ---
def init_db():
    """Initializes the SQLite database with required tables."""
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    
    # Table for manual and Google users
    c.execute('''CREATE TABLE IF NOT EXISTS userstable (
                    username TEXT PRIMARY KEY, 
                    password TEXT)''')
    
    # Table for property valuation logs
    c.execute('''CREATE TABLE IF NOT EXISTS historytable (
                    username TEXT, 
                    location TEXT, 
                    area REAL, 
                    price REAL, 
                    sentiment TEXT, 
                    timestamp TEXT)''')
    conn.commit()
    conn.close()

# --- 3. USER MANAGEMENT ---
def add_userdata(username, password):
    """Adds a new manual user with a hashed password."""
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', 
                  (username, make_hashes(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # User already exists
    finally:
        conn.close()

def add_google_userdata(email):
    """
    Registers a Google user if they don't exist.
    Uses a placeholder password as they authenticate via Google.
    """
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    # Check if user exists first to avoid overwriting manual accounts
    c.execute('SELECT * FROM userstable WHERE username = ?', (email,))
    if not c.fetchone():
        c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', 
                  (email, 'GOOGLE_AUTH_USER'))
        conn.commit()
    conn.close()

def login_user(username, password):
    """Validates login credentials for manual access."""
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username = ? AND password = ?', 
              (username, make_hashes(password)))
    data = c.fetchall()
    conn.close()
    return data

# --- 4. HISTORY TRACKING ---
def add_history(username, location, area, price, sentiment):
    """Logs a successful property valuation."""
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO historytable(username, location, area, price, sentiment, timestamp) 
                 VALUES (?,?,?,?,?,?)''', 
              (username, location, area, price, sentiment, timestamp))
    conn.commit()
    conn.close()

def view_user_history(username):
    """Retrieves all past valuations for a specific user as a DataFrame."""
    conn = sqlite3.connect('zameen_data.db')
    query = 'SELECT location, area, price, sentiment, timestamp FROM historytable WHERE username = ?'
    df = pd.read_sql(query, conn, params=(username,))
    conn.close()
    return df
