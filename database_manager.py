import sqlite3
import hashlib
import pandas as pd
from datetime import datetime

# --- SECURITY FUNCTIONS ---
def make_hashes(password):
    """Encrypts a plain text password using SHA-256."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    """Compares a plain text password with a stored hash."""
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# --- DATABASE CORE ---
def init_db():
    """Initializes the database and creates necessary tables."""
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    # User table stores both manual and Google users
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT PRIMARY KEY, password TEXT)')
    # History table tracks every valuation generated
    c.execute('''CREATE TABLE IF NOT EXISTS historytable(
                 username TEXT, 
                 timestamp TEXT, 
                 location TEXT, 
                 area REAL, 
                 price REAL, 
                 sentiment TEXT)''')
    conn.commit()
    conn.close()

# --- USER MANAGEMENT ---
def add_userdata(username, password):
    """Registers a new manual user with a hashed password."""
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', 
                  (username, make_hashes(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def add_google_userdata(username):
    """
    Registers a Google user automatically upon login.
    Uses 'GOOGLE_AUTH' as a placeholder to prevent manual login bypass.
    """
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    try:
        # INSERT OR IGNORE ensures we don't get errors for returning users
        c.execute('INSERT OR IGNORE INTO userstable(username, password) VALUES (?,?)', 
                  (username, 'GOOGLE_AUTH'))
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()

def login_user(username, password):
    """Validates credentials for manual login."""
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    c.execute('SELECT password FROM userstable WHERE username =?', (username,))
    data = c.fetchone()
    conn.close()
    if data:
        return check_hashes(password, data[0])
    return False

# --- HISTORY TRACKING ---
def add_history(username, location, area, price, sentiment):
    """Saves a property valuation record to the history table."""
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    c.execute('INSERT INTO historytable VALUES (?,?,?,?,?,?)', 
              (username, timestamp, location, area, price, sentiment))
    conn.commit()
    conn.close()

def view_user_history(username):
    """Retrieves all past valuations for a specific user as a DataFrame."""
    conn = sqlite3.connect('zameen_data.db')
    query = "SELECT timestamp, location, area, price, sentiment FROM historytable WHERE username=?"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df