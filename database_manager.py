import sqlite3
import hashlib
import pandas as pd
from datetime import datetime

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def init_db():
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT PRIMARY KEY, password TEXT)')
    c.execute('''CREATE TABLE IF NOT EXISTS historytable(
                 username TEXT, timestamp TEXT, location TEXT, 
                 area REAL, price REAL, sentiment TEXT)''')
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', (username, make_hashes(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    c.execute('SELECT password FROM userstable WHERE username =?', (username,))
    data = c.fetchone()
    conn.close()
    if data:
        return check_hashes(password, data[0])
    return False

def add_history(username, location, area, price, sentiment):
    conn = sqlite3.connect('zameen_data.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    c.execute('INSERT INTO historytable VALUES (?,?,?,?,?,?)', (username, timestamp, location, area, price, sentiment))
    conn.commit()
    conn.close()

def view_user_history(username):
    conn = sqlite3.connect('zameen_data.db')
    df = pd.read_sql_query("SELECT timestamp, location, area, price, sentiment FROM historytable WHERE username=?", conn, params=(username,))
    conn.close()
    return df