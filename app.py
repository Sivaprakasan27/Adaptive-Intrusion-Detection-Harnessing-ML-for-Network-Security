import numpy as np
from flask import Flask, render_template, request, redirect,session, url_for, Response
import secrets
import sqlite3
import cv2
import joblib
import os
app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/home')
def home():
    return render_template('index.html')
app.config['SECRET_KEY'] = secrets.token_hex(16)
# Function to connect to SQLite database
def connect_db():
    conn = sqlite3.connect('users.db')
    return conn

# Function to create users table
def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            mobile TEXT NOT NULL,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            address TEXT
        )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS animal_detect (
        id INTEGER PRIMARY KEY,
        date TEXT,
        time TEXT,
        animal TEXT,
        username TEXT
    )
    ''')
   
    conn.commit()
    conn.close()

# Function to insert a new user into the database
def insert_user(name, email, mobile, username, password, address):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (name, email, mobile, username, password, address) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, email, mobile, username, password, address))
    conn.commit()
    conn.close()
from datetime import datetime

# Function to authenticate user login
def authenticate_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE username = ? AND password = ?
    ''', (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

def user_exists(username):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE username = ? 
    ''', (username,))
    user = cursor.fetchall()
    conn.close()
    return user


@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        username = request.form['username']
        password = request.form['password']
        address = request.form['address']

        # Check if user already exists
        if user_exists(username):
            error_message = 'Username already exists. Please choose a different username.'
            return render_template('register.html', error_message=error_message)

        # Insert user into the database
        insert_user(name, email, mobile, username, password, address)
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['login_username']
        password = request.form['login_password']

        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user[0]
            session['name'] = username
            return redirect('home')
        else:
            error_message = 'Invalid username or password.'
            return render_template('login.html', error_message=error_message)
    return render_template('login.html')

def logout():
    session.pop('user_id', None)
    session.pop('name', None)
    return redirect('/')
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        first_name = request.form.get("login_username")
        last_name = request.form.get("login_password") 
        if first_name == 'admin' and last_name == 'admin':
            return redirect("/users")
    else:
        return render_template("admin.html")
    return render_template("admin.html")
@app.route('/users')
def users():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, email, mobile, username, password, address FROM users')
    users = cursor.fetchall()
    return render_template('users.html', users=users)
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]

    if int_features[0]==0:
        f_features=[0,0,0]+int_features[1:]
    elif int_features[0]==1:
        f_features=[1,0,0]+int_features[1:]
    elif int_features[0]==2:
        f_features=[0,1,0]+int_features[1:]
    else:
        f_features=[0,0,1]+int_features[1:]

    if f_features[6]==0:
        fn_features=f_features[:6]+[0,0]+f_features[7:]
    elif f_features[6]==1:
        fn_features=f_features[:6]+[1,0]+f_features[7:]
    else:
        fn_features=f_features[:6]+[0,1]+f_features[7:]

    final_features = [np.array(fn_features)]
    predict = model.predict(final_features)

    if predict==0:
        output='Normal'
    elif predict==1:
        output='DOS'
    elif predict==2:
        output='PROBE'
    elif predict==3:
        output='R2L'
    else:
        output='U2R'

    return render_template('result.html', output=output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    predict = model.predict([np.array(list(data.values()))])

    if predict==0:
        output='Normal'
    elif predict==1:
        output='DOS'
    elif predict==2:
        output='PROBE'
    elif predict==3:
        output='R2L'
    else:
        output='U2R'

    return jsonify(output)

if __name__ == "__main__":
    create_table()
    app.run()
