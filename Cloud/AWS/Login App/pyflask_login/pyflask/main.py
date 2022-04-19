"""
Created on Mon 02 February 2022

@author: Nikhil Adlakha
"""

from flask import Flask, render_template, request, flash, redirect, url_for
from flask import send_from_directory
from rds_db import UserDetailsDataLayer
import os

app = Flask(__name__)
user_db = UserDetailsDataLayer()
keys = 'firstname, lastname, username, password, email'.split(', ')
FILE_DIRECTORY = './static'
count = 0
filenames = {}


# User Login Page
@app.route('/')
@app.route('/index')
def index():
    msg = request.args.get('msg')
    return render_template('/index.html', msg=msg)


# User Registration Page
@app.route('/register', methods=['get', 'post'])
def insert():
    if request.method == 'POST':
        msg = None
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        if request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            else:
                global filenames
                if username in filenames.keys():
                    if filenames[username] != '' and os.path.exists(os.path.join(FILE_DIRECTORY, filenames[username])):
                        os.remove(os.path.join(FILE_DIRECTORY, filenames[username]))
                file.save(os.path.join(FILE_DIRECTORY, file.filename))
                filenames[username] = file.filename
        try:
            user_db.insert_details(firstname, lastname, email, username, password)
            msg = 'User successfully Created!!!'
        except Exception as e:
            print(e)
            msg = str(e).strip().split(', ')[1].split(r'"')[1]
        return redirect(url_for("index", msg=msg))
    elif request.method == 'GET':
        return render_template('/register.html')


# API Call for Login
@app.route('/login', methods=['post'])
def login():
    username = request.form['username']
    password = request.form['password']
    details = user_db.get_user_details(username)
    if details is not None:
        details = {key: detail for key, detail in zip(keys, details)}
        if details['password'] != password:
            return redirect(url_for("index", msg="Password is incorrect !!!"))
        filename = filenames[username] if username in filenames else 'Sonnet.txt'
        return render_template('/home.html', username=details['username'], details=details, filename=filename,
                               count=count_words(filename))
    else:
        return render_template('/register.html', username=username, password=password)


@app.route("/get/<filename>", methods=['GET'])
def get_file(filename):
    if os.path.exists(os.path.join(FILE_DIRECTORY, filename)):
        return send_from_directory(FILE_DIRECTORY, filename, as_attachment=True)
    else:
        return f"{filename} Not Found", 404


def count_words(filename=None):
    if filename is None:
        filename = "Sonnet.txt"
    filepath = os.path.join(FILE_DIRECTORY, filename)
    global count
    count = 0
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.readlines()
        for line in content:
            count += len(line.split())
        return count
    else:
        raise Exception(f"'{filename}' file not found !!!")


if __name__ == "__main__":
    count_words()
    app.run(debug=True)
