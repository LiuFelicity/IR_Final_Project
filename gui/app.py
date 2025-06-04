from flask import Flask, render_template, request, redirect, url_for
import os
import random
import json
import numpy as np
import sys

# Add the parent directory to the Python path to ensure the method2 module can be imported.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from method2.orginal import User
from constants import AGES_OPTIONS, DEPARTMENTS_OPTIONS
# Define lists directly in app.py based on gen_user_profile.py


app = Flask(__name__) # templates are now in ./templates relative to this app.py

# Data file paths - now relative to the new app.py location (gui/)
# Assuming doc_data_lsi.npz and activity_data_text/ are in the parent 'grep' directory
# Adjust these paths if the data files are located elsewhere relative to the new gui/app.py
PARENT_DIR = os.path.dirname(os.path.dirname(__file__)) # Goes up to IR_Final_Project/
GREP_DIR = os.path.join(PARENT_DIR, 'grep')

ACTIVITIES_FILE = os.path.join(GREP_DIR, 'doc_data_lsi.npz')
ACTIVITY_TEXT_DIR = os.path.join(GREP_DIR, 'activity_data_text')
USERS_FILE = 'users.json' # This is now gui/users.json, local to app.py

def load_activities():
    """Loads activity file names from the .npz file."""
    try:
        data = np.load(ACTIVITIES_FILE, allow_pickle=True)
        return list(data['file_names'])
    except FileNotFoundError:
        print(f"Error: Activities file not found at {ACTIVITIES_FILE}. Please ensure the path is correct.")
        return []
    except Exception as e:
        print(f"Error loading activities: {e}")
        return []

all_activity_filenames = load_activities()

def get_activity_details(filename):
    """Gets activity title, one line of content, and link."""
    txt_path = os.path.join(ACTIVITY_TEXT_DIR, filename)
    title = "Error: Title not found (" + filename + ")" # Default title if file processing fails
    content_line = "Error: Content not found"
    link = "#" 

    try:
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    title = lines[0].strip()
                    # Find the first non-empty line for content after the title
                    content_line = "No additional content available." # Default if no suitable line found
                    for i in range(1, len(lines)):
                        line_content = lines[i].strip()
                        if line_content: # If the stripped line is not empty
                            content_line = line_content[:150] + "..." # Increased length slightly
                            break # Found the first non-empty line
                else:
                    title = filename # Fallback if file is empty
                    content_line = "Content file is empty."
        else:
            print(f"Warning: Activity text file not found: {txt_path}")
            title = filename # Fallback if file not found
            content_line = "Content file not found."
    except Exception as e:
        print(f"Error reading activity file {filename}: {e}")
        title = filename # Fallback on error
        content_line = f"Error reading content: {e}"

    base_url = "https://www.opportunitiescircle.com/"
    activity_slug = filename.replace('.txt', '')
    link = f"{base_url}{activity_slug}"

    return {'title': title, 'content': content_line, 'link': link, 'id': filename}

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {USERS_FILE} is empty or corrupted. Starting with an empty user list.")
                return {}
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_name = request.form['name']
        users = load_users()
        if user_name in users:
            return redirect(url_for('recommendations', user_name=user_name, existing_user='yes'))
        else:
            return redirect(url_for('create_profile', user_name=user_name))
    return render_template('index.html')

@app.route('/create_profile/<user_name>', methods=['GET', 'POST'])
def create_profile(user_name):
    if request.method == 'POST':
        users = load_users()
        if user_name not in users:
            users[user_name] = {
                'name': user_name,
                'age': request.form['age'],
                'department': request.form['department'],
                'ratings': {}
            }
            save_users(users)
        return redirect(url_for('recommendations', user_name=user_name, existing_user='no'))

    return render_template('profile.html',
                           user_name=user_name,
                           age_options=AGES_OPTIONS,
                           department_options=DEPARTMENTS_OPTIONS)

@app.route('/recommendations/<user_name>', methods=['GET', 'POST'])
def recommendations(user_name):
    users = load_users()
    if user_name not in users:
        return redirect(url_for('index'))

    is_existing_user = request.args.get('existing_user', 'no').lower() == 'yes'

    if request.method == 'POST':
        current_user_ratings = users[user_name].get('ratings', {})
        for key, value in request.form.items():
            if key.startswith('rating_'):
                activity_id = key.replace('rating_', '')
                try:
                    current_user_ratings[activity_id] = int(value)
                except ValueError:
                    print(f"Warning: Invalid rating value for {activity_id}: {value}")
        users[user_name]['ratings'] = current_user_ratings
        save_users(users)

        # Update the dataset after user submits their ratings
        user_instance = User(name=user_name)
        user_instance.update_user_scores(user_name, current_user_ratings)

        return redirect(url_for('thank_you', user_name=user_name))

    if not all_activity_filenames:
        return "Error: No activities loaded. Please check activity data files (e.g., grep/doc_data_lsi.npz).", 500

    # Obtain recommendations for the user
    user_instance = User(name=user_name)
    recommended_items = user_instance.recommend(user_name=user_name, top_k=10)
    print(f"Recommended items for {user_name}: {recommended_items}")
    recommended_activities = [get_activity_details(item+".txt") for item in recommended_items]
    user_ratings = users[user_name].get('ratings', {})

    return render_template('recommendations.html',
                           user_name=user_name,
                           activities=recommended_activities,
                           existing_user=is_existing_user,
                           user_ratings=user_ratings)

@app.route('/thank_you/<user_name>')
def thank_you(user_name):
    return render_template('thank_you.html', user_name=user_name)

if __name__ == '__main__':
    # templates_dir is now ./templates relative to this app.py in gui/
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"Created templates directory at: {templates_dir}")

    # Ensure activity text directory (expected in ../grep/activity_data_text/) exists
    # This path is relative to the new app.py location in gui/
    activity_text_full_path = ACTIVITY_TEXT_DIR 
    if not os.path.exists(activity_text_full_path):
        # This might be an issue if it's expected to be pre-populated by other scripts.
        # For the GUI to run, it needs this data.
        print(f"Warning: Activity text directory not found at: {activity_text_full_path}. GUI might not show full content.")
        # os.makedirs(activity_text_full_path) # Avoid creating it if it should be managed by other scripts

    basic_templates_content = {
        "index.html": '''
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Recommender</title></head><body>
<h1>Welcome!</h1><form method="post"><label for="name">Name:</label><input type="text" id="name" name="name" required><button type="submit">Submit</button></form>
</body></html>''',
        "profile.html": '''
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Profile</title></head><body>
<h1>Profile for {{ user_name }}</h1><form method="post">
<label for="age">Age:</label><select name="age" required>{% for o in age_options %}<option value="{{o}}">{{o}}</option>{% endfor %}</select><br>
<label for="department">Dept:</label><select name="department" required>{% for o in department_options %}<option value="{{o}}">{{o}}</option>{% endfor %}</select><br>
<button type="submit">Save</button></form></body></html>''',
        "recommendations.html": '''
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Rate</title></head><body>
<h1>Rate for {{ user_name }}</h1><form method="post">
{% for act in activities %}<div><h2>{{act.title}}</h2><p>{{act.content}}</p><a href="{{act.link}}" target="_blank">Link</a><br>
Rating: <input type="number" name="rating_{{act.id}}" min="1" max="5" value="{{ user_ratings.get(act.id, '') }}" required></div><hr>{% endfor %}
<button type="submit">Submit Ratings</button></form>
<a href="{{ url_for('index') }}">Home</a></body></html>''',
        "thank_you.html": '''
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Thanks</title></head><body>
<h1>Thanks, {{ user_name }}!</h1><p>Ratings saved.</p><a href="{{ url_for('index') }}">Home</a></body></html>'''
    }

    for f_name, f_content in basic_templates_content.items():
        f_path = os.path.join(templates_dir, f_name)
        if not os.path.exists(f_path):
             with open(f_path, 'w', encoding='utf-8') as f:
                f.write(f_content)
                print(f"Created basic template: {f_path}")

    # Check if essential data files exist before running
    if not os.path.exists(ACTIVITIES_FILE):
        print(f"CRITICAL ERROR: ACTIVITIES_FILE not found at {ACTIVITIES_FILE}. The application cannot run without it.")
        print("Please ensure 'doc_data_lsi.npz' is in the 'grep' directory.")
    elif not all_activity_filenames:
        print(f"WARNING: No activities loaded from {ACTIVITIES_FILE}. Recommendations will be empty.")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
