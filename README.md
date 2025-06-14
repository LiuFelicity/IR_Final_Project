# IR_Final_Project

This project is a comprehensive web scraping and content extraction pipeline designed to collect and process opportunity listings from [opportunitiescircle.com](https://www.opportunitiescircle.com/). It features a recommendation system based on Latent Semantic Indexing (LSI) and Bayesian Personalized Ranking (BPR), which is accessible through a graphical user interface (GUI).

The pipeline automates the process of fetching, extracting, and analyzing data, making it easier to discover and recommend opportunities tailored to user preferences. The GUI allows users to interact with the system by creating profiles, receiving recommendations, and providing feedback through ratings. These ratings are then used to improve the recommendation model.

## Directory Structure

```
IR_Final_Project/
├── requirements.txt                # Python dependencies
├── README.md                       # This readme file
├── init.sh                         # Fetch data and train model
├── evaluation.sh                   # Evaluate Model
├── launch_app.sh                   # Run the app
├── grep/
│   ├── page.py                     # Scrapes paginated listing pages and saves HTML
│   ├── activity.py                 # Extracts opportunity links from listing HTML files
│   ├── activity_html.py            # Downloads each opportunity's detail page HTML
│   ├── content.py                  # Extracts and saves main text content from each detail HTML
│   ├── fetch_missing_activities.py # Fetch all missing item from train data
│   ├── activity_data/              # Downloaded HTML files for each opportunity
│   ├── activity_data_text/         # Extracted text content for each opportunity
│   └── page_data/                  # Downloaded HTML files for each listing page
├── gui/
│   ├── app.py                      # Flask application for the GUI
│   └── templates/                  # HTML templates for the GUI
│       ├── index.html
│       ├── profile.html
│       ├── recommendations.html
│       └── thank_you.html
├── method2/                        # Scripts and files to train and run the models
│   ├── cold_start_model.py
│   ├── baseline.py
│   └── *.pkl                       # Trained model file
└── evaluation/                     
    └── *.py                        # Scripts to evaluate the models
```

## Usage
1. Build a Python environment.
2. Run the following command to install the required packages, fetch data and train models:
```bash
./init.sh
```
3. Run the Flask application using the following command and open your web browser and navigate to `http://127.0.0.1:5001` (or `http://localhost:5001`).:
```bash
./launch_app.sh
```

The GUI allows new users to:
* Register with their name.
* Create a profile by specifying their age group and department.
* Receive and rate a selection of 5 activities recommended by our model.

User profiles and their ratings are stored in `method2/user_ratings.jsonl`.

4. You can use the following command to evaluate the model:
```bash
./evaluation.sh
```
