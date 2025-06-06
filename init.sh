pip freeze | xargs pip uninstall -y
pip install -r requirements.txt

cd grep
python page.py
python activity.py
python activity_html.py
python content.py

cd ..
python grep/fetch_missing_activities.py

cd grep
python content.py

cd ..
python method2/cold_start_model.py
python method2/baseline.py
