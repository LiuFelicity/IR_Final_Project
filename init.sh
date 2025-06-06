find . -name "*.npz" -delete
find . -name "*.pkl" -delete
rm -rf grep/page_data
rm -rf grep/activity_data
rm -rf grep/activity_data_text

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
