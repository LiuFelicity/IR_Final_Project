# pip freeze | xargs pip uninstall -y
# pip install -r requirements.txt
# cd grep
# python page.py
# python activity.py
# python activity_html.py
# python content.py
# cd ..
find . -name "*.pkl" -delete
python method2/orginal.py -m item_cold_start
python method2/baseline.py -m baseline
