# IR_Final_Project

This project is a web scraping and content extraction pipeline for collecting and processing opportunity listings from [opportunitiescircle.com](https://www.opportunitiescircle.com/).

## Directory Structure

```
IR_Final_Project/
├── requirements.txt           # Python dependencies
└── grep/
    ├── page.py                # Scrapes paginated listing pages and saves HTML
    ├── activity.py            # Extracts opportunity links from listing HTML files
    ├── activity_html.py       # Downloads each opportunity's detail page HTML
    ├── content.py             # Extracts and saves main text content from each detail HTML
    ├── activity_data/         # Downloaded HTML files for each opportunity
    ├── activity_data_text/    # Extracted text content for each opportunity
    └── page_data/             # Downloaded HTML files for each listing page
```

## Usage

> **Note:** All Python scripts should be executed from within the `grep` folder. Use `cd grep` before running the commands below.

1. **Install dependencies**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Step 1: Download listing pages**
   - Run `page.py` to download paginated opportunity listings (pages 1–25) into `page_data/`.
   ```bash
   python page.py
   ```

3. **Step 2: Extract opportunity links**
   - Run `activity.py` to parse each listing page and collect unique opportunity links into `activity_data/opportunity_links.txt`.
   ```bash
   python activity.py
   ```

4. **Step 3: Download opportunity detail pages**
   - Run `activity_html.py` to download each opportunity's detail HTML into `activity_data/`.
   ```bash
   python activity_html.py
   ```

5. **Step 4: Extract main content**
   - Run `content.py` to extract and save the main text content from each HTML file into `activity_data_text/`.
   ```bash
   python content.py
   ```

## Notes
- All scripts are written in Python and use `requests` and `BeautifulSoup` for web scraping and parsing.
- The pipeline is modular; you can run each step independently.
- The output folders will be created automatically if they do not exist.

## License
This project is for educational and research purposes only.
