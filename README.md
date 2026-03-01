# Tourism Experience Analytics

End-to-end analytics and ML project for tourism experience data, including:
- Data preprocessing and quality checks
- EDA with exported charts
- Regression model (rating prediction)
- Classification model (visit mode prediction)
- Recommendation system (collaborative filtering)
- Streamlit app for interactive exploration

## Repository Highlights
- Reproducible pipeline: one command runs full workflow
- Report artifacts generated automatically
- Raw dataset intentionally excluded from git for size/privacy safety

## Project Structure
```text
Tourism_Experience_Analytics/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── preprocess.py
│   ├── eda.py
│   ├── train.py
│   ├── generate_report.py
│   ├── generate_detailed_report.py
│   └── build_pdf_report.py
├── run_pipeline.py
└── requirements.txt
```

## Dataset Setup (Required)
Expected raw file path:

`data/raw/master_data.csv`

Steps:
1. Obtain `master_data.csv` from your assignment/source data.
2. Place it in `data/raw/`.
3. Keep filename exactly `master_data.csv`.

Reference: `data/README.md`

## Installation
```bash
python3 -m pip install -r requirements.txt
```

## Run Full Pipeline
From repository root:

```bash
python3 run_pipeline.py
```

## Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Generated Outputs
- Clean dataset: `data/processed/cleaned_tourism.csv`
- Trained models: `models/*.pkl`
- Metrics summary: `reports/metrics.json`
- Main report: `reports/PROJECT_REPORT.md`
- Detailed report: `reports/REPORT_DETAILED.md`
- Final PDF report: `reports/TOURISM_EXPERIENCE_ANALYTICS_FINAL_REPORT.pdf`
- Presentation script: `reports/VIDEO_PRESENTATION_SCRIPT.md`
- EDA/model figures: `reports/figures/*.png`

## Notes for GitHub Upload
- Raw files are ignored via `.gitignore` (`data/raw/*.csv`, etc.).
- Upload this folder as-is; do not include dataset files.
- Anyone cloning can run by adding `data/raw/master_data.csv`.

## Author
Sanjay
