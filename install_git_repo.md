# data_analysis_utils

Reusable Python package for `AnalyzeDataset`, the Data Analyzer Streamlit app, and analyzer quickstart notebooks.

## Install From Public GitHub

```bash
pip install "data-analysis-utils @ git+https://github.com/Leeishere/data_analysis_utils.git@main"
```

Then import it from any project in the same Python environment:

```python
from data_analysis_utils import AnalyzeDataset
from data_analysis_utils.Consumer_Habits_file_loader import load_consumer_habits
```

## Local Development

From this repository:

```bash
python3 -m pip install -r requirements.txt
```

The Streamlit app lives in `apps/APP_DataAnalyzer.py`, feedback is stored in `data_analyzer_app_feedback.txt`, and analyzer notebooks live in `examples/`.