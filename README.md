# data_analysis_utils

Reusable Python package for data analysis helpers, quickstart notebooks, and the Streamlit data analyzer app.

## Install From GitHub

```bash
pip install "data-analysis-utils @ git+https://github.com/Leeishere/data_analysis_utils.git@main"
```

Then import it from any project in the same Python environment:

```python
from data_analysis_utils import AnalyzeDataset
from data_analysis_utils import MuEstimator, PoissonSalesForecasting
from data_analysis_utils import ProbabilisticModeling, RegressionOrdinalizer
from data_analysis_utils.Clustering import ClusterFeatureModels
```

The bundled Consumer Habits sample dataset can be loaded with:

```python
from data_analysis_utils.Consumer_Habits_file_loader import load_consumer_habits

df = load_consumer_habits()
```

## Local Development

From this repository:

```bash
python3 -m pip install -e .
```

The Streamlit app lives in `apps/data_analyzer_app.py`, and the notebook quickstart lives in `examples/analyze_dataset_quickstart.ipynb`.
