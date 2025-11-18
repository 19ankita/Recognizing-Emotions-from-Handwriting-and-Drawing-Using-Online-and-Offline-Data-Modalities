# Recognizing Emotions from Handwriting and Drawing

This project explores handwriting/drawing tasks (Words, Cursive, House, Pentagon, CDT) 
to predict psychological states using regression models (Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting).

## Project Structure
- `main.py`: Entry point. Run experiments across tasks.
- `src/`: All helper modules (reading `.svc`, feature extraction, training).
- `results/`: Contains regression_summary.csv with metrics.
- `dataset/`: Raw handwriting/drawing data (excluded from repo).

## How to Run
Install dependencies:
```bash
pip install -r requirements.txt

## Run experiments (example for words + cursive):
python main.py words cursive --mode all

