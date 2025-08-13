# Calorie Burn Prediction — Streamlit App

## Quickstart

1. Create & activate a virtual environment
   - Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```
     python -m venv venv
     source venv/bin/activate
     ```

2. Install dependencies:
pip install -r requirements.txt

markdown
Copy code

3. Ensure CSV files are placed in `data/`:
- `data/exercise.csv`
- `data/calories.csv`

4. Run the app:
streamlit run app.py

markdown
Copy code

## Notes
- Models are saved in `models/` after training.
- If columns names mismatch, update mappings in `utils/data_loader.py`.