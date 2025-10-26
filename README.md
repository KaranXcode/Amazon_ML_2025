# Smart Pricing — Our Approach

## overview (high-level)

Inputs: `catalog_content` (text), `image_link` (optional), `sample_id`.

Pipeline (summary):

1. Text preprocessing & structured extraction

- Use regex-based extractors to pull structured numeric fields and flags (pack size, ounce/fl_oz, gluten-free, organic, bullet counts, etc.).
- Clean and combine text fields into a single string for embedding.

2. Text features

- DistilBERT embeddings (mean-pooled last hidden state) computed per sample.
- TF-IDF vectorizer (top 1,000 tokens) computed on combined train+test text.

3. Numeric features

- Extracted numeric and boolean features are standardized (StandardScaler) before fusion.

4. Visual features (optional)

- ResNet50 (pretrained) used as a feature extractor. If images are not present on disk, the pipeline assigns reproducible random vectors to missing images so training remains deterministic.

5. Feature fusion

- Concatenate [BERT embeddings, TF-IDF, visual features, numeric features] into a single feature matrix.

6. Modeling

- Train LightGBM and XGBoost regressors on log1p(price) to reduce skew.
- Ensemble by averaging predictions from both models.
- Metric: SMAPE (Symmetric Mean Absolute Percentage Error) reported on validation data.

7. Output

- Inverse transform predictions with expm1 and save a submission CSV (`output/submission.csv`) with columns `sample_id,price`.

## Design contract (inputs/outputs, success criteria)

- Inputs: `dataset/train.csv` (with `price`), `dataset/test.csv` (no `price`), plus optional local image files pointed to by `image_link`.
- Output: `output/submission.csv` with `sample_id` and `price` (positive float). Also saves models and component feature artifacts in `output/`.
- Success: Pipeline runs end-to-end and produces `output/submission.csv` formatted like `dataset/sample_test_out.csv`. Validation SMAPE should decrease after tuning.

## How to run (recommended quick test and full run)

Important note about data paths: `src/Final.py` defaults to a hard-coded `BASE_DIR` value intended for the original dev environment. Before running, edit the `BASE_DIR` constant near the top of `src/Final.py` to point to this project's dataset root, for example:

Change the top of `src/Final.py` from:

BASE_DIR = Path('/home/dc_gr1/DC_Biswas')

to (example relative to repository root):

BASE_DIR = Path(**file**).resolve().parents[1] / 'dataset'

Or replace with the absolute path to `Final_Project/dataset` on your machine.

Once paths are set, create a virtual environment and install dependencies. A minimal `requirements.txt` should include:

```
numpy
pandas
scikit-learn
torch
torchvision
transformers
joblib
lightgbm
xgboost
nltk
requests
tqdm
Pillow
```

Quick test (small sample): set an environment variable to limit samples and run the script (PowerShell example):

```powershell
#$env:SAMPLE_LIMIT = 200  # uncomment and set to run a quick local debug
#python src/Final.py

# Example to run a short test (set SAMPLE_LIMIT to 200):
$env:SAMPLE_LIMIT=200; python src/Final.py
```

Full run (no SAMPLE_LIMIT):

```powershell
python src/Final.py
```

Outputs will be written to `output/` (created by the script). The main submission file is `output/submission.csv`.

## Notes, assumptions and caveats

- BASE_DIR path must be set correctly for `train.csv` and `test.csv` locations; the script currently expects `train.csv` and `test.csv` directly under `BASE_DIR` (see `load_data()` in `src/Final.py`).
- Visual feature extraction expects image directories as in the code (`/home/dc_gr1/DC_Biswas/Trainimages` and `/home/dc_gr1/DC_Biswas/TestImges`) by default. If you have images locally, either update those paths in `src/Final.py` or place images in the same names/structure. When images are missing, the script uses reproducible random vectors so the pipeline still runs.
- The pipeline uses DistilBERT; computing embeddings may be slow on CPU and requires significant memory. Use `SAMPLE_LIMIT` to test quickly.
- The model trains on log1p-transformed prices. All model-related predictions and SMAPE evaluation functions operate in log space and transform back with expm1 for final outputs.
- Licensing: downstream model artifacts should comply with MIT/Apache-2.0 if you plan to release them, as required by the challenge.

## Reproducibility & debugging tips

- When iterating, run with `SAMPLE_LIMIT` to shorten runtime.
- Component arrays are cached into `output/` (e.g., `train_bert.npy`, `train_tfidf.npy`) — reusing them avoids expensive recomputation.
- If you add new Python dependencies, add them to `requirements.txt` and re-run in a new venv.

## Next steps and improvements (suggested)

- Hyperparameter tuning (Optuna / grid search) for LightGBM/XGBoost.
- Replace random-fallback visual features by downloading images using `src/utils.py` and storing them under the expected image directories.
- Try larger or fine-tuned transformer models for text (BERT variants), or train a small gradient-boosted model on hand-crafted features only (faster baseline).
- Add unit tests for numeric extraction functions (e.g., `extract_numeric_features`) and a small CI job to run quick smoke tests.

---

If you want, I can:

- Edit `src/Final.py` to automatically derive `BASE_DIR` from the repository layout so you don't need to modify the file.
- Add a `requirements.txt` and set up a small PowerShell run script that sets `SAMPLE_LIMIT` and runs the pipeline.

Tell me which of those you'd like me to do next.

1. `sample_id` — unique identifier per sample
2. `catalog_content` — text that concatenates title, description, and Item Pack Quantity (IPQ)
3. `image_link` — public URL for the product image. Use `src/utils.py` to download images.
4. `price` — target (only in training data)

## How to run the provided sample code

The repository includes a small `sample_code.py` to demonstrate producing an output file in the correct format. To run it (PowerShell / Windows):

```powershell
python sample_code.py
```

The script should write a CSV matching the format of `dataset/sample_test_out.csv`. Use that file to confirm your pipeline output formatting.

## Downloading images

`src/utils.py` contains a `download_images` helper that can download images from `image_link`. Note that downloads may need retries because of throttling. Example usage is included in `src/example.ipynb`.

## Output / submission format

Your submission must be a CSV with exactly two columns and no extra index column or headers beyond the column names:

1. `sample_id` — must exactly match the test set IDs
2. `price` — predicted price (positive float)

Match the formatting in `dataset/sample_test_out.csv` exactly. Missing or extra rows will cause evaluation failure.

## Evaluation metric

Submissions are evaluated using Symmetric Mean Absolute Percentage Error (SMAPE):

$$\mathrm{SMAPE} = \frac{1}{n} \sum \frac{|\text{predicted} - \text{actual}|}{(|\text{actual}| + |\text{predicted}|)/2}$$

Lower SMAPE is better. SMAPE ranges from 0% to 200%.

## Constraints & rules

- Predicted prices must be positive floats.
- Participants must not use external price lookups or external price data (web scraping, APIs, manual lookup, etc.). Using external pricing information will result in disqualification.
- Final model artifacts must be compatible with MIT or Apache-2.0 licensing and be no larger than ~8B parameters.

## Submission checklist

1. Produce `test_out.csv` in the same format as `dataset/sample_test_out.csv`.
2. Provide a 1-page methodology document (use `Documentation_template.md`). The document should cover:
   - Methodology and model architecture
   - Feature engineering and preprocessing
   - Any ensembling, training, and evaluation details

## Tips and suggestions

- Use both textual features (`catalog_content`) and image features when helpful.
- Handle outliers and data cleaning carefully. Ensure no negative or zero prices are submitted.
- Consider ensembles combining different model families (text models, vision models, gradient boosted trees, etc.).

## Notes for developers / contributors

- If you add dependencies, include a `requirements.txt` at project root and document the install steps here.
- Keep model training reproducible: store random seeds, environment, and model checkpoints.

## Contact / License

This repository is provided as an educational challenge. See `Documentation_template.md` for submission documentation guidance.

---

If you'd like, I can also:

- Add a minimal `requirements.txt` with common ML packages used in the examples.
- Add a short PowerShell script to run `sample_code.py` and produce `test_out.csv`.

Tell me which of those you'd like me to add next.
