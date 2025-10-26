"""
End-to-end pipeline: text (BERT + TF-IDF + numeric), visual, fuse, train LightGBM/XGBoost, predict

Usage: python final_code.py
Requires: transformers, torch, sklearn, lightgbm, xgboost, pandas, numpy, joblib

This script reuses preprocessing logic in text_preprocess.py and visual_preprocess.py where possible.
"""

import os
import sys
import gc
import time
import json
import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from torchvision import models, transforms
from PIL import Image

# Local utilities (reuse functions from text_preprocess.py where helpful)
import re
import nltk
from nltk.corpus import stopwords

# Ensure necessary NLTK resources are available (silent if already present)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def extract_structured_content(text):
    """Extract structured information from catalog content (robust, returns dict).
    This mirrors the logic from the original text_preprocess module but keeps it
    lightweight: it looks for common labeled fields and falls back to empty values.
    """
    if not isinstance(text, str):
        return {}
    features = {}
    # Item Name
    item_name_match = re.search(r'Item Name:(.*?)(?:\n|$)', text)
    features['item_name'] = item_name_match.group(1).strip() if item_name_match else ""

    # Bullet points (numbered or generic)
    bullet_points = []
    bullet_pattern = r'Bullet Point \d+:(.*?)(?:\n|$)'
    bullet_matches = re.findall(bullet_pattern, text)
    if not bullet_matches:
        generic_bullet_pattern = r'Bullet Point:(.*?)(?:\n|$)'
        bullet_matches = re.findall(generic_bullet_pattern, text)
    for bullet in bullet_matches:
        bullet_points.append(bullet.strip())
    features['bullet_points'] = bullet_points
    features['bullet_count'] = len(bullet_points)

    # Product description
    desc_match = re.search(r'Product Description:(.*?)(?:\n|$)', text)
    features['product_description'] = desc_match.group(1).strip() if desc_match else ""

    # Value and unit
    value_match = re.search(r'Value:(.*?)(?:\n|$)', text)
    if value_match and value_match.group(1).strip():
        try:
            features['value'] = float(value_match.group(1).strip())
        except ValueError:
            features['value'] = None
    else:
        features['value'] = None

    unit_match = re.search(r'Unit:(.*?)(?:\n|$)', text)
    features['unit'] = unit_match.group(1).strip() if unit_match else ""

    return features


def extract_numeric_features(item_text):
    """Extract numeric and boolean features from free-text catalog content.
    Returns a dict with a fixed set of keys (fill with None/0 where not found).
    """
    numeric_features = {}
    if not isinstance(item_text, str):
        return {
            'pack_size': 1,
            'ounce_size': None,
            'total_ounce': None,
            'count_size': None,
            'fl_oz_size': None,
            'total_fl_oz': None,
            'is_gluten_free': 0,
            'is_organic': 0,
        }

    # pack size patterns
    pack_pattern = r'(?:Pack of|pack of|pack)\s+(\d+)|(\d+)(?:\s*-\s*|\s+)(?:pack|pk|count per order)'
    pack_match = re.search(pack_pattern, item_text, re.IGNORECASE)
    if not pack_match:
        alt_pack_pattern = r'(\d+)\s+(?:per case|per pack|count\b|ct\b)'
        pack_match = re.search(alt_pack_pattern, item_text, re.IGNORECASE)
    if not pack_match:
        numeric_features['pack_size'] = 1
    else:
        matched_groups = pack_match.groups()
        matched_value = next((g for g in matched_groups if g), None)
        numeric_features['pack_size'] = int(matched_value) if matched_value else 1

    # ounce information
    ounce_pattern = r'(\d+\.?\d*)\s*(?:Ounce|Oz\b|oz\b)'
    ounce_match = re.search(ounce_pattern, item_text, re.IGNORECASE)
    numeric_features['ounce_size'] = float(ounce_match.group(1)) if ounce_match else None

    # fluid ounce
    fl_oz_pattern = r'(\d+\.?\d*)\s*(?:Fl Oz|Fluid Ounce|fl oz|fl\.oz)'
    fl_oz_match = re.search(fl_oz_pattern, item_text, re.IGNORECASE)
    numeric_features['fl_oz_size'] = float(fl_oz_match.group(1)) if fl_oz_match else None

    # totals
    if numeric_features.get('ounce_size') is not None and numeric_features['pack_size'] > 1:
        numeric_features['total_ounce'] = numeric_features['ounce_size'] * numeric_features['pack_size']
    else:
        numeric_features['total_ounce'] = numeric_features.get('ounce_size')

    if numeric_features.get('fl_oz_size') is not None and numeric_features['pack_size'] > 1:
        numeric_features['total_fl_oz'] = numeric_features['fl_oz_size'] * numeric_features['pack_size']
    else:
        numeric_features['total_fl_oz'] = numeric_features.get('fl_oz_size')

    # count
    count_pattern = r'(\d+\.?\d*)\s*(?:count|piece|tablet|capsule)s?'
    count_match = re.search(count_pattern, item_text, re.IGNORECASE)
    numeric_features['count_size'] = float(count_match.group(1)) if count_match else None

    numeric_features['is_gluten_free'] = 1 if re.search(r'gluten[ -]free', item_text, re.IGNORECASE) else 0
    numeric_features['is_organic'] = 1 if re.search(r'\borganic\b', item_text, re.IGNORECASE) else 0

    return numeric_features


def clean_text_for_embedding(text):
    """Basic cleaning for text before BERT/TF-IDF: lowercasing, remove extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ')
    # remove control characters and weird punctuation, keep usual punctuation
    text = re.sub(r'[^\w\s.,;:!?%\-\(\)]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def combine_text_fields(row):
    """Create a combined text string for embedding/TF-IDF. Prefer structured fields if available,
    otherwise fall back to the raw 'catalog_content' field.
    """
    combined = ""
    # If structured fields exist on the row, use them
    if isinstance(row, dict):
        # dict-like row (not a pandas Series)
        if row.get('item_name'):
            combined += str(row.get('item_name')) + ' '
        if row.get('bullet_points'):
            combined += ' '.join(row.get('bullet_points')) + ' '
        if row.get('product_description'):
            combined += str(row.get('product_description')) + ' '

    else:
        # pandas Series: try to use fields if present
        if 'item_name' in row and pd.notna(row.get('item_name')) and row.get('item_name'):
            combined += str(row.get('item_name')) + ' '
        if 'bullet_points' in row and isinstance(row.get('bullet_points'), list):
            combined += ' '.join(row.get('bullet_points')) + ' '
        if 'product_description' in row and pd.notna(row.get('product_description')) and row.get('product_description'):
            combined += str(row.get('product_description')) + ' '

        # Fallback: use catalog_content
        if not combined and 'catalog_content' in row and pd.notna(row.get('catalog_content')):
            combined = str(row.get('catalog_content'))

    return clean_text_for_embedding(combined)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths (adjust as needed)
BASE_DIR = Path('/home/dc_gr1/DC_Biswas')
DATA_DIR = BASE_DIR 
OUTPUT_DIR = BASE_DIR / 'output'
IMAGES_OUTPUT_DIR = BASE_DIR / 'output'  # where visual_preprocess saved image_features.npz

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
BATCH_SIZE = 32
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    train_path = DATA_DIR / 'train.csv'
    test_path = DATA_DIR / 'test.csv'
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError('train.csv or test.csv not found in dataset folder')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_bert_model(device):
    logger.info('Loading DistilBERT model/tokenizer...')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.eval()
    model.to(device)
    return tokenizer, model


def compute_bert_embeddings(texts, tokenizer, model, device, batch_size=32, max_length=128):
    """Compute BERT embeddings (mean-pooled last hidden state) for a list of texts"""
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc='BERT batches'):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            # mean pooling
            pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(pooled)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return np.vstack(embeddings)


def generate_tfidf_features(texts, max_features=1000):
    logger.info(f'Generating TF-IDF features (max_features={max_features})')
    vec = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vec.fit_transform(texts)
    # sklearn <1.0 uses get_feature_names(), newer versions have get_feature_names_out()
    try:
        feature_names = vec.get_feature_names_out()
    except AttributeError:
        feature_names = vec.get_feature_names()
    return X.toarray(), feature_names, vec


def load_visual_features(sample_ids, visual_npz_path=None, feature_dim=2048):
    """Load visual features from npz saved by visual_preprocess.py and map to sample ids.
    If missing, fill with random features (fixed seed) or zeros.
    visual_npz is expected to contain keys that are image filenames or sample ids depending on earlier script.
    We'll try to map by filename first, then by sample_id as str.
    """
    logger.info('Loading visual features...')
    features = {}
    if visual_npz_path is None:
        # look for likely files
        candidates = list(IMAGES_OUTPUT_DIR.glob('*.npz'))
        if not candidates:
            logger.warning('No visual features npz found; returning zeros')
            return np.zeros((len(sample_ids), feature_dim), dtype=np.float32)
        visual_npz_path = candidates[0]
    try:
        loaded = np.load(visual_npz_path, allow_pickle=True)
        logger.info(f'Loaded visual features file: {visual_npz_path}')
    except Exception as e:
        logger.warning(f'Could not load visual features: {e}');
        return np.zeros((len(sample_ids), feature_dim), dtype=np.float32)

    # Convert keys to str
    available_keys = {str(k): loaded[k] for k in loaded.files}

    # map by sample_id (string) or filename
    out = np.zeros((len(sample_ids), next(iter(available_keys.values())).shape[0]), dtype=np.float32)
    rng = np.random.RandomState(RANDOM_SEED)
    for i, sid in enumerate(sample_ids):
        sid_str = str(sid)
        if sid_str in available_keys:
            out[i] = available_keys[sid_str]
        elif (sid_str + '.jpg') in available_keys:
            out[i] = available_keys[sid_str + '.jpg']
        else:
            # fallback random features
            out[i] = rng.normal(0, 0.1, size=out.shape[1]).astype(np.float32)
    return out


def create_resnet50_extractor(device):
    logger.info('Loading ResNet50 feature extractor...')
    model = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)
    return feature_extractor


def load_and_preprocess_image(image_path, transform):
    try:
        img = Image.open(image_path).convert('RGB')
        return transform(img)
    except Exception as e:
        logger.debug(f'Error processing image {image_path}: {e}')
        return None


def extract_visual_features_from_images(df, images_dir, device, batch_size=16, feature_extractor=None):
    """Extract visual features for each row in df. Returns numpy array with shape (n_samples, feat_dim).
    images_dir: Path or str pointing to directory containing image files named by basename of image_link or sample_id.
    Strategy: for each row, try basename(image_link) in images_dir; if missing, try sample_id with common extensions; if still missing, fill with random vector.
    """
    images_dir = Path(images_dir)
    n = len(df)
    # prepare transform matching visual_preprocess.py
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # determine feature dim by running a dummy tensor through extractor
    if feature_extractor is None:
        feature_extractor = create_resnet50_extractor(device)

    # Prepare mapping of index -> image_path or None
    image_paths = [None] * n
    for i, row in df.iterrows():
        image_link = row.get('image_link', '')
        sample_id = row.get('sample_id')
        candidate = None
        if pd.notna(image_link) and image_link:
            img_name = os.path.basename(image_link)
            candidate_path = images_dir / img_name
            if candidate_path.exists():
                candidate = candidate_path
        if candidate is None:
            # try sample_id with common extensions
            for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                p = images_dir / f"{sample_id}{ext}"
                if p.exists():
                    candidate = p
                    break
        image_paths[i] = candidate

    # Pre-allocate outputs
    # get feature dim
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        feat = feature_extractor(dummy)
    feat_dim = feat.squeeze().cpu().numpy().shape[0]
    out = np.zeros((n, feat_dim), dtype=np.float32)

    rng = np.random.RandomState(RANDOM_SEED)

    # Process in batches
    indices = list(range(n))
    for start in tqdm(range(0, n, batch_size), desc='Visual batches'):
        batch_idx = indices[start:start+batch_size]
        batch_tensors = []
        valid_idx = []
        for idx in batch_idx:
            p = image_paths[idx]
            if p is not None:
                img_t = load_and_preprocess_image(p, transform)
                if img_t is not None:
                    batch_tensors.append(img_t)
                    valid_idx.append(idx)
        if not batch_tensors:
            # fill these with random vectors
            for idx in batch_idx:
                out[idx] = rng.normal(0, 0.1, size=feat_dim).astype(np.float32)
            continue

        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            batch_feats = feature_extractor(batch_tensor).squeeze().cpu().numpy()
        # Handle single-sample case
        if batch_feats.ndim == 1:
            batch_feats = batch_feats.reshape(1, -1)
        for i_local, idx in enumerate(valid_idx):
            out[idx] = batch_feats[i_local]

        # for any invalid in this batch, fill random
        invalid_idx = [idx for idx in batch_idx if idx not in valid_idx]
        for idx in invalid_idx:
            out[idx] = rng.normal(0, 0.1, size=feat_dim).astype(np.float32)

        del batch_tensor, batch_feats
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return out


def extract_numeric_df(df):
    logger.info('Extracting numeric features from catalog_content')
    # reuse extract_numeric_features from text_preprocess.py which returns dict per row
    rows = []
    for txt in tqdm(df['catalog_content'].fillna('').astype(str), desc='Numeric features'):
        rows.append(extract_numeric_features(txt))
    num_df = pd.DataFrame(rows).fillna(0)
    return num_df


def prepare_features(train_df, test_df, device, bert_tokenizer=None, bert_model=None):
    # If component feature arrays already exist on disk, load them and skip
    # expensive extraction steps (BERT, ResNet). Otherwise compute from data.
    comp_paths = {
        'train_bert': OUTPUT_DIR / 'train_bert.npy',
        'test_bert': OUTPUT_DIR / 'test_bert.npy',
        'train_tfidf': OUTPUT_DIR / 'train_tfidf.npy',
        'test_tfidf': OUTPUT_DIR / 'test_tfidf.npy',
        'train_vis': OUTPUT_DIR / 'train_vis.npy',
        'test_vis': OUTPUT_DIR / 'test_vis.npy',
        'train_num': OUTPUT_DIR / 'train_num.npy',
        'test_num': OUTPUT_DIR / 'test_num.npy',
    }

    if all(p.exists() for p in comp_paths.values()):
        logger.info('Found saved component feature arrays — loading from disk')
        train_bert = np.load(comp_paths['train_bert'])
        test_bert = np.load(comp_paths['test_bert'])
        train_tfidf = np.load(comp_paths['train_tfidf'])
        test_tfidf = np.load(comp_paths['test_tfidf'])
        train_vis = np.load(comp_paths['train_vis'])
        test_vis = np.load(comp_paths['test_vis'])
        train_num = np.load(comp_paths['train_num'])
        test_num = np.load(comp_paths['test_num'])

        # Attempt to load optional artifacts
        tfidf_vec = None
        scaler_num = None
        try:
            tfidf_path = OUTPUT_DIR / 'tfidf_vectorizer.joblib'
            if tfidf_path.exists():
                tfidf_vec = joblib.load(tfidf_path)
        except Exception:
            tfidf_vec = None
        try:
            scaler_path = OUTPUT_DIR / 'numeric_scaler.joblib'
            if scaler_path.exists():
                scaler_num = joblib.load(scaler_path)
        except Exception:
            scaler_num = None

        # If numeric arrays were saved as 2D arrays, keep them as numpy arrays for scaling
        # ensure train_num/test_num are numpy arrays
        train_num = np.array(train_num)
        test_num = np.array(test_num)

        # At this point we have loaded all component arrays. Build fused X_train/X_test
        logger.info('All component arrays loaded from disk — skipping extraction. Building fused feature matrices...')

        # Attempt to load tfidf vectorizer and numeric scaler
        tfidf_vec = None
        scaler_num = None
        try:
            tfidf_path = OUTPUT_DIR / 'tfidf_vectorizer.joblib'
            if tfidf_path.exists():
                tfidf_vec = joblib.load(tfidf_path)
                logger.info(f'Loaded TF-IDF vectorizer from {tfidf_path}')
        except Exception:
            tfidf_vec = None
        try:
            scaler_path = OUTPUT_DIR / 'numeric_scaler.joblib'
            if scaler_path.exists():
                scaler_num = joblib.load(scaler_path)
                logger.info(f'Loaded numeric scaler from {scaler_path}')
        except Exception:
            scaler_num = None

        # If scaler not found, fit a new one on train_num
        if scaler_num is None:
            scaler_num = StandardScaler()
            train_num_scaled = scaler_num.fit_transform(train_num)
        else:
            train_num_scaled = scaler_num.transform(train_num)
        test_num_scaled = scaler_num.transform(test_num)

        # Concatenate features
        logger.info('Fusing features from cached components...')
        X_train = np.hstack([train_bert, train_tfidf, train_vis, train_num_scaled])
        X_test = np.hstack([test_bert, test_tfidf, test_vis, test_num_scaled])

        return X_train, X_test, tfidf_vec, scaler_num

    else:
        # Combine text fields as in text_preprocess
        logger.info('Preparing combined text for BERT/TF-IDF...')
        train_combined = [combine_text_fields(r) for _, r in train_df.iterrows()]
        test_combined = [combine_text_fields(r) for _, r in test_df.iterrows()]

        # Clean texts
        train_combined = [clean_text_for_embedding(t) for t in train_combined]
        test_combined = [clean_text_for_embedding(t) for t in test_combined]

        # BERT embeddings
        if bert_tokenizer is None or bert_model is None:
            bert_tokenizer, bert_model = get_bert_model(device)
        logger.info('Computing BERT embeddings for train...')
        train_bert = compute_bert_embeddings(train_combined, bert_tokenizer, bert_model, device, batch_size=BATCH_SIZE)
        logger.info('Computing BERT embeddings for test...')
        test_bert = compute_bert_embeddings(test_combined, bert_tokenizer, bert_model, device, batch_size=BATCH_SIZE)

        # TF-IDF features (fit on combined train+test to avoid mismatch)
        all_texts = train_combined + test_combined
        tfidf_X, tfidf_names, tfidf_vec = generate_tfidf_features(all_texts, max_features=1000)
        train_tfidf = tfidf_X[:len(train_combined), :]
        test_tfidf = tfidf_X[len(train_combined):, :]

        # Numeric features
        train_num = extract_numeric_df(train_df).values
        test_num = extract_numeric_df(test_df).values

        # Visual features: follow visual_preprocess.py flow to extract for all sample_ids
        # Use fixed known image directories for train and test
    # Use the fixed known image directories for train and test (do not use env vars)
    candidate_train = Path('/home/dc_gr1/DC_Biswas/Trainimages')
    candidate_test = Path('/home/dc_gr1/DC_Biswas/TestImges')

    # Strict mode: require the specified directories to exist. Do not fallback.
    # Candidate paths are either env vars or the known Abhijanteja paths.
    train_images_dir = candidate_train
    test_images_dir = candidate_test
    if not train_images_dir.exists():
        raise FileNotFoundError(f'Required train images directory not found: {train_images_dir}. Please ensure the directory exists and contains train images.')
    if not test_images_dir.exists():
        raise FileNotFoundError(f'Required test images directory not found: {test_images_dir}. Please ensure the directory exists and contains test images.')
    print(f'Train images directory: {train_images_dir}')
    print(f'Test images directory: {test_images_dir}')
    logger.info(f'Using train images dir: {train_images_dir}')
    logger.info(f'Using test images dir: {test_images_dir}')
    device_for_vis = device
    vis_extractor = create_resnet50_extractor(device_for_vis)

    from urllib.parse import urlparse, unquote

    def _basename_from_url(url):
        if not url or not pd.notna(url):
            return ''
        try:
            path = urlparse(url).path
            name = unquote(os.path.basename(path))
            return name
        except Exception:
            return os.path.basename(url)

    def extract_visual_features_full(train_df, test_df, train_images_dir, test_images_dir, extractor, device, batch_size=16, output_dir=OUTPUT_DIR):
        # Combine dataframes to scan all sample IDs
        print('\nSCANNING ALL TRAIN+TEST DATA FOR IMAGES...')
        combined = pd.concat([train_df, test_df], ignore_index=True)
        sample_id_to_image_id = {}
        all_image_ids = set()
        image_paths = []
        image_sample_ids = []
        skipped_images = []

        for idx, row in tqdm(combined.iterrows(), total=len(combined), desc='Scanning sample IDs'):
            sample_id = row['sample_id']
            image_url = row.get('image_link', '')
            image_id = _basename_from_url(image_url) if pd.notna(image_url) and image_url else ''
            sample_id_to_image_id[str(sample_id)] = image_id

            if image_id and image_id not in all_image_ids:
                all_image_ids.add(image_id)
                # Decide whether this sample comes from train or test (by index)
                # combined = concat([train, test]) in that order -> indices < len(train_df) are train
                search_dir = Path(train_images_dir) if idx < len(train_df) else Path(test_images_dir)
                image_path = search_dir / image_id
                # also allow sample_id with common extensions
                candidate_paths = [image_path]
                for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                    candidate_paths.append(search_dir / f"{sample_id}{ext}")
                found = None
                for p in candidate_paths:
                    if p.exists():
                        found = p
                        break
                if found is not None:
                    image_paths.append(str(found))
                    image_sample_ids.append(sample_id)
                else:
                    skipped_images.append((sample_id, image_id))

        print('\n===== IMAGE AVAILABILITY SUMMARY =====')
        print(f'Total unique images in dataset: {len(all_image_ids)}')
        print(f'Images found in directory: {len(image_paths)}')
        print(f'Images not found: {len(skipped_images)}')
        print(f'Total sample IDs (train+test): {len(combined)}')

        if skipped_images:
            print('\nFirst 10 missing images:')
            for i, (sid, iid) in enumerate(skipped_images[:10]):
                print(f'  Sample ID: {sid}, Image: {iid}')
            if len(skipped_images) > 10:
                print(f'  (and {len(skipped_images)-10} more)')
            print(f'Note: {len(skipped_images)} images were not found on disk. These will be assigned reproducible random feature vectors.')

        # Extract features for available images in batches
        print(f'\nProcessing {len(image_paths)} available images...')
        transform_local = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        features_dict = {}
        for i in tqdm(range(0, len(image_paths), batch_size), desc='Processing image batches'):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_paths = []
            for p in batch_paths:
                img_t = load_and_preprocess_image(p, transform_local)
                if img_t is not None:
                    batch_images.append(img_t)
                    valid_paths.append(p)
            if not batch_images:
                continue
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                batch_features = extractor(batch_tensor)
                batch_features = batch_features.squeeze().cpu().numpy()
            if batch_features.ndim == 1:
                batch_features = batch_features.reshape(1, -1)
            for path, feat in zip(valid_paths, batch_features):
                features_dict[os.path.basename(path)] = feat
            del batch_tensor, batch_features, batch_images
            gc.collect()

        if not features_dict:
            print('Error: No image features extracted')
            # create random fallback
            feature_dim = 2048
        else:
            feature_dim = next(iter(features_dict.values())).shape[0]

        # Save extracted image features
        images_feat_path = Path(output_dir) / 'image_features.npz'
        np.savez_compressed(images_feat_path, **features_dict)
        print(f'Saved {len(features_dict)} image features to {images_feat_path}')

        # Map features to all sample IDs. For any unique image basename that wasn't
        # extracted (missing files), create one reproducible random vector and
        # store it in features_dict so all sample IDs that reference the same
        # basename receive identical visual features.
        print('Creating mapping from sample_id to features (including random for missing images)...')
        sample_features = {}
        rng = np.random.RandomState(RANDOM_SEED)
        # compute stats from real features if any
        if features_dict:
            all_feats = np.vstack(list(features_dict.values()))
            feat_mean = float(all_feats.mean())
            feat_std = float(all_feats.std())
        else:
            feat_mean = 0.0
            feat_std = 1.0

        # Ensure every unique image_id referenced by the dataset has an entry in features_dict.
        unique_image_ids = set(v for v in sample_id_to_image_id.values() if v)
        for img_id in unique_image_ids:
            if img_id not in features_dict:
                features_dict[img_id] = rng.normal(loc=feat_mean, scale=feat_std, size=feature_dim).astype(np.float32)

        missing_count = 0
        for idx, row in tqdm(combined.iterrows(), total=len(combined), desc='Mapping features for all sample IDs'):
            sid = str(row['sample_id'])
            img_id = _basename_from_url(row.get('image_link', '')) if pd.notna(row.get('image_link', '')) else ''
            if img_id and img_id in features_dict:
                sample_features[sid] = features_dict[img_id]
            else:
                # fallback: try sample_id with extension keys in features_dict
                found = None
                for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                    key = f"{sid}{ext}"
                    if key in features_dict:
                        found = features_dict[key]
                        break
                if found is not None:
                    sample_features[sid] = found
                else:
                    # final fallback: generate per-sample random vector
                    sample_features[sid] = rng.normal(loc=feat_mean, scale=feat_std, size=feature_dim).astype(np.float32)
                    missing_count += 1

        print(f'Created mapping for {len(sample_features)} samples')
        print(f'  - {len(sample_features)-missing_count} samples with real features')
        print(f'  - {missing_count} samples with random features (missing images)')

        # Save sample-level mapping
        sample_feat_path = Path(output_dir) / 'sample_image_features_all.npz'
        np.savez_compressed(sample_feat_path, **{k: v for k, v in sample_features.items()})
        print(f'Saved sample->image features mapping to {sample_feat_path}')

        # Create arrays for train and test in original order
        def build_array_for_df(df_local):
            arr = np.zeros((len(df_local), feature_dim), dtype=np.float32)
            for i, row in df_local.iterrows():
                sid = str(row['sample_id'])
                arr[i] = sample_features[sid]
            return arr

        train_array = build_array_for_df(train_df)
        test_array = build_array_for_df(test_df)
        return train_array, test_array

        train_vis, test_vis = extract_visual_features_full(train_df, test_df, train_images_dir, test_images_dir, vis_extractor, device_for_vis, batch_size=16, output_dir=OUTPUT_DIR)

    # --- Save per-component feature arrays for reproducibility and faster iteration ---
    try:
        print('\nSaving component feature arrays to output directory...')
        np.save(OUTPUT_DIR / 'train_bert.npy', train_bert.astype(np.float32))
        np.save(OUTPUT_DIR / 'test_bert.npy', test_bert.astype(np.float32))
        np.save(OUTPUT_DIR / 'train_tfidf.npy', train_tfidf.astype(np.float32))
        np.save(OUTPUT_DIR / 'test_tfidf.npy', test_tfidf.astype(np.float32))
        np.save(OUTPUT_DIR / 'train_vis.npy', train_vis.astype(np.float32))
        np.save(OUTPUT_DIR / 'test_vis.npy', test_vis.astype(np.float32))
        # numeric features as arrays
        np.save(OUTPUT_DIR / 'train_num.npy', train_num.values.astype(np.float32))
        np.save(OUTPUT_DIR / 'test_num.npy', test_num.values.astype(np.float32))
        print('Saved: train_bert.npy, test_bert.npy, train_tfidf.npy, test_tfidf.npy, train_vis.npy, test_vis.npy, train_num.npy, test_num.npy')
    except Exception as e:
        print(f'Warning: failed to save component arrays: {e}')

    # Print shapes and a sample preview for clarity
    try:
        print('\n=== Component feature shapes ===')
        print(f'train_bert: {train_bert.shape}, test_bert: {test_bert.shape}')
        print(f'train_tfidf: {train_tfidf.shape}, test_tfidf: {test_tfidf.shape}')
        print(f'train_vis: {train_vis.shape}, test_vis: {test_vis.shape}')
        print(f'train_num: {train_num.shape}, test_num: {test_num.shape}')

        print('\n=== Sample feature preview (train sample 0) ===')
        # BERT
        print('BERT (first 10):', train_bert[0][:10])
        # TF-IDF
        print('TF-IDF (first 10):', train_tfidf[0][:10])
        # Visual
        print('Visual (first 10):', train_vis[0][:10])
        # Numeric (all features)
        print('Numeric features:', train_num.iloc[0].to_dict())
    except Exception as e:
        print(f'Could not print sample preview: {e}')

    # Align shapes and combine
    logger.info('Fusing features...')
    # Standardize numeric features (fit on train)
    scaler_num = StandardScaler()
    train_num_scaled = scaler_num.fit_transform(train_num)
    test_num_scaled = scaler_num.transform(test_num)

    # Optionally scale TF-IDF too (but we keep raw)

    # Final concatenation: BERT (768), TF-IDF (1000), Visual (dim), Numeric (k)
    X_train = np.hstack([train_bert, train_tfidf, train_vis, train_num_scaled])
    X_test = np.hstack([test_bert, test_tfidf, test_vis, test_num_scaled])

    return X_train, X_test, tfidf_vec, scaler_num


def train_and_evaluate(X, y):
    logger.info('Splitting data for training/validation...')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED)

    # LightGBM
    logger.info('Training LightGBM...')
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbosity': -1,
        'seed': RANDOM_SEED
    }
    # Define SMAPE evaluation for LightGBM
    def lgb_smape_eval(preds, train_data):
        labels = train_data.get_label()
        # preds and labels are in log1p space -> convert back
        preds_orig = np.expm1(preds)
        labels_orig = np.expm1(labels)
        denom = (np.abs(labels_orig) + np.abs(preds_orig)) / 2.0
        mask = denom != 0
        res = np.zeros_like(labels_orig)
        res[mask] = np.abs(preds_orig[mask] - labels_orig[mask]) / denom[mask]
        val = float(np.mean(res) * 100)
        return 'smape', val, False

    try:
        lgb_model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_val], feval=lgb_smape_eval, early_stopping_rounds=50, verbose_eval=100)
    except TypeError:
        try:
            callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
            lgb_model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_val], feval=lgb_smape_eval, callbacks=callbacks)
        except Exception:
            print('Warning: LightGBM early stopping not available; training without early stopping')
            lgb_model = lgb.train(params, lgb_train, num_boost_round=500, feval=lgb_smape_eval)

    # XGBoost using sklearn API (XGBRegressor) with early stopping
    logger.info('Training XGBoost (sklearn API) with early stopping...')
    try:
        xgb_model_skl = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=RANDOM_SEED, verbosity=0)
        xgb_model_skl.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=100)
        xgb_model = xgb_model_skl
    except TypeError:
        # Fallback: older xgboost versions may not support early stopping via sklearn API
        # Train without custom feval
        xgb_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'seed': RANDOM_SEED,
            'verbosity': 0
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=[(dtrain, 'train'), (dval, 'eval')], early_stopping_rounds=50, verbose_eval=100)

    # Evaluate: compute SMAPE on train and val using expm1
    lgb_pred_val = lgb_model.predict(X_val)
    xgb_pred_val = xgb_model.predict(xgb.DMatrix(X_val))
    ens_pred_val = 0.5 * lgb_pred_val + 0.5 * xgb_pred_val

    def smape_from_log(y_true_log, y_pred_log):
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        mask = denom != 0
        res = np.zeros_like(y_true)
        res[mask] = np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
        return float(np.mean(res) * 100)

    lgb_smape = smape_from_log(y_val, lgb_pred_val)
    xgb_smape = smape_from_log(y_val, xgb_pred_val)
    ens_smape = smape_from_log(y_val, ens_pred_val)

    logger.info(f'LightGBM SMAPE (val): {lgb_smape:.4f}%')
    logger.info(f'XGBoost SMAPE (val): {xgb_smape:.4f}%')
    logger.info(f'Ensemble SMAPE (val): {ens_smape:.4f}%')

    return lgb_model, xgb_model


def main():
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    train_df, test_df = load_data()
    # By default run on full datasets. You can optionally set the environment variable SAMPLE_LIMIT
    # to a positive integer to run a smaller quick-test (useful during development).
    sample_limit_env = os.environ.get('SAMPLE_LIMIT')
    if sample_limit_env:
        try:
            SAMPLE_LIMIT = int(sample_limit_env)
            if SAMPLE_LIMIT > 0:
                if len(train_df) > SAMPLE_LIMIT:
                    train_df = train_df.sample(n=SAMPLE_LIMIT, random_state=RANDOM_SEED).reset_index(drop=True)
                if len(test_df) > SAMPLE_LIMIT:
                    test_df = test_df.sample(n=SAMPLE_LIMIT, random_state=RANDOM_SEED).reset_index(drop=True)
                print(f'Running in quick-test mode (env): using {len(train_df)} train samples and {len(test_df)} test samples')
            else:
                print(f'Environment SAMPLE_LIMIT is <=0, ignoring and running full dataset')
        except ValueError:
            print('Invalid SAMPLE_LIMIT environment variable; running on full dataset')
    else:
        print(f'Running on full dataset: using {len(train_df)} train samples and {len(test_df)} test samples')

    # Apply log transform to price
    logger.info('Applying log1p transform to price to reduce skew')
    train_df['price_log'] = np.log1p(train_df['price'].astype(float))

    # Prepare features
    bert_tokenizer, bert_model = get_bert_model(device)
    X_train, X_test, tfidf_vec, scaler_num = prepare_features(train_df, test_df, device, bert_tokenizer, bert_model)

    y = train_df['price_log'].values

    # Train models
    lgb_model, xgb_model = train_and_evaluate(X_train, y)

    # Retrain on full data using best iteration counts when available
    logger.info('Retraining models on full training set...')
    try:
        lgb_best_iter = lgb_model.best_iteration
    except Exception:
        lgb_best_iter = 500
    lgb_full = lgb.train({'objective': 'regression', 'metric': 'mae', 'learning_rate': 0.05, 'num_leaves': 31, 'verbosity': -1, 'seed': RANDOM_SEED}, lgb.Dataset(X_train, label=y), num_boost_round=lgb_best_iter)

    try:
        xgb_best_iter = xgb_model.get_booster().best_iteration if hasattr(xgb_model, 'get_booster') else getattr(xgb_model, 'best_iteration', None)
        if xgb_best_iter is None:
            xgb_best_iter = getattr(xgb_model, 'n_estimators', 500)
    except Exception:
        xgb_best_iter = 500
    xgb_full = xgb.XGBRegressor(n_estimators=xgb_best_iter, learning_rate=0.05, max_depth=6, random_state=RANDOM_SEED)
    xgb_full.fit(X_train, y)

    # Predict on test
    logger.info('Predicting on test set...')
    lgb_test_pred = lgb_full.predict(X_test)
    xgb_test_pred = xgb_full.predict(X_test)
    ens_test = 0.5 * lgb_test_pred + 0.5 * xgb_test_pred

    # Inverse transform (expm1)
    preds_expm1 = np.expm1(ens_test)

    # Save models and predictions
    logger.info('Saving models and predictions...')
    joblib.dump(lgb_full, OUTPUT_DIR / 'lgb_model.joblib')
    joblib.dump(xgb_full, OUTPUT_DIR / 'xgb_model.joblib')
    np.save(OUTPUT_DIR / 'test_predictions.npy', preds_expm1)

    submission = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': preds_expm1})
    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False, float_format='%.4f')
    logger.info(f'Submission saved to {OUTPUT_DIR / "submission.csv"}')

    # Save TF-IDF vectorizer and numeric scaler
    joblib.dump(tfidf_vec, OUTPUT_DIR / 'tfidf_vectorizer.joblib')
    joblib.dump(scaler_num, OUTPUT_DIR / 'numeric_scaler.joblib')

    elapsed = time.time() - start
    logger.info(f'Pipeline completed in {elapsed/60:.2f} minutes')

if __name__ == '__main__':
    main()
