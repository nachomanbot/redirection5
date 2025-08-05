# Refactored AI-Powered Redirect Mapping Tool for Custom GPT
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import re
from difflib import SequenceMatcher
import time

# ------------------------
# UI CONFIGURATION
# ------------------------
st.set_page_config(page_title="AI-Powered Redirect Mapping Tool v5.1", layout="wide")
st.title("⚡ AI-Powered Redirect Mapping Tool - Refactored v5.1")

st.markdown("""
**Relevancy Script** by Daniel Emery  
**Tool Dev & Refactor** by NDA

### 🔧 What It Does
Matches URLs from an origin site to a destination site during migrations using:
- Partial URL similarity
- Sentence-level semantic matching (via Sentence Transformers)
- Rule-based fallbacks (from `rules.csv`)

### 🚀 How to Use
1. Upload your `origin.csv` and `destination.csv`. Ensure columns: Address, Title 1, Meta Description 1, H1-1
2. Files should contain **relative URLs only**, with status 200
3. Adjust matching thresholds
4. Hit **"Let's Go!"**
5. Download the result
""")

# ------------------------
# AUTHENTICATION
# ------------------------
if st.text_input("Enter Password to Access the Tool:", type="password") != "@SEOvaga!!!":
    st.warning("Please enter the correct password to proceed.")
    st.stop()

# ------------------------
# FILE UPLOAD
# ------------------------
st.header("Step 1: Upload Files")
origin_file = st.file_uploader("Upload origin.csv", type="csv")
dest_file = st.file_uploader("Upload destination.csv", type="csv")

rules_path = 'rules.csv'
if not os.path.exists(rules_path):
    st.error("Rules file (rules.csv) not found in the backend.")
    st.stop()

rules_df = pd.read_csv(rules_path, encoding="ISO-8859-1")
if 'Destination URL Pattern' not in rules_df.columns or 'Keyword' not in rules_df.columns:
    st.error("'rules.csv' must contain 'Keyword' and 'Destination URL Pattern' columns.")
    st.stop()

rules_df = rules_df.sort_values(by='Priority')

# ------------------------
# LOAD & CLEAN DATA
# ------------------------
def clean_and_prepare(df):
    df.columns = df.columns.str.replace('ï»¿', '').str.strip()
    if 'Address' not in df.columns:
        df.rename(columns={df.columns[0]: 'Address'}, inplace=True)
    df['combined_text'] = df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return df

if origin_file and dest_file:
    st.success("Files uploaded successfully!")
    try:
        origin_df = clean_and_prepare(pd.read_csv(origin_file, encoding="ISO-8859-1"))
        dest_df = clean_and_prepare(pd.read_csv(dest_file, encoding="ISO-8859-1"))
    except UnicodeDecodeError:
        st.error("File encoding error. Use UTF-8 or ISO-8859-1.")
        st.stop()

    # ------------------------
    # USER SETTINGS
    # ------------------------
    st.header("Step 2: Customize Settings")
    strategy = st.radio("Primary Matching Strategy", ("Partial Match", "Similarity Score"))
    partial_thresh = st.slider("Partial Match Threshold (%)", 50, 100, 65, 5)
    sim_thresh = st.slider("Similarity Score Threshold (%)", 50, 100, 60, 5)

    # ------------------------
    # PROCESS MATCHING
    # ------------------------
    if st.button("Let's Go!"):
        start = time.time()
        st.info("Processing... please wait.")

        def partial_match(origin):
            best_score, best_url = 0, '/'
            origin_str = str(origin).lower()
            for dest in dest_df['Address']:
                dest_str = str(dest).lower()
                score = SequenceMatcher(None, origin_str, dest_str).ratio() * 100
                if score > best_score:
                    best_score, best_url = score, dest
            return best_url if best_score > partial_thresh else '/'

        matches = []
        for i, row in origin_df.iterrows():
            origin_url = row['Address']
            match_url, method = '/', 'Unmatched'

            if strategy == 'Partial Match':
                match_url = partial_match(origin_url)
                method = 'Partial Match' if match_url != '/' else 'Unmatched'

            matches.append([origin_url, match_url, method])

        matches_df = pd.DataFrame(matches, columns=['origin_url', 'matched_url', 'method'])

        # SEMANTIC MATCH FOR UNMATCHED
        if 'Similarity Score' in strategy or (strategy == 'Partial Match' and (matches_df['matched_url'] == '/').any()):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
            dest_embeddings = model.encode(dest_df['combined_text'].tolist(), show_progress_bar=True)

            index = faiss.IndexFlatL2(origin_embeddings.shape[1])
            index.add(dest_embeddings.astype('float32'))
            D, I = index.search(origin_embeddings.astype('float32'), 1)

            for idx in matches_df[matches_df['matched_url'] == '/'].index:
                sim_score = 1 - (D[idx][0] / (np.max(D) + 1e-10))
                if sim_score * 100 >= sim_thresh:
                    matches_df.at[idx, 'matched_url'] = dest_df.iloc[I[idx][0]]['Address']
                    matches_df.at[idx, 'method'] = f"Sim Score: {round(sim_score*100, 2)}%"

        # FALLBACK MATCHING
        def apply_fallback(origin_url):
            origin_url = origin_url.lower().strip().rstrip('/')
            for _, rule in rules_df.iterrows():
                keyword = str(rule.get('Keyword', '')).lower().strip()
                destination_pattern = str(rule.get('Destination URL Pattern', '')).strip()
                if keyword and re.search(re.escape(keyword), origin_url):
                    for pattern in destination_pattern.split('|'):
                        cleaned_pattern = pattern.strip()
                        if cleaned_pattern in dest_df['Address'].values:
                            return cleaned_pattern
            return '/'

        for idx in matches_df[matches_df['matched_url'] == '/'].index:
            fallback_url = apply_fallback(matches_df.at[idx, 'origin_url'])
            matches_df.at[idx, 'matched_url'] = fallback_url
            matches_df.at[idx, 'method'] = 'Fallback Rule'

        # FORCE HOMEPAGE FOR HOMEPAGE
        homepage_variants = ['/', 'index.html', 'index.php']
        matches_df.loc[matches_df['origin_url'].str.lower().isin(homepage_variants), ['matched_url', 'method']] = ['/', 'Homepage']

        # ------------------------
        # DISPLAY RESULTS
        # ------------------------
        elapsed = time.time() - start
        st.success(f"✅ Matching complete in {elapsed:.2f} seconds for {len(matches_df)} URLs.")
        st.dataframe(matches_df)

        st.download_button("⬇️ Download Mapped Redirects", data=matches_df.to_csv(index=False),
                           file_name="redirect_mapping_output_v5.1.csv", mime="text/csv")
