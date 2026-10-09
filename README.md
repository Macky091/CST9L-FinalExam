# Phishing Detector (Streamlit)

Folder layout (all files go in the root of your GitHub repo, with `.streamlit/` as a sub-folder):

```
app.py
requirements.txt
tuned_random_forest.skl     (33 MB, under GitHub's 100 MB limit)
dataset1.csv
.streamlit/config.toml
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push the files above to a GitHub repo.
2. Go to share.streamlit.io, choose New app, pick the repo, set the main file to `app.py`, then Deploy.

## Notes
- `scikit-learn==1.6.1` is pinned because the model file was saved with that version. A different version may fail to load it or give different results.
- Set `SHOW_DUPLICATE_NOTE = False` near the top of `app.py` to hide the duplicate-records note.
