"""
Phishing Website Detection Using Random Forest Classification
Streamlit app  ·  Mc Gabriel C. Limbojan  ·  CCE105, University of Mindanao

Files needed in the same folder:
    app.py · tuned_random_forest.skl · dataset1.csv · requirements.txt · .streamlit/config.toml
"""
import time
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------
BASE = Path(__file__).parent
MODEL_PATH = BASE / "tuned_random_forest.skl"
DATA_PATH = BASE / "dataset1.csv"
SHOW_DUPLICATE_NOTE = True        # set to False to hide the duplicate-records note

st.set_page_config(page_title="Phishing Detector · Random Forest", page_icon="🎣", layout="wide")

# Initial (baseline) model numbers come from the report, the tuned model is evaluated live.
INITIAL = {"Accuracy": 0.9751, "Precision": 0.9755, "Recall": 0.9741, "F1-score": 0.9748, "ROC-AUC": 0.9978}
INITIAL_CM = [[946, 34], [21, 1210]]          # rows: true -1, 1   cols: predicted -1, 1
INITIAL_CLASS = {"Phishing": (0.9783, 0.9653, 0.97, 980), "Legitimate": (0.9727, 0.9829, 0.98, 1231)}
IMPORTANCE_TABLE = [  # (feature, initial, tuned) from Table 4 of the report
    ("SSLfinal_State", 0.3124, 0.3093), ("URL_of_Anchor", 0.2398, 0.2449), ("web_traffic", 0.0763, 0.0737),
    ("having_Sub_Domain", 0.0685, 0.0665), ("Prefix_Suffix", 0.0449, 0.0455), ("Links_in_tags", 0.0444, 0.0451),
    ("SFH", 0.0216, 0.0215), ("Links_pointing_to_page", 0.0199, 0.0198)]

# ----------------------------------------------------------------------------------------
# Feature dictionary: plain-language label, group and what each coded value (-1 / 0 / 1) means
# ----------------------------------------------------------------------------------------
META = {
    "having_IPhaving_IP_Address": ("IP address in URL", "Address bar", {1: "No IP address in the URL", -1: "URL uses an IP address"}),
    "URLURL_Length": ("URL length", "Address bar", {1: "Short (under 54 characters)", 0: "Medium (54 to 75 characters)", -1: "Long (over 75 characters)"}),
    "Shortining_Service": ("URL shortener", "Address bar", {1: "No shortening service", -1: "Uses a shortener (e.g. TinyURL)"}),
    "having_At_Symbol": ("'@' symbol", "Address bar", {1: "No '@' in the URL", -1: "'@' present in the URL"}),
    "double_slash_redirecting": ("Double-slash redirect", "Address bar", {1: "No '//' redirect in the path", -1: "'//' redirect in the path"}),
    "Prefix_Suffix": ("Hyphen in domain", "Address bar", {1: "No hyphen in the domain", -1: "Hyphen in the domain"}),
    "having_Sub_Domain": ("Sub-domains", "Address bar", {1: "One dot (no sub-domain)", 0: "Two dots (one sub-domain)", -1: "Three or more dots"}),
    "HTTPS_token": ("'https' in domain name", "Address bar", {1: "No 'https' token in the domain", -1: "'https' token inside the domain"}),
    "port": ("Port", "Address bar", {1: "Standard port", -1: "Non-standard port open"}),
    "Abnormal_URL": ("Hostname in URL", "Address bar", {1: "Hostname matches the URL", -1: "Hostname missing from the URL"}),
    "SSLfinal_State": ("SSL certificate", "Certificate and domain", {1: "HTTPS with a trusted issuer", 0: "HTTPS, untrusted issuer", -1: "No HTTPS / invalid certificate"}),
    "Domain_registeration_length": ("Domain registration length", "Certificate and domain", {1: "Registered for more than 1 year", -1: "Expires within 1 year"}),
    "age_of_domain": ("Domain age", "Certificate and domain", {1: "Older than 6 months", -1: "Younger than 6 months"}),
    "DNSRecord": ("DNS record", "Certificate and domain", {1: "DNS record found", -1: "No DNS record"}),
    "Favicon": ("Favicon source", "Page content", {1: "Loaded from the same domain", -1: "Loaded from another domain"}),
    "Request_URL": ("External objects", "Page content", {1: "Under 22% from other domains", 0: "22% to 61% from other domains", -1: "Over 61% from other domains"}),
    "URL_of_Anchor": ("Anchor links to other domains", "Page content", {1: "Under 31% of anchors", 0: "31% to 67% of anchors", -1: "Over 67% of anchors"}),
    "Links_in_tags": ("Links in meta/script/link tags", "Page content", {1: "Under 17% to other domains", 0: "17% to 81% to other domains", -1: "Over 81% to other domains"}),
    "SFH": ("Server form handler", "Page content", {1: "Form handled on the same domain", 0: "Form handled on another domain", -1: "Form handler blank / about:blank"}),
    "Submitting_to_email": ("Submits to email", "Page content", {1: "Forms submit to a server", -1: "Forms submit to email (mailto)"}),
    "Iframe": ("Iframe", "Page content", {1: "No hidden iframe", -1: "Iframe present"}),
    "Redirect": ("Redirects", "Page behaviour", {0: "One redirect or fewer", 1: "Two or more redirects"}),
    "on_mouseover": ("Status bar on hover", "Page behaviour", {1: "Status bar unchanged", -1: "Status bar changes on hover"}),
    "RightClick": ("Right-click", "Page behaviour", {1: "Right-click enabled", -1: "Right-click disabled"}),
    "popUpWidnow": ("Pop-up window", "Page behaviour", {1: "No pop-up asking for input", -1: "Pop-up asks for input"}),
    "web_traffic": ("Web traffic rank", "Reputation", {1: "Ranked in the top 100,000", 0: "Ranked below 100,000", -1: "No traffic ranking"}),
    "Page_Rank": ("PageRank", "Reputation", {1: "PageRank above 0.2", -1: "PageRank below 0.2"}),
    "Google_Index": ("Google index", "Reputation", {1: "Indexed by Google", -1: "Not indexed"}),
    "Links_pointing_to_page": ("Inbound links", "Reputation", {1: "More than 2 links point to it", 0: "1 to 2 links point to it", -1: "No links point to it"}),
    "Statistical_report": ("Phishing reports", "Reputation", {1: "Not in phishing reports", -1: "Listed in phishing reports"}),
}
GROUPS = ["Address bar", "Certificate and domain", "Page content", "Page behaviour", "Reputation"]
LABEL_NAME = {-1: "Phishing", 1: "Legitimate"}

# ----------------------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600;700&display=swap');
:root{--navy:#1b1936;--ink:#16142b;--lime:#d4f25a;--indigo:#4b3fd1;--indigo-soft:#ecebfb;--bg:#f5f5fa;--card:#ffffff;
--line:#e9e9f3;--muted:#6b6a80;--red:#d6336c;--red-soft:#fdeaf1;--display:'Bricolage Grotesque',system-ui,sans-serif;}
html,body,[class*="css"],.stApp{font-family:'Instrument Sans',system-ui,sans-serif;color:var(--ink);}
.stApp{background:var(--bg);}
#MainMenu{visibility:visible;} footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1180px;padding-top:2.4rem;padding-bottom:4rem;}
/* sidebar */
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);}
[data-testid="stSidebar"]>div:first-child{width:310px;}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarUserContent"]{padding-top:2.4rem;}
.side-title{font-family:var(--display);font-weight:800;font-size:2.15rem;letter-spacing:-.04em;line-height:1;margin:0;}
.side-sub{color:var(--muted);font-size:.95rem;margin:.35rem 0 1.6rem 0;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.55rem;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]{background:#fff;border:1px solid #d9d9e6;border-radius:999px;
padding:.62rem 1.1rem .62rem .9rem;margin:0;width:fit-content;transition:background .15s,border-color .15s;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover{border-color:var(--indigo);}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] p{font-weight:600;font-size:.98rem;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked){background:var(--ink);border-color:var(--ink);}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p{color:#fff;}
.side-card{background:#f6f6fb;border-radius:18px;padding:1.1rem 1.2rem;margin-top:1.8rem;font-size:.86rem;line-height:1.55;color:var(--muted);}
.side-card b{color:var(--ink);display:block;margin-bottom:.35rem;font-size:.9rem;}
.side-foot{color:var(--muted);font-size:.82rem;margin-top:1.1rem;}
/* page head */
.page-head{display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;margin-bottom:1.3rem;}
.page-head .ph-title{font-family:var(--display);font-weight:800;font-size:2.6rem;letter-spacing:-.045em;margin:0;padding:0;line-height:1.05;}
.page-head span{color:var(--muted);font-size:1rem;}
/* cards */
.card{background:var(--card);border-radius:24px;padding:1.6rem 1.7rem;margin-bottom:1.1rem;}
.card h3{font-family:var(--display);font-weight:700;font-size:1.4rem;letter-spacing:-.03em;margin:0 0 .35rem 0;}
.card h4{font-family:var(--display);font-weight:700;font-size:1.1rem;letter-spacing:-.02em;margin:1.2rem 0 .3rem 0;}
.card p,.card li{font-size:1rem;line-height:1.65;max-width:75ch;}
.card ul{padding-left:1.15rem;margin:.4rem 0 0 0;}
.card li{margin-bottom:.45rem;}
.card .hint{color:var(--muted);font-size:.9rem;margin:0 0 .8rem 0;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;}
.grid2>.card{margin-bottom:0;}
.grid2w{display:grid;grid-template-columns:1.9fr 1fr;gap:1.1rem;margin-bottom:1.1rem;}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:1.1rem;margin-bottom:1.1rem;}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1rem;margin-bottom:1.1rem;}
@media(max-width:980px){.grid2,.grid2w,.grid4,.grid3{grid-template-columns:1fr 1fr;}.grid2w{grid-template-columns:1fr;}}
@media(max-width:640px){.grid2,.grid4,.grid3{grid-template-columns:1fr;}}
/* hero */
.hero{background:var(--navy);color:#fff;border-radius:30px;padding:2.4rem 2.5rem;}
.hero h2{font-family:var(--display);font-weight:800;font-size:2.7rem;letter-spacing:-.05em;line-height:1.02;margin:0 0 1rem 0;color:#fff;}
.hero p{color:#cfcfe6;font-size:1.05rem;line-height:1.6;max-width:56ch;margin:0;}
.hero .big{display:flex;align-items:flex-end;gap:1.2rem;flex-wrap:wrap;margin-top:2rem;}
.hero .num{font-family:var(--display);font-weight:800;font-size:6.2rem;letter-spacing:-.06em;line-height:.85;color:var(--lime);}
.hero .cap{font-family:var(--display);font-weight:700;font-size:1.2rem;line-height:1.3;color:#fff;padding-bottom:.4rem;}
.lime{background:var(--lime);border-radius:30px;padding:2.2rem 2rem;display:flex;flex-direction:column;justify-content:space-between;}
.lime .t{font-family:var(--display);font-weight:700;font-size:1.15rem;}
.lime .num{font-family:var(--display);font-weight:800;font-size:5rem;letter-spacing:-.06em;line-height:.9;margin:1.5rem 0 .6rem 0;}
.lime p{font-size:1.05rem;line-height:1.5;margin:0;}
.tile{background:#fff;border-radius:22px;padding:1.3rem 1.4rem;}
.tile .n{font-family:var(--display);font-weight:800;font-size:2.4rem;letter-spacing:-.05em;color:var(--indigo);line-height:1;}
.tile .l{color:var(--muted);font-size:.93rem;line-height:1.45;margin-top:.5rem;}
/* badges and rows */
.badge{display:inline-block;font-weight:700;font-size:.82rem;padding:.2rem .75rem;border-radius:999px;white-space:nowrap;}
.b-ok{background:var(--indigo);color:#fff;} .b-bad{background:var(--red-soft);color:var(--red);}
.row{display:flex;align-items:center;gap:.9rem;background:#f7f7fb;border-radius:16px;padding:.8rem 1rem;margin-bottom:.6rem;flex-wrap:wrap;}
.row .nm{font-family:var(--display);font-weight:700;font-size:1.05rem;}
.row .ds{color:var(--muted);font-size:.9rem;}
/* flow steps */
.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:.7rem;margin-top:.8rem;}
.step{background:#f7f7fb;border-radius:18px;padding:1rem .9rem;position:relative;font-size:.88rem;line-height:1.4;}
.step b{display:block;font-family:var(--display);font-size:1rem;margin-bottom:.25rem;letter-spacing:-.02em;}
.step.last{background:var(--navy);color:#fff;} .step.last b{color:var(--lime);}
@media(max-width:980px){.flow{grid-template-columns:repeat(2,1fr);}}
/* bars */
.bar{display:grid;grid-template-columns:200px 1fr 62px;gap:.8rem;align-items:center;margin:.38rem 0;font-size:.92rem;}
.bar .track{background:#efeff7;border-radius:999px;height:12px;overflow:hidden;}
.bar .fill{height:100%;border-radius:999px;background:var(--indigo);}
.bar .fill.lime{background:var(--lime);} .bar .fill.navy{background:var(--navy);}
.bar .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);}
@media(max-width:640px){.bar{grid-template-columns:120px 1fr 52px;}}
/* tables */
table.t{width:100%;border-collapse:collapse;font-size:.95rem;}
table.t th{text-align:left;font-family:var(--display);font-weight:700;padding:.65rem .7rem;background:#f1f1f9;}
table.t th:first-child{border-radius:12px 0 0 12px;} table.t th:last-child{border-radius:0 12px 12px 0;}
table.t td{padding:.65rem .7rem;border:0;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;vertical-align:top;}
table.t th{border:0;}
.pair{margin-bottom:.85rem;}
.pair .bar{margin:.12rem 0;}
.tbl-wrap{overflow-x:auto;}
/* confusion matrix */
.cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:.45rem;align-items:stretch;font-size:.85rem;color:var(--muted);}
.cm .cell{border-radius:16px;padding:1.2rem .5rem;text-align:center;font-family:var(--display);font-weight:800;font-size:2rem;color:#fff;}
.cm .hit{background:var(--indigo);} .cm .miss{background:#e6e6f3;color:var(--ink);} .cm .miss.danger{background:var(--red-soft);color:var(--red);}
.cm .ax{display:flex;align-items:center;justify-content:center;text-align:center;}
/* verdict */
.verdict{border-radius:26px;padding:1.7rem 1.6rem;}
.verdict.ok{background:var(--lime);} .verdict.bad{background:var(--navy);color:#fff;}
.verdict .t{font-family:var(--display);font-weight:700;font-size:1.05rem;}
.verdict .big{font-family:var(--display);font-weight:800;font-size:3.4rem;letter-spacing:-.05em;line-height:1;margin:.6rem 0 .3rem 0;}
.verdict.bad .big{color:var(--lime);}
.meter{height:14px;border-radius:999px;background:linear-gradient(90deg,var(--lime) 0%,#f3d24f 50%,var(--red) 100%);position:relative;margin:1.1rem 0 .4rem 0;}
.meter i{position:absolute;top:-5px;width:6px;height:24px;border-radius:4px;background:var(--ink);box-shadow:0 0 0 3px #fff;transform:translateX(-50%);}
.verdict.bad .meter i{box-shadow:0 0 0 3px var(--navy);background:#fff;}
.meter-l{display:flex;justify-content:space-between;font-size:.78rem;opacity:.75;}
.note{background:#fff8e6;border:1px solid #f3e2a8;border-radius:16px;padding:.9rem 1.1rem;font-size:.92rem;line-height:1.55;margin-bottom:1.1rem;}
/* feed */
.feed{display:grid;grid-template-columns:70px 1fr 90px 1fr 34px;gap:.6rem;align-items:center;background:#fff;border-radius:14px;padding:.55rem .9rem;margin-bottom:.4rem;font-size:.92rem;}
.feed .badge{justify-self:start;}
.feed.wrong{background:var(--red-soft);}
.ok-mark{font-weight:800;color:var(--indigo);} .no-mark{font-weight:800;color:var(--red);}
/* streamlit widgets */
.stButton>button,.stDownloadButton>button{border-radius:999px;font-weight:600;padding:.5rem 1.3rem;border:1px solid #cfcfe0;background:#fff;color:var(--ink);}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--indigo);color:var(--indigo);}
.stButton>button[kind="primary"]{background:var(--ink);color:#fff;border-color:var(--ink);}
.stButton>button[kind="primary"]:hover{background:var(--indigo);border-color:var(--indigo);color:#fff;}
[data-testid="stTabs"] button[role="tab"] p{font-weight:600;}
div[data-baseweb="select"]>div{border-radius:14px;background:#fff;}
[data-testid="stFileUploaderDropzone"]{border-radius:20px;background:#fff;}
:focus-visible{outline:2px solid var(--indigo);outline-offset:2px;}
.grid2>*,.grid2w>*,.grid4>*,.grid3>*{min-width:0;}
.hero h2{overflow-wrap:anywhere;}
@media(max-width:640px){
.hero{padding:1.6rem 1.4rem;border-radius:24px;} .hero h2{font-size:1.9rem;} .hero .num{font-size:4rem;}
.lime{padding:1.6rem 1.4rem;border-radius:24px;} .lime .num{font-size:3.6rem;}
.page-head .ph-title{font-size:2rem;} .verdict .big{font-size:2.4rem;}
.feed{grid-template-columns:52px 1fr 48px 1fr 22px;font-size:.8rem;gap:.35rem;padding:.5rem .6rem;}
.card{padding:1.2rem 1.1rem;}
}
</style>

"""


def H(html: str):
    """Render HTML. Leading whitespace and blank lines are stripped so Markdown never turns it into a code block."""
    st.markdown("\n".join(l.strip() for l in html.strip().splitlines() if l.strip()), unsafe_allow_html=True)


def head(title: str, sub: str):
    H(f'<div class="page-head"><div class="ph-title">{title}</div><span>{sub}</span></div>')


def badge(label: int) -> str:
    return f'<span class="badge {"b-ok" if label == 1 else "b-bad"}">{LABEL_NAME[label]}</span>'


def bar(label, value, vmax, text=None, cls=""):
    pct = max(0.0, min(100.0, 100 * value / vmax))
    return (f'<div class="bar"><div>{label}</div><div class="track"><div class="fill {cls}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="v">{text if text is not None else f"{value:.4f}"}</div></div>')


# ----------------------------------------------------------------------------------------
# Data, model and evaluation (cached)
# ----------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


MODEL = load_model()
FEATURES = list(MODEL.feature_names_in_)
CLASSES = list(MODEL.classes_)               # [-1, 1]
IDX_PHISH = CLASSES.index(-1)
assert all(f in META for f in FEATURES), "Feature dictionary is out of sync with the model."


def phish_prob(frame: pd.DataFrame) -> np.ndarray:
    return MODEL.predict_proba(frame[FEATURES])[:, IDX_PHISH]


@st.cache_data(show_spinner="Evaluating the model on the held-out test set…")
def evaluate():
    df = load_data()
    X, y = df[FEATURES], df["Result"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    proba = MODEL.predict_proba(Xte)
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    p, r, f, s = precision_recall_fscore_support(yte, pred, labels=[-1, 1], zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    fpr, tpr, _ = roc_curve(yte, proba[:, CLASSES.index(1)], pos_label=1)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 140).astype(int))
    train_rows = set(map(tuple, Xtr.values))
    overlap = float(np.mean([tuple(v) in train_rows for v in Xte.values]))
    return {
        "Xte": Xte, "yte": yte, "pred": pred, "p_phish": proba[:, IDX_PHISH],
        "acc": accuracy_score(yte, pred), "prec": pm, "rec": rm, "f1": fm,
        "auc": roc_auc_score(yte, proba[:, CLASSES.index(1)]),
        "cm": confusion_matrix(yte, pred, labels=[-1, 1]).tolist(),
        "cls": {"Phishing": (p[0], r[0], f[0], int(s[0])), "Legitimate": (p[1], r[1], f[1], int(s[1]))},
        "roc": (fpr[keep], tpr[keep]), "n": len(yte),
        "dups": int(df[FEATURES + ["Result"]].duplicated().sum()), "overlap": overlap,
    }


EV = evaluate()
DF = load_data()
CM = EV["cm"]
IMP = pd.Series(MODEL.feature_importances_, index=FEATURES).sort_values(ascending=False)


def dup_note():
    if SHOW_DUPLICATE_NOTE:
        H(f'<div class="note"><b>Worth knowing:</b> after the index column is removed, {EV["dups"]:,} of the {len(DF):,} records '
          f'repeat another record\'s 30 feature values, and {EV["overlap"]*100:.1f}% of the test records also appear in the training set. '
          f'Scores on a de-duplicated split would be somewhat lower, so read them as performance on this dataset rather than on brand-new websites.</div>')


# ----------------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------------
def page_overview():
    head("Overview", "Classifying websites as legitimate or phishing from 30 URL and page characteristics")
    miss_rate = CM[0][1] / sum(CM[0])
    false_alarm = CM[1][0] / sum(CM[1])
    errors = CM[0][1] + CM[1][0]
    H(f"""
    <div class="grid2w">
      <div class="hero">
        <h2>Phishing Website Detection Using Random Forest</h2>
        <p>One website's URL and page traits in, one of two labels out. Trained and tested on dataset1.csv from the Phishing Website Detection Datasets collection.</p>
        <div class="big"><div class="num">{EV['acc']*100:.2f}%</div><div class="cap">accuracy on<br>{EV['n']:,} held-out records</div></div>
      </div>
      <div class="lime">
        <div class="t">As a phishing detector</div>
        <div><div class="num">{(1-miss_rate)*100:.2f}%</div>
        <p>of phishing sites caught, with a <b>{false_alarm*100:.2f}%</b> false-alarm rate on legitimate sites.</p></div>
      </div>
    </div>
    <div class="grid4">
      <div class="tile"><div class="n">{EV['f1']:.4f}</div><div class="l">Macro F1-score. Both classes are weighted equally.</div></div>
      <div class="tile"><div class="n">{errors}</div><div class="l">Errors out of {EV['n']:,} test records ({errors/EV['n']*100:.2f}%).</div></div>
      <div class="tile"><div class="n">{EV['auc']:.4f}</div><div class="l">ROC-AUC. Probabilities separate the two classes very well.</div></div>
      <div class="tile"><div class="n">{len(FEATURES)}</div><div class="l">Model inputs, read by {MODEL.n_estimators} decision trees.</div></div>
    </div>
    <div class="grid2">
      <div class="card">
        <h3>The problem</h3>
        <ul>
          <li>Phishing sites imitate real ones to steal usernames, passwords and personal data, so spotting them by eye is hard.</li>
          <li>Blacklists and fixed security rules struggle with newly created or modified phishing sites.</li>
          <li>A classifier can learn patterns from sites already labelled legitimate or phishing, then apply them to new ones.</li>
        </ul>
        <h4>What we built</h4>
        <ul>
          <li>A Random Forest of {MODEL.n_estimators} trees trained on {len(DF):,} labelled website records.</li>
          <li>An evaluation on a stratified 80/20 split: accuracy, precision, recall, F1, confusion matrix and ROC-AUC.</li>
          <li>This app, so you can test a website profile, replay a scan, or classify a whole CSV.</li>
        </ul>
      </div>
      <div class="card">
        <h3>The 2 website classes</h3>
        <p class="hint">The target column, Result, in dataset1.csv</p>
        <div class="row">{badge(1)}<span class="nm">Legitimate</span><span class="ds">Coded 1 · {int((DF.Result==1).sum()):,} records</span></div>
        <div class="row">{badge(-1)}<span class="nm">Phishing</span><span class="ds">Coded -1 · {int((DF.Result==-1).sum()):,} records</span></div>
        <h4>Strongest signals</h4>
        <p class="hint">Share of the model's decisions explained by each feature</p>
        {"".join(bar(META[f][0], v, IMP.iloc[0], f"{v*100:.1f}%") for f, v in IMP.head(5).items())}
      </div>
    </div>
    """)


def page_background():
    head("Background of the Study", "Why phishing detection, and what this project sets out to do")
    H("""
    <div class="card">
      <h3>Background</h3>
      <p>Phishing is a common cybersecurity threat in which attackers create deceptive websites or links designed to trick users into providing sensitive information such as usernames, passwords and other personal information. Because phishing websites can be made to resemble legitimate ones, identifying them manually can be difficult.</p>
      <p>Traditional approaches rely on security rules, blacklists or manually identified indicators, which can fall short against newly created or modified phishing websites. Machine learning offers an alternative: a model learns patterns from websites already identified as legitimate or phishing.</p>
      <p>This project proposes a phishing website detection system using Random Forest classification. The model analyzes characteristics of websites and their URLs and classifies each as either legitimate or phishing.</p>
      <p>It uses the Phishing Website Detection Datasets collection on Kaggle. The file dataset1.csv holds 11,055 samples and 32 columns focused on traditional phishing indicators, and its target variable, Result, records whether the website is legitimate or phishing.</p>
    </div>
    <div class="card" style="background:#1b1936;color:#fff;">
      <h3 style="color:#d4f25a;">General objective</h3>
      <p style="color:#e6e6f5;">To develop a Random Forest-based machine learning model for detecting phishing websites using website and URL-related characteristics.</p>
    </div>
    <div class="card">
      <h3>Specific objectives</h3>
      <div class="row"><span class="badge b-ok">1</span><span>Obtain, examine and preprocess the dataset by identifying and handling missing values, duplicate records and inconsistent data.</span></div>
      <div class="row"><span class="badge b-ok">2</span><span>Transform and engineer website and URL features into a format suitable for machine learning, and identify characteristics that distinguish phishing from legitimate websites.</span></div>
      <div class="row"><span class="badge b-ok">3</span><span>Divide the prepared data into training and testing subsets and develop a Random Forest model using Python and scikit-learn.</span></div>
      <div class="row"><span class="badge b-ok">4</span><span>Evaluate the model using accuracy, precision, recall, F1-score and a confusion matrix.</span></div>
      <div class="row"><span class="badge b-ok">5</span><span>Analyze feature importance and determine each feature's contribution to phishing website detection.</span></div>
    </div>
    <div class="grid2">
      <div class="card">
        <h3>Scope</h3>
        <ul>
          <li>Uses dataset1.csv from the Kaggle collection: 11,055 samples and 32 columns.</li>
          <li>All features are considered during preprocessing, then feature selection and engineering identify the most relevant ones.</li>
          <li>Binary classification: each sample is legitimate or phishing.</li>
          <li>Built with Python and scikit-learn, evaluated with accuracy, precision, recall, F1-score and a confusion matrix.</li>
        </ul>
      </div>
      <div class="card">
        <h3>Limitations</h3>
        <ul>
          <li>Analyzes the URL and website characteristics in the dataset, not full page content, visual appearance or live behavior.</li>
          <li>Limited to the features and samples available in the selected dataset.</li>
          <li>No real-time website scanning, web crawling or analysis of live webpage content.</li>
          <li>The app takes the 30 indicators as inputs. It does not extract them from a typed URL.</li>
        </ul>
      </div>
    </div>
    """)


def page_dataset():
    head("Dataset & Pipeline", "From dataset1.csv to a tuned Random Forest")
    H(f"""
    <div class="grid4">
      <div class="tile"><div class="n">{len(DF):,}</div><div class="l">Samples (websites) in dataset1.csv</div></div>
      <div class="tile"><div class="n">32</div><div class="l">Columns: index, 30 predictors and the target</div></div>
      <div class="tile"><div class="n">{int((DF.Result==-1).sum()):,}</div><div class="l">Phishing records (coded -1)</div></div>
      <div class="tile"><div class="n">{int((DF.Result==1).sum()):,}</div><div class="l">Legitimate records (coded 1)</div></div>
    </div>
    <div class="card">
      <h3>Conceptual framework</h3>
      <p class="hint">The workflow followed in the notebook, from loading the dataset to evaluating the initial and tuned models.</p>
      <div class="flow">
        <div class="step"><b>dataset1.csv</b>11,055 rows × 32 columns</div>
        <div class="step"><b>Data cleaning</b>Missing values and duplicates checked, index column dropped</div>
        <div class="step"><b>Stratified split</b>80% train, 20% test</div>
        <div class="step"><b>Baseline RF</b>Initial Random Forest (200 trees)</div>
        <div class="step"><b>Tuned RF</b>Randomized search, 5-fold cross-validation</div>
        <div class="step last"><b>Evaluation</b>Metrics, confusion matrix, ROC, feature importance</div>
      </div>
    </div>
    <div class="grid2">
      <div class="card"><h3>Data cleaning</h3>
        <p>The dataset was examined for missing values and duplicate records before model development. It contained 11,055 samples and 32 columns and no missing values were found. The index column was removed because it was only a record identifier. The remaining data has 30 predictor features and one target variable, Result.</p></div>
      <div class="card"><h3>Data transformation</h3>
        <p>The website and URL characteristics were already numeric, coded as -1, 0 and 1. Categorical encoding and feature scaling were therefore not required, and the numeric codes were kept for the Random Forest.</p></div>
      <div class="card"><h3>Feature selection</h3>
        <p>Random Forest feature importance, calculated on the training set, was used to identify the most relevant features for phishing classification. Features meeting the chosen importance threshold were retained for the final model.</p></div>
      <div class="card"><h3>Feature engineering</h3>
        <p>Two aggregate indicators were derived from the existing characteristics: <b>negative_indicator_count</b> and <b>zero_indicator_count</b>, the number of -1 and 0 values in each website record.</p>
        <p class="hint">The deployed model file reads the {len(FEATURES)} original indicators, so this app does not ask for the two derived counts.</p></div>
    </div>
    """)
    mp = MODEL.get_params()
    H(f"""
    <div class="card"><h3>The tuned model</h3>
      <p class="hint">Read straight from tuned_random_forest.skl</p>
      <div class="grid4" style="margin:0">
        <div class="tile"><div class="n">{mp['n_estimators']}</div><div class="l">Trees (n_estimators)</div></div>
        <div class="tile"><div class="n">{mp['max_features']}</div><div class="l">Features tried at each split (max_features)</div></div>
        <div class="tile"><div class="n">{mp['class_weight']}</div><div class="l">Class weighting (class_weight)</div></div>
        <div class="tile"><div class="n">{mp['random_state']}</div><div class="l">Random seed (random_state)</div></div>
      </div>
    </div>
    <div class="card"><h3>The 30 predictors</h3><p class="hint">Each indicator is coded -1, 0 or 1. Expand a group to see what each code means.</p></div>
    """)
    for g in GROUPS:
        with st.expander(g):
            rows = "".join(
                f'<tr><td><b>{META[f][0]}</b><br><span style="color:#6b6a80;font-size:.82rem">{f}</span></td>'
                f'<td>{"<br>".join(f"<b>{v}</b>&nbsp; {t}" for v, t in sorted(META[f][2].items(), reverse=True))}</td></tr>'
                for f in FEATURES if META[f][1] == g)
            H(f'<div class="tbl-wrap"><table class="t"><tr><th>Feature</th><th>Coded values</th></tr>{rows}</table></div>')
    with st.expander("Preview the dataset"):
        st.dataframe(DF.head(100), width="stretch", hide_index=True)
    dup_note()


def confusion_html(cm, title):
    return f"""
    <div><h4 style="margin-top:0">{title}</h4>
    <div class="cm">
      <div></div><div class="ax">Predicted phishing</div><div class="ax">Predicted legitimate</div>
      <div class="ax">True<br>phishing</div><div class="cell hit">{cm[0][0]}</div><div class="cell miss danger">{cm[0][1]}</div>
      <div class="ax">True<br>legitimate</div><div class="cell miss">{cm[1][0]}</div><div class="cell hit">{cm[1][1]}</div>
    </div></div>"""


def roc_svg():
    fpr, tpr = EV["roc"]
    W, Hh, L, T, R, B = 380, 320, 46, 14, 14, 44
    px = lambda x: L + x * (W - L - R)
    py = lambda y: T + (1 - y) * (Hh - T - B)
    pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(fpr, tpr))
    grid = "".join(f'<line x1="{px(t)}" y1="{py(0)}" x2="{px(t)}" y2="{py(1)}" stroke="#ececf4"/><line x1="{px(0)}" y1="{py(t)}" x2="{px(1)}" y2="{py(t)}" stroke="#ececf4"/>'
                   f'<text x="{px(t)}" y="{py(0)+18}" font-size="11" text-anchor="middle" fill="#6b6a80">{t:g}</text>'
                   f'<text x="{L-8}" y="{py(t)+4}" font-size="11" text-anchor="end" fill="#6b6a80">{t:g}</text>' for t in (0, .25, .5, .75, 1))
    return (f'<svg viewBox="0 0 {W} {Hh}" width="100%" role="img" aria-label="ROC curve of the tuned model, AUC {EV["auc"]:.4f}">{grid}'
            f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" stroke="#b9b8cc" stroke-dasharray="5 5"/>'
            f'<polyline points="{pts}" fill="none" stroke="#4b3fd1" stroke-width="3" stroke-linejoin="round"/>'
            f'<text x="{(L+W-R)/2}" y="{Hh-6}" font-size="12" text-anchor="middle" fill="#6b6a80">False positive rate</text>'
            f'<text transform="translate(12 {(T+Hh-B)/2}) rotate(-90)" font-size="12" text-anchor="middle" fill="#6b6a80">True positive rate</text>'
            f'<text x="{px(.97)}" y="{py(.12)}" font-size="14" font-weight="700" text-anchor="end" fill="#16142b">AUC = {EV["auc"]:.4f}</text></svg>')


def page_results():
    head("Model Results", f"Both models evaluated on the same {EV['n']:,} held-out records")
    ph, lg = EV["cls"]["Phishing"], EV["cls"]["Legitimate"]
    tuned = {"Accuracy": EV["acc"], "Precision": EV["prec"], "Recall": EV["rec"], "F1-score": EV["f1"], "ROC-AUC": EV["auc"]}
    n_ph, n_lg = sum(CM[0]), sum(CM[1])
    H(f"""
    <div class="card"><h3>Overall performance</h3>
      <p class="hint">Precision, recall and F1-score are macro averages, which give each class equal weight. The tuned row is calculated live from the model file, and the initial row is from the report.</p>
      <div class="tbl-wrap"><table class="t"><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>ROC-AUC</th></tr>
      <tr><td><b>Initial Random Forest</b></td>{"".join(f"<td>{v:.4f}</td>" for v in INITIAL.values())}</tr>
      <tr><td><b>Tuned Random Forest</b></td>{"".join(f"<td>{v:.4f}</td>" for v in tuned.values())}</tr></table></div>
      <h4>Metric comparison</h4>
      {"".join('<div class="pair">' + bar(k, INITIAL[k], 1, f"{INITIAL[k]:.4f}", "navy") + bar("", tuned[k], 1, f"{tuned[k]:.4f}") + "</div>" for k in INITIAL)}
      <p class="hint" style="margin-top:.8rem">Dark bar: initial model. Purple bar: tuned model.</p>
      <p>The two models performed almost identically. The initial model classified 2,156 of 2,211 test records correctly and the tuned model 2,155, a difference of a single record. Tuning did not improve on the baseline, which suggests the default-style Random Forest was already close to the best this feature set can achieve.</p>
    </div>
    <div class="card"><h3>Per-class performance</h3>
      <div class="tbl-wrap"><table class="t"><tr><th>Model / class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>
      <tr><td>Initial · Phishing (-1)</td><td>{INITIAL_CLASS['Phishing'][0]:.4f}</td><td>{INITIAL_CLASS['Phishing'][1]:.4f}</td><td>{INITIAL_CLASS['Phishing'][2]:.2f}</td><td>{INITIAL_CLASS['Phishing'][3]:,}</td></tr>
      <tr><td>Initial · Legitimate (1)</td><td>{INITIAL_CLASS['Legitimate'][0]:.4f}</td><td>{INITIAL_CLASS['Legitimate'][1]:.4f}</td><td>{INITIAL_CLASS['Legitimate'][2]:.2f}</td><td>{INITIAL_CLASS['Legitimate'][3]:,}</td></tr>
      <tr><td><b>Tuned · Phishing (-1)</b></td><td>{ph[0]:.4f}</td><td>{ph[1]:.4f}</td><td>{ph[2]:.2f}</td><td>{ph[3]:,}</td></tr>
      <tr><td><b>Tuned · Legitimate (1)</b></td><td>{lg[0]:.4f}</td><td>{lg[1]:.4f}</td><td>{lg[2]:.2f}</td><td>{lg[3]:,}</td></tr></table></div>
    </div>
    <div class="grid2">
      <div class="card">{confusion_html(INITIAL_CM, "Initial Random Forest")}</div>
      <div class="card">{confusion_html(CM, "Tuned Random Forest")}</div>
    </div>
    <div class="card" style="margin-top:1.1rem">
      <p>The initial model classified 946 phishing and 1,210 legitimate sites correctly, missed 34 phishing sites and wrongly flagged 21 legitimate ones. The tuned model got {CM[0][0]} and {CM[1][1]} right, missed <b>{CM[0][1]}</b> phishing sites and raised <b>{CM[1][0]}</b> false alarms.</p>
      <p>In a security setting a missed phishing site is the more costly error, because users may be exposed to the threat. For both models these misses were around {CM[0][1]/n_ph*100:.1f}% of phishing sites, slightly higher than the false-alarm rate on legitimate sites ({CM[1][0]/n_lg*100:.1f}%).</p>
    </div>
    <div class="grid2" style="margin-top:1.1rem;margin-bottom:1.1rem">
      <div class="card"><h3>ROC curve</h3><p class="hint">Tuned model, positive label = 1</p>{roc_svg()}
        <p>Both models reached a ROC-AUC of 0.9978, so predicted probabilities separate legitimate and phishing websites very well across thresholds.</p></div>
      <div class="card"><h3>Feature importance</h3><p class="hint">Top 15 of {len(FEATURES)}, tuned model (live from the file)</p>
        {"".join(bar(META[f][0], v, IMP.iloc[0], f"{v:.4f}") for f, v in IMP.head(15).items())}</div>
    </div>
    <div class="card"><h3>Initial vs tuned importance</h3><p class="hint">Top eight features by Random Forest importance (report, Table 4)</p>
      <div class="tbl-wrap"><table class="t"><tr><th>Feature</th><th>Initial importance</th><th>Tuned importance</th></tr>
      {"".join(f"<tr><td>{f}</td><td>{a:.4f}</td><td>{b:.4f}</td></tr>" for f, a, b in IMPORTANCE_TABLE)}</table></div>
      <p style="margin-top:1rem">The rankings are nearly identical in both models. SSLfinal_State (the SSL certificate status) and URL_of_Anchor (the proportion of anchor links pointing to other domains) are by far the most influential, together accounting for about 55% of total importance. They are followed by web_traffic, having_Sub_Domain, Prefix_Suffix and Links_in_tags. Feature importance shows how much the model relies on a feature. It does not prove the feature causes a site to be phishing.</p>
    </div>
    """)
    dup_note()


# ---- Check a website --------------------------------------------------------------------
def ss_key(f):
    return f"f_{f}"


def load_row(row: pd.Series, actual=None):
    for f in FEATURES:
        st.session_state[ss_key(f)] = int(row[f])
    st.session_state["actual"] = actual


def pick(label=None):
    Xte, yte = EV["Xte"], EV["yte"]
    pool = Xte if label is None else Xte[yte == label]
    i = random.choice(list(pool.index))
    load_row(Xte.loc[i], int(yte.loc[i]))


def clear_actual():
    st.session_state["actual"] = None


def sensitivity(row: dict, base: float):
    variants, tags = [], []
    for f in FEATURES:
        for v in META[f][2]:
            if v != row[f]:
                variants.append({**row, f: v})
                tags.append((f, v))
    probs = phish_prob(pd.DataFrame(variants))
    best = {}
    for (f, v), p in zip(tags, probs):
        d = p - base
        if f not in best or abs(d) > abs(best[f][2]):
            best[f] = (v, p, d)
    return sorted(best.items(), key=lambda kv: -abs(kv[1][2]))[:5]


def page_check():
    head("Check a Website", "Describe a website with the 30 indicators and see what the model says")
    if "init" not in st.session_state:
        st.session_state["init"] = True
        pick(1)
    c1, c2, c3, _ = st.columns([1.1, 1.1, 1.1, 2])
    c1.button("Legitimate example", on_click=pick, args=(1,), width="stretch")
    c2.button("Phishing example", on_click=pick, args=(-1,), width="stretch")
    c3.button("Random record", on_click=pick, args=(None,), width="stretch")
    st.caption("Examples are real records from the held-out test set. You can change any indicator afterwards.")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        tabs = st.tabs(GROUPS)
        for tab, g in zip(tabs, GROUPS):
            with tab:
                fs = [f for f in FEATURES if META[f][1] == g]
                cols = st.columns(2)
                for i, f in enumerate(fs):
                    opts = sorted(META[f][2], reverse=True)
                    cols[i % 2].selectbox(META[f][0], opts, key=ss_key(f), on_change=clear_actual,
                                          format_func=lambda v, f=f: META[f][2][v])
    row = {f: int(st.session_state[ss_key(f)]) for f in FEATURES}
    p = float(phish_prob(pd.DataFrame([row]))[0])
    is_phish = p >= 0.5
    conf = p if is_phish else 1 - p
    with right:
        title = "Likely phishing" if is_phish else "Likely legitimate"
        sub = "This profile looks like the phishing sites the model learned from." if is_phish else "This profile looks like the legitimate sites the model learned from."
        H(f"""
        <div class="verdict {'bad' if is_phish else 'ok'}">
          <div class="t">Verdict</div>
          <div class="big">{title}</div>
          <div>{sub}</div>
          <div class="meter"><i style="left:{p*100:.1f}%"></i></div>
          <div class="meter-l"><span>Legitimate</span><span>Phishing probability {p*100:.1f}%</span></div>
          <div style="margin-top:.9rem;font-size:.9rem;opacity:.85">Model confidence in this label: {conf*100:.1f}%</div>
        </div>""")
        actual = st.session_state.get("actual")
        if actual is not None:
            ok = (actual == -1) == is_phish
            H(f'<div class="card" style="margin-top:1rem;padding:1rem 1.2rem">Recorded label for this example: {badge(actual)} '
              f'<b style="color:{"#4b3fd1" if ok else "#d6336c"}">{"The model agrees." if ok else "The model disagrees."}</b></div>')
        top = sensitivity(row, p)
        items = ""
        for f, (v, newp, d) in top:
            if abs(d) < 0.01:
                continue
            items += (f'<li><b>{META[f][0]}</b>: switching to "{META[f][2][v]}" would move the phishing probability '
                      f'from {p*100:.0f}% to {newp*100:.0f}%.</li>')
        H(f'<div class="card" style="margin-top:1rem"><h3>What would change the answer</h3>'
          f'<p class="hint">Each indicator is changed on its own while the others stay fixed.</p>'
          f'<ul>{items or "<li>No single change moves the probability by more than 1 point, so this verdict is stable.</li>"}</ul></div>')


# ---- Live scan simulation ---------------------------------------------------------------
def page_sim():
    head("Live Scan Simulation", "Replay held-out websites through the model one at a time")
    c1, c2, c3 = st.columns([1, 1, 1])
    n = c1.slider("Websites to scan", 10, 100, 30, step=5)
    speed = c2.select_slider("Speed", ["Slow", "Normal", "Fast"], value="Normal")
    c3.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    go = c3.button("Start scan", type="primary", width="stretch")
    delay = {"Slow": 0.9, "Normal": 0.4, "Fast": 0.08}[speed]
    stats_box, prog_box, feed_box = st.empty(), st.empty(), st.empty()

    def stats(done, flagged, passed, correct):
        acc = f"{correct/done*100:.0f}%" if done else "-"
        stats_box.markdown(
            f'<div class="grid4" style="margin-bottom:1rem"><div class="tile"><div class="n">{done}</div><div class="l">Scanned</div></div>'
            f'<div class="tile"><div class="n" style="color:#d6336c">{flagged}</div><div class="l">Flagged as phishing</div></div>'
            f'<div class="tile"><div class="n">{passed}</div><div class="l">Passed as legitimate</div></div>'
            f'<div class="tile"><div class="n">{acc}</div><div class="l">Matched the recorded label</div></div></div>', unsafe_allow_html=True)

    if not go:
        stats(0, 0, 0, 0)
        feed_box.markdown('<div class="card"><p>Choose how many websites to scan and press <b>Start scan</b>. Each record is a real, held-out example, so you can watch where the model is right and where it slips.</p></div>', unsafe_allow_html=True)
        return
    Xte, yte = EV["Xte"], EV["yte"]
    sample = Xte.sample(n, random_state=int(time.time()) % 100000)
    probs = phish_prob(sample)
    done = flagged = passed = correct = 0
    lines = []
    for (idx, _), pp in zip(sample.iterrows(), probs):
        pred = -1 if pp >= 0.5 else 1
        actual = int(yte.loc[idx])
        done += 1
        flagged += pred == -1
        passed += pred == 1
        correct += pred == actual
        conf = pp if pred == -1 else 1 - pp
        lines.insert(0, f'<div class="feed {"" if pred == actual else "wrong"}"><span>#{idx+1}</span>{badge(pred)}'
                        f'<span>{conf*100:.0f}%</span><span>Recorded: {badge(actual)}</span>'
                        f'<span class="{"ok-mark" if pred == actual else "no-mark"}">{"✓" if pred == actual else "✗"}</span></div>')
        stats(done, flagged, passed, correct)
        prog_box.progress(done / n, text=f"Scanning {done} of {n}")
        feed_box.markdown("".join(lines[:9]), unsafe_allow_html=True)
        time.sleep(delay)
    prog_box.empty()
    st.success(f"Scan complete: {correct} of {n} predictions matched the recorded label ({correct/n*100:.1f}%).")


# ---- Batch prediction -------------------------------------------------------------------
ALIASES = {"having_IP_Address": "having_IPhaving_IP_Address", "URL_Length": "URLURL_Length"}


def page_batch():
    head("Batch Prediction", "Upload a CSV of websites and classify them all at once")
    H(f'<div class="card"><h3>What the file needs</h3><p>One row per website with the {len(FEATURES)} indicator columns, coded -1, 0 or 1. '
      'Extra columns such as <b>index</b> are ignored. If a <b>Result</b> column is present (1 legitimate, -1 phishing), the app also reports how often the predictions match it.</p></div>')
    sample = EV["Xte"].head(30).copy()
    sample["Result"] = EV["yte"].head(30)
    st.download_button("Download a sample file", sample.to_csv(index=False).encode(), "sample_websites.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.info("Upload a CSV to see predictions. The sample file above works as a quick test.")
        return
    try:
        data = pd.read_csv(up)
    except Exception as e:
        st.error(f"The file could not be read as a CSV: {e}")
        return
    data.columns = [str(c).strip() for c in data.columns]
    data = data.rename(columns={k: v for k, v in ALIASES.items() if k in data.columns and v not in data.columns})
    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        st.error("These required columns are missing: " + ", ".join(missing))
        return
    feats = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad = feats.isna().any(axis=1) | ~feats.isin([-1, 0, 1]).all(axis=1)
    if bad.all():
        st.error("No valid rows found. Indicator values must be -1, 0 or 1.")
        return
    if bad.any():
        st.warning(f"{int(bad.sum())} row(s) have values outside -1, 0, 1 and were skipped.")
    ok = ~bad
    pp = phish_prob(feats[ok].astype(int))
    out = data[ok].copy()
    out["Prediction"] = np.where(pp >= 0.5, "Phishing", "Legitimate")
    out["Phishing_probability"] = pp.round(4)
    n, nph = len(out), int((pp >= 0.5).sum())
    extra = ""
    if "Result" in out.columns:
        actual = pd.to_numeric(out["Result"], errors="coerce")
        match = float(((actual == -1) == (pp >= 0.5))[actual.isin([-1, 1])].mean())
        extra = f'<div class="tile"><div class="n">{match*100:.1f}%</div><div class="l">Predictions matching the Result column</div></div>'
    H(f'<div class="grid4"><div class="tile"><div class="n">{n:,}</div><div class="l">Websites classified</div></div>'
      f'<div class="tile"><div class="n" style="color:#d6336c">{nph:,}</div><div class="l">Flagged as phishing</div></div>'
      f'<div class="tile"><div class="n">{n-nph:,}</div><div class="l">Classified legitimate</div></div>{extra}</div>')
    show = out.copy()
    cols = ["Prediction", "Phishing_probability"] + [c for c in show.columns if c not in ("Prediction", "Phishing_probability")]
    st.dataframe(show[cols], width="stretch", hide_index=True,
                 column_config={"Phishing_probability": st.column_config.ProgressColumn("Phishing probability", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv", type="primary")


# ---- Conclusion -------------------------------------------------------------------------
def page_conclusion():
    head("Conclusion & References", "What the project found and where it can go next")
    H(f"""
    <div class="card"><h3>Conclusion</h3>
      <p>A Random Forest trained on 30 website and URL characteristics separated legitimate and phishing websites very well on the held-out test set: {EV['acc']*100:.2f}% accuracy, a macro F1-score of {EV['f1']:.4f} and a ROC-AUC of {EV['auc']:.4f}. Tuning matched the baseline rather than beating it.</p>
      <p>SSL certificate status and the share of anchor links pointing to other domains drove about 55% of the model's decisions, followed by web traffic, sub-domains, hyphens in the domain and links in tags. Missed phishing sites ({CM[0][1]/sum(CM[0])*100:.1f}% of phishing records) are the costlier error, so that is the number to improve first.</p></div>
    <div class="card"><h3>Recommendations</h3>
      <ul><li>Test additional datasets, ideally with newer phishing sites and de-duplicated records.</li>
      <li>Update and extend the features so the model keeps up with changing phishing techniques.</li>
      <li>Develop real-time detection that extracts the indicators from a live URL instead of asking for them.</li></ul></div>
    <div class="card"><h3>References</h3>
      <ul style="font-size:.93rem">
        <li>L. Breiman, "Random forests," <i>Machine Learning</i>, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.</li>
        <li>M. S. Islam Ovi, "Phishing Website Detection Datasets," <i>Kaggle</i>. <a href="https://www.kaggle.com/datasets/mdsultanulislamovi/phishing-website-detection-datasets" target="_blank">kaggle.com/datasets/mdsultanulislamovi/phishing-website-detection-datasets</a></li>
        <li>Scikit-learn Developers, "RandomForestClassifier," <i>Scikit-learn</i>. <a href="https://scikit-learn.org/" target="_blank">scikit-learn.org</a></li>
        <li>AlEroud, A., &amp; Karabiyik, U. (2021). Towards benchmark datasets for machine learning based website phishing detection: An experimental study. <i>Engineering Applications of Artificial Intelligence, 104</i>, 104347.</li>
        <li>Aljabri, M., et al. (2021). URL phishing detection using machine learning techniques based on URLs lexical analysis. <i>2021 12th International Conference on Information and Communication Systems (ICICS)</i>.</li>
        <li>Gupta, B. B., Yadav, K., Razzak, I., Psannis, K. E., Castiglione, A., &amp; Chang, X. (2021). A novel approach for phishing URLs detection using lexical based machine learning in a real-time environment. <i>Computer Communications, 175</i>, 47–57.</li>
        <li>Lokesh, G. H., et al. (2020). Phishing website detection based on effective machine learning approach. <i>Journal of Cyber Security Technology, 5</i>(1).</li>
        <li>Safi, A., &amp; Singh, S. (2023). A systematic literature review on phishing website detection techniques. <i>Journal of King Saud University – Computer and Information Sciences, 35</i>(2), 590–611.</li>
        <li>Sinha, D., &amp; Sandeep, A. (2020). Phishing website URL detection using machine learning. <i>International Journal of Advanced Science and Technology, 29</i>(7), 2495–2504.</li></ul></div>
    <div class="card"><h3>Appendix</h3>
      <p>Notebook and project files: <a href="https://github.com/Macky091/CST9L-FinalExam" target="_blank">github.com/Macky091/CST9L-FinalExam</a></p></div>
    """)


# ----------------------------------------------------------------------------------------
# Layout and routing
# ----------------------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
PAGES = {
    "Overview": page_overview,
    "Background of the Study": page_background,
    "Dataset & Pipeline": page_dataset,
    "Model Results": page_results,
    "Check a Website": page_check,
    "Live Scan Simulation": page_sim,
    "Batch Prediction": page_batch,
    "Conclusion & References": page_conclusion,
}
with st.sidebar:
    H('<div class="side-title">Phishing Detector</div><div class="side-sub">Website classification · Random Forest</div>')
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    H(f"""
    <div class="side-card"><b>Phishing Website Detection Using Random Forest Classification</b>
    Mc Gabriel C. Limbojan<br>CCE105 · University of Mindanao<br>College of Computing Education<br>S.Y. 2026–2027</div>
    <div class="side-foot">{MODEL.n_estimators} trees · {len(FEATURES)} features · seed {MODEL.random_state}</div>""")

PAGES[choice]()
