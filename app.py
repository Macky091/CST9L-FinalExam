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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--midnight:#0f1a3a;--ink:#16203d;--cobalt:#2f4bff;--cobalt-soft:#e6eaff;--apricot:#ffb27a;--paper:#f4efe6;--card:#fffdf9;
--soft:#f3eee4;--line:#e6dfd1;--muted:#5d6580;--ok:#138a5b;--ok-soft:#d9f1e5;--red:#d1344f;--red-soft:#fde8ec;
--display:'Fraunces',Georgia,serif;--body:'Manrope',system-ui,sans-serif;}
html,body,[class*="css"],.stApp{font-family:var(--body);color:var(--ink);color-scheme:light;}
.stApp{background:var(--paper);}
.stApp p,.stApp label,.stApp li,[data-testid="stWidgetLabel"] p{color:var(--ink);}
#MainMenu{visibility:visible;} footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1180px;padding-top:2.4rem;padding-bottom:4rem;}

/* ---------- sidebar: dark rail with numbered nav ---------- */
[data-testid="stSidebar"]{background:var(--midnight);border-right:0;color-scheme:dark;}
[data-testid="stSidebar"]>div:first-child{width:310px;}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarUserContent"]{padding-top:2rem;}
[data-testid="stSidebar"] *{color:#c9d1ee;}
[data-testid="stSidebarCollapseButton"] *,[data-testid="stSidebarCollapsedControl"] *{color:#fff;}
[data-testid="stSidebar"] .brand{display:flex;align-items:center;gap:.8rem;margin-bottom:.4rem;}
[data-testid="stSidebar"] .brand .mark{width:42px;height:42px;border-radius:12px;background:var(--apricot);display:flex;align-items:center;justify-content:center;font-size:1.3rem;flex:none;}
[data-testid="stSidebar"] .side-title{font-family:var(--display);font-weight:600;font-size:1.45rem;letter-spacing:-.01em;line-height:1.1;color:#fff;}
[data-testid="stSidebar"] .side-sub{color:#8f9ac4;font-size:.8rem;margin:.9rem 0 1.4rem 0;padding-bottom:1.1rem;border-bottom:1px solid rgba(255,255,255,.12);letter-spacing:.02em;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.15rem;counter-reset:nav;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]{background:transparent;border:0;border-radius:0 10px 10px 0;
padding:.62rem 1rem .62rem 1rem;margin:0;width:100%;counter-increment:nav;transition:background .15s;box-shadow:inset 3px 0 0 transparent;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]{display:flex!important;flex-direction:row!important;flex-wrap:nowrap!important;align-items:center!important;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] div:not(:has(p)){display:none!important;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]::before{content:counter(nav,decimal-leading-zero);font-family:var(--display);font-size:.8rem;color:#6f7cb0;margin-right:.9rem;min-width:1.4rem;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover{background:rgba(255,255,255,.06);}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] p{font-weight:600;font-size:.95rem;color:#b8c1e2!important;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked){background:rgba(255,255,255,.1);box-shadow:inset 3px 0 0 var(--apricot);}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p{color:#fff!important;}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked)::before{color:var(--apricot);}
/* credit block */
[data-testid="stSidebar"] .colo{margin-top:1.6rem;border-top:1px solid rgba(255,255,255,.12);padding-top:1.3rem;}
[data-testid="stSidebar"] .colo .co-kicker{color:var(--apricot);text-transform:uppercase;letter-spacing:.14em;font-size:.66rem;font-weight:800;margin-bottom:.55rem;}
[data-testid="stSidebar"] .colo .co-title{font-family:var(--display);color:#fff;font-size:1.05rem;line-height:1.3;font-weight:600;margin-bottom:1rem;}
[data-testid="stSidebar"] .colo .co-author{display:flex;gap:.75rem;align-items:center;margin-bottom:1rem;}
[data-testid="stSidebar"] .colo .co-av{width:38px;height:38px;border-radius:50%;background:var(--cobalt);color:#fff;font-weight:800;font-size:.8rem;display:flex;align-items:center;justify-content:center;flex:none;letter-spacing:.03em;}
[data-testid="stSidebar"] .colo .co-who b{display:block;color:#fff;font-size:.88rem;font-weight:700;}
[data-testid="stSidebar"] .colo .co-who span{display:block;color:#9aa5cf;font-size:.74rem;line-height:1.5;}
[data-testid="stSidebar"] .colo .co-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;}
[data-testid="stSidebar"] .colo .co-stats div{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:.55rem .3rem;text-align:center;}
[data-testid="stSidebar"] .colo .co-stats b{display:block;font-family:var(--display);color:#fff;font-size:1.3rem;line-height:1;font-weight:600;}
[data-testid="stSidebar"] .colo .co-stats span{display:block;color:#8f9ac4;font-size:.64rem;text-transform:uppercase;letter-spacing:.1em;margin-top:.3rem;font-weight:700;}

/* ---------- page head ---------- */
.page-head{display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;margin-bottom:1.3rem;padding-bottom:1rem;border-bottom:2px solid var(--ink);}
.page-head .ph-title{font-family:var(--display);font-weight:600;font-size:2.6rem;letter-spacing:-.02em;margin:0;padding:0;line-height:1.05;color:var(--midnight);}
.page-head span{color:var(--muted);font-size:1rem;}

/* ---------- cards ---------- */
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.6rem 1.7rem;margin-bottom:1.1rem;color:var(--ink);}
.card h3{font-family:var(--display);font-weight:600;font-size:1.4rem;letter-spacing:-.01em;margin:0 0 .35rem 0;color:var(--midnight);}
.card h4{font-family:var(--display);font-weight:600;font-size:1.1rem;margin:1.2rem 0 .3rem 0;color:var(--midnight);}
.card p,.card li{font-size:1rem;line-height:1.7;max-width:75ch;color:#38425f;}
.card ul{padding-left:1.15rem;margin:.4rem 0 0 0;}
.card li{margin-bottom:.45rem;}
.card .hint{color:var(--muted);font-size:.9rem;margin:0 0 .8rem 0;}
.card a{color:var(--cobalt);}
.card.dark{background:var(--midnight);border-color:var(--midnight);}
.card.dark h3{color:var(--apricot);} .card.dark p{color:#dbe1f7;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;}
.grid2>.card{margin-bottom:0;}
.grid2w{display:grid;grid-template-columns:1.9fr 1fr;gap:1.1rem;margin-bottom:1.1rem;}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:1.1rem;margin-bottom:1.1rem;}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1rem;margin-bottom:1.1rem;}
@media(max-width:980px){.grid2,.grid2w,.grid4,.grid3{grid-template-columns:1fr 1fr;}.grid2w{grid-template-columns:1fr;}}
@media(max-width:640px){.grid2,.grid4,.grid3{grid-template-columns:1fr;}}

/* ---------- hero ---------- */
.hero{background:linear-gradient(135deg,#0f1a3a 0%,#1f2f7a 100%);color:#fff;border-radius:22px;padding:2.4rem 2.5rem;}
.hero h2{font-family:var(--display);font-weight:600;font-size:2.6rem;letter-spacing:-.02em;line-height:1.08;margin:0 0 1rem 0;color:#fff;overflow-wrap:anywhere;}
.hero p{color:#d3dbf7;font-size:1.05rem;line-height:1.6;max-width:56ch;margin:0;}
.hero .big{display:flex;align-items:flex-end;gap:1.2rem;flex-wrap:wrap;margin-top:2rem;}
.hero .num{font-family:var(--display);font-weight:600;font-size:6rem;letter-spacing:-.04em;line-height:.85;color:var(--apricot);}
.hero .cap{font-family:var(--display);font-weight:500;font-size:1.2rem;line-height:1.3;color:#fff;padding-bottom:.4rem;}
.lime{background:linear-gradient(160deg,#ffd9bd,#ffb27a);color:var(--midnight);border-radius:22px;padding:2.2rem 2rem;display:flex;flex-direction:column;justify-content:space-between;}
.lime .t{font-family:var(--display);font-weight:600;font-size:1.2rem;color:var(--midnight);}
.lime .num{font-family:var(--display);font-weight:600;font-size:4.8rem;letter-spacing:-.04em;line-height:.9;margin:1.5rem 0 .6rem 0;color:var(--midnight);}
.lime p{font-size:1.05rem;line-height:1.5;margin:0;color:var(--midnight);}
.tile{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--cobalt);border-radius:14px;padding:1.2rem 1.3rem;}
.tile .n{font-family:var(--display);font-weight:600;font-size:2.3rem;letter-spacing:-.03em;color:var(--cobalt);line-height:1;}
.tile .l{color:var(--muted);font-size:.93rem;line-height:1.45;margin-top:.5rem;}

/* ---------- badges and rows ---------- */
.badge{display:inline-block;font-weight:700;font-size:.82rem;padding:.2rem .75rem;border-radius:6px;white-space:nowrap;}
.b-ok{background:var(--ok);color:#fff;} .b-bad{background:var(--red-soft);color:var(--red);}
.row{display:flex;align-items:center;gap:.9rem;background:var(--soft);border-radius:12px;padding:.8rem 1rem;margin-bottom:.6rem;flex-wrap:wrap;color:var(--ink);}
.row .nm{font-family:var(--display);font-weight:600;font-size:1.05rem;color:var(--midnight);}
.row .ds{color:var(--muted);font-size:.9rem;}

/* ---------- flow steps ---------- */
.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:.7rem;margin-top:.8rem;}
.step{background:var(--soft);border-top:3px solid var(--cobalt);border-radius:10px;padding:1rem .9rem;position:relative;font-size:.88rem;line-height:1.4;color:#38425f;}
.step b{display:block;font-family:var(--display);font-size:1rem;margin-bottom:.25rem;color:var(--midnight);}
.step.last{background:var(--midnight);border-top-color:var(--apricot);color:#dbe1f7;} .step.last b{color:var(--apricot);}
@media(max-width:980px){.flow{grid-template-columns:repeat(2,1fr);}}

/* ---------- bars ---------- */
.bar{display:grid;grid-template-columns:200px 1fr 62px;gap:.8rem;align-items:center;margin:.38rem 0;font-size:.92rem;color:var(--ink);}
.bar .track{background:#ebe5d8;border-radius:4px;height:12px;overflow:hidden;}
.bar .fill{height:100%;border-radius:4px;background:var(--cobalt);}
.bar .fill.lime{background:var(--apricot);} .bar .fill.navy{background:var(--midnight);}
.bar .v{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);}
@media(max-width:640px){.bar{grid-template-columns:120px 1fr 52px;}}

/* ---------- tables ---------- */
table.t{width:100%;border-collapse:collapse;font-size:.95rem;color:var(--ink);}
table.t th{text-align:left;font-family:var(--display);font-weight:600;padding:.65rem .7rem;background:var(--midnight);color:#fff;border:0;}
table.t th:first-child{border-radius:8px 0 0 8px;} table.t th:last-child{border-radius:0 8px 8px 0;}
table.t td{padding:.65rem .7rem;border:0;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;vertical-align:top;color:var(--ink);}
.pair{margin-bottom:.85rem;}
.pair .bar{margin:.12rem 0;}
.tbl-wrap{overflow-x:auto;}

/* ---------- confusion matrix ---------- */
.cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:.45rem;align-items:stretch;font-size:.85rem;color:var(--muted);}
.cm .cell{border-radius:12px;padding:1.2rem .5rem;text-align:center;font-family:var(--display);font-weight:600;font-size:2rem;color:#fff;}
.cm .hit{background:var(--cobalt);} .cm .miss{background:#ebe5d8;color:var(--ink);} .cm .miss.danger{background:var(--red-soft);color:var(--red);}
.cm .ax{display:flex;align-items:center;justify-content:center;text-align:center;}

/* ---------- verdict ---------- */
.verdict{border-radius:20px;padding:1.7rem 1.6rem;}
.verdict.ok{background:var(--ok-soft);color:var(--midnight);} .verdict.bad{background:var(--midnight);color:#fff;}
.verdict .t{font-family:var(--display);font-weight:600;font-size:1.1rem;}
.verdict .big{font-family:var(--display);font-weight:600;font-size:3.3rem;letter-spacing:-.03em;line-height:1;margin:.6rem 0 .3rem 0;}
.verdict.ok .big{color:var(--ok);} .verdict.bad .big{color:#ff8fa3;}
.verdict.ok .t,.verdict.ok div{color:var(--midnight);} .verdict.bad .t,.verdict.bad div{color:#fff;}
.meter{height:14px;border-radius:999px;background:linear-gradient(90deg,#138a5b 0%,#f0b43c 50%,#d1344f 100%);position:relative;margin:1.1rem 0 .4rem 0;}
.meter i{position:absolute;top:-5px;width:6px;height:24px;border-radius:4px;background:var(--midnight);box-shadow:0 0 0 3px #fff;transform:translateX(-50%);}
.verdict.bad .meter i{box-shadow:0 0 0 3px var(--midnight);background:#fff;}
.meter-l{display:flex;justify-content:space-between;font-size:.78rem;opacity:.75;}
.note{background:#fff4dc;border:1px solid #f1d9a0;border-radius:12px;padding:.9rem 1.1rem;font-size:.92rem;line-height:1.55;margin-bottom:1.1rem;color:#4a3a14;}

/* ---------- feed ---------- */
.feed{display:grid;grid-template-columns:70px 1fr 90px 1fr 34px;gap:.6rem;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.55rem .9rem;margin-bottom:.4rem;font-size:.92rem;color:var(--ink);}
.feed .badge{justify-self:start;}
.feed.wrong{background:var(--red-soft);border-color:#f4c3cc;}
.ok-mark{font-weight:800;color:var(--ok);} .no-mark{font-weight:800;color:var(--red);}

/* ---------- streamlit widgets ---------- */
.stButton>button,.stDownloadButton>button{border-radius:8px;font-weight:700;padding:.5rem 1.3rem;border:1.5px solid var(--ink);background:var(--card);color:var(--ink);}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--cobalt);color:var(--cobalt);}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--midnight);color:#fff;border-color:var(--midnight);}
.stButton>button[kind="primary"]:hover,.stDownloadButton>button[kind="primary"]:hover{background:var(--cobalt);border-color:var(--cobalt);color:#fff;}
[data-testid="stTabs"] button[role="tab"] p{font-weight:700;}
div[data-baseweb="select"]>div{border-radius:10px;background:var(--card);}
[data-testid="stFileUploaderDropzone"]{border-radius:14px;background:var(--card);}
:focus-visible{outline:2px solid var(--cobalt);outline-offset:2px;}
.grid2>*,.grid2w>*,.grid4>*,.grid3>*{min-width:0;}
@media(max-width:640px){
.hero{padding:1.6rem 1.4rem;border-radius:18px;} .hero h2{font-size:1.9rem;} .hero .num{font-size:4rem;}
.lime{padding:1.6rem 1.4rem;border-radius:18px;} .lime .num{font-size:3.6rem;}
.page-head .ph-title{font-size:2rem;} .verdict .big{font-size:2.4rem;}
.feed{grid-template-columns:52px 1fr 48px 1fr 22px;font-size:.8rem;gap:.35rem;padding:.5rem .6rem;}
.card{padding:1.2rem 1.1rem;}
}
/* ---------- overview (v2 layout) ---------- */
.ov-banner{display:grid;grid-template-columns:1.6fr 1fr 1fr;background:linear-gradient(120deg,#0f1a3a 0%,#1f2f7a 100%);border-radius:22px;margin-bottom:.9rem;overflow:hidden;}
.ov-intro{padding:1.9rem 2rem;}
.ov-kicker{color:var(--apricot);text-transform:uppercase;letter-spacing:.14em;font-size:.7rem;font-weight:800;margin-bottom:.6rem;}
.ov-intro h2{font-family:var(--display);font-weight:600;font-size:1.9rem;line-height:1.12;letter-spacing:-.02em;color:#fff;margin:0 0 .8rem 0;}
.ov-intro p{color:#d3dbf7;font-size:.95rem;line-height:1.6;margin:0;max-width:52ch;}
.ov-zone{padding:1.9rem 1.6rem;border-left:1px solid rgba(255,255,255,.14);display:flex;flex-direction:column;justify-content:center;}
.ov-num{font-family:var(--display);font-weight:600;font-size:3.4rem;letter-spacing:-.04em;line-height:1;color:#fff;}
.ov-num.apri{color:var(--apricot);}
.ov-cap{color:#cdd6f5;font-size:.9rem;line-height:1.45;margin-top:.7rem;} .ov-cap b{color:#fff;}
.strip{display:grid;grid-template-columns:repeat(4,1fr);background:var(--card);border:1px solid var(--line);border-radius:16px;margin-bottom:.9rem;}
.strip .cell{padding:1rem 1.3rem;border-left:1px solid var(--line);}
.strip .cell:first-child{border-left:0;}
.strip .n{font-family:var(--display);font-weight:600;font-size:1.7rem;letter-spacing:-.02em;color:var(--cobalt);line-height:1.1;}
.strip .l{color:var(--muted);font-size:.8rem;line-height:1.4;margin-top:.3rem;}
.ov-cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.9rem;margin-bottom:.9rem;}
.ov-cols>.card{margin-bottom:0;}
.card.tint-soft{background:var(--soft);} .card.tint-blue{background:var(--cobalt-soft);border-color:#cdd5ff;}
.numlist{list-style:none;padding:0!important;margin:.7rem 0 0 0!important;counter-reset:nl;}
.numlist li{counter-increment:nl;position:relative;padding-left:2.1rem;margin-bottom:.8rem!important;font-size:.93rem!important;line-height:1.55!important;}
.numlist li::before{content:counter(nl);position:absolute;left:0;top:.05rem;width:1.45rem;height:1.45rem;border-radius:50%;background:var(--midnight);color:#fff;font-size:.74rem;font-weight:800;display:flex;align-items:center;justify-content:center;}
.split{display:flex;height:16px;border-radius:5px;overflow:hidden;margin:.4rem 0 .35rem 0;}
.split .s-ok{background:var(--ok);} .split .s-bad{background:var(--red);}
.split-l{display:flex;justify-content:space-between;font-size:.78rem;color:var(--muted);margin-bottom:.9rem;font-weight:600;}
.card.wide .bar{grid-template-columns:260px 1fr 70px;margin:.55rem 0;font-size:.95rem;}
.card.wide .bar .track{height:16px;}
@media(max-width:980px){.ov-banner{grid-template-columns:1fr 1fr;}.ov-intro{grid-column:1/-1;}.ov-zone:nth-child(2){border-left:0;}
.strip{grid-template-columns:1fr 1fr;}.strip .cell:nth-child(3){border-left:0;}.ov-cols{grid-template-columns:1fr;}}
@media(max-width:640px){.ov-banner{grid-template-columns:1fr;}.ov-zone{border-left:0;border-top:1px solid rgba(255,255,255,.14);}.strip{grid-template-columns:1fr;}.strip .cell{border-left:0;border-top:1px solid var(--line);}.strip .cell:first-child{border-top:0;}.card.wide .bar{grid-template-columns:120px 1fr 52px;}}
/* ---------- URL scanner ---------- */
.urlbox{background:var(--midnight);border-radius:16px 16px 0 0;padding:1.2rem 1.5rem .9rem 1.5rem;margin-top:.2rem;}
.urlbox .ub-title{font-family:var(--display);color:#fff;font-size:1.3rem;font-weight:600;}
.urlbox .ub-sub{color:#b8c1e2;font-size:.88rem;margin-top:.2rem;}
.scanbar{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--cobalt);border-radius:12px;padding:.9rem 1.2rem;margin:.8rem 0 1rem 0;font-size:.9rem;color:var(--ink);word-break:break-all;}
.scanbar .chips{display:flex;flex-wrap:wrap;gap:.4rem;margin:.55rem 0;}
.chip{font-size:.76rem;font-weight:700;padding:.2rem .6rem;border-radius:6px;}
.chip.c-ok{background:var(--ok-soft);color:var(--ok);} .chip.c-est{background:var(--cobalt-soft);color:var(--cobalt);} .chip.c-def{background:#fff4dc;color:#8a5a00;}
.scanbar .sb-note{color:var(--muted);font-size:.84rem;line-height:1.5;word-break:normal;}
.scanbar div:not(.chips):not(.sb-note){color:#8a5a00;font-size:.84rem;margin-top:.3rem;word-break:normal;}
[data-baseweb="tab-highlight"]{background-color:var(--cobalt)!important;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p{color:var(--cobalt);}
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
    path = MODEL_PATH
    if not path.exists():
        # Tolerate renamed downloads such as "tuned_random_forest (1).skl"
        found = sorted(p for p in BASE.rglob("*") if p.is_file() and p.suffix in (".skl", ".joblib", ".pkl"))
        if not found:
            st.error(f"No model file (.skl) was found in the repo. Upload '{MODEL_PATH.name}' to the same folder as app.py.")
            st.stop()
        path = found[0]
    return joblib.load(path)


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset '{DATA_PATH.name}' was not found next to app.py. Upload it to the same folder as app.py.")
        st.stop()
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
    n_leg, n_phi = int((DF.Result == 1).sum()), int((DF.Result == -1).sum())
    pl = 100 * n_leg / (n_leg + n_phi)
    H(f"""
    <div class="ov-banner">
      <div class="ov-intro">
        <div class="ov-kicker">Random Forest · {MODEL.n_estimators} trees</div>
        <h2>Phishing Website Detection Using Random Forest</h2>
        <p>One website's URL and page traits in, one of two labels out. Trained and tested on dataset1.csv from the Phishing Website Detection Datasets collection.</p>
      </div>
      <div class="ov-zone"><div class="ov-num apri">{EV['acc']*100:.2f}%</div><div class="ov-cap">accuracy on {EV['n']:,} held-out records</div></div>
      <div class="ov-zone"><div class="ov-num">{(1-miss_rate)*100:.2f}%</div><div class="ov-cap">of phishing sites caught, with a <b>{false_alarm*100:.2f}%</b> false-alarm rate on legitimate sites</div></div>
    </div>
    <div class="strip">
      <div class="cell"><div class="n">{EV['f1']:.4f}</div><div class="l">Macro F1-score<br>both classes weighted equally</div></div>
      <div class="cell"><div class="n">{errors}</div><div class="l">Errors in {EV['n']:,} test records<br>({errors/EV['n']*100:.2f}%)</div></div>
      <div class="cell"><div class="n">{EV['auc']:.4f}</div><div class="l">ROC-AUC<br>the two classes separate very well</div></div>
      <div class="cell"><div class="n">{len(FEATURES)}</div><div class="l">Model inputs<br>read by {MODEL.n_estimators} decision trees</div></div>
    </div>
    <div class="ov-cols">
      <div class="card tint-soft">
        <h3>The problem</h3>
        <ul class="numlist">
          <li>Phishing sites imitate real ones to steal usernames, passwords and personal data, so spotting them by eye is hard.</li>
          <li>Blacklists and fixed security rules struggle with newly created or modified phishing sites.</li>
          <li>A classifier can learn patterns from sites already labelled legitimate or phishing, then apply them to new ones.</li>
        </ul>
      </div>
      <div class="card tint-blue">
        <h3>What we built</h3>
        <ul class="numlist">
          <li>A Random Forest of {MODEL.n_estimators} trees trained on {len(DF):,} labelled website records.</li>
          <li>An evaluation on a stratified 80/20 split: accuracy, precision, recall, F1, confusion matrix and ROC-AUC.</li>
          <li>This app, so you can scan a URL, test a website profile, replay a scan, or classify a whole CSV.</li>
        </ul>
      </div>
      <div class="card">
        <h3>The 2 website classes</h3>
        <p class="hint">The target column, Result, in dataset1.csv</p>
        <div class="split"><div class="s-ok" style="width:{pl:.1f}%"></div><div class="s-bad" style="width:{100-pl:.1f}%"></div></div>
        <div class="split-l"><span>{pl:.0f}% legitimate</span><span>{100-pl:.0f}% phishing</span></div>
        <div class="row">{badge(1)}<span class="nm">Legitimate</span><span class="ds">Coded 1 · {n_leg:,} records</span></div>
        <div class="row">{badge(-1)}<span class="nm">Phishing</span><span class="ds">Coded -1 · {n_phi:,} records</span></div>
      </div>
    </div>
    <div class="card wide">
      <h3>Strongest signals</h3>
      <p class="hint">Share of the model's decisions explained by each feature</p>
      {"".join(bar(META[f][0], v, IMP.iloc[0], f"{v*100:.1f}%", "navy" if i == 0 else "") for i, (f, v) in enumerate(IMP.head(5).items()))}
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
    <div class="card dark">
      <h3>General objective</h3>
      <p>To develop a Random Forest-based machine learning model for detecting phishing websites using website and URL-related characteristics.</p>
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
                f'<tr><td><b>{META[f][0]}</b><br><span style="color:#5d6580;font-size:.82rem">{f}</span></td>'
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
    grid = "".join(f'<line x1="{px(t)}" y1="{py(0)}" x2="{px(t)}" y2="{py(1)}" stroke="#e6e1d6"/><line x1="{px(0)}" y1="{py(t)}" x2="{px(1)}" y2="{py(t)}" stroke="#e6e1d6"/>'
                   f'<text x="{px(t)}" y="{py(0)+18}" font-size="11" text-anchor="middle" fill="#5d6580">{t:g}</text>'
                   f'<text x="{L-8}" y="{py(t)+4}" font-size="11" text-anchor="end" fill="#5d6580">{t:g}</text>' for t in (0, .25, .5, .75, 1))
    return (f'<svg viewBox="0 0 {W} {Hh}" width="100%" role="img" aria-label="ROC curve of the tuned model, AUC {EV["auc"]:.4f}">{grid}'
            f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" stroke="#b8b3a6" stroke-dasharray="5 5"/>'
            f'<polyline points="{pts}" fill="none" stroke="#2f4bff" stroke-width="3" stroke-linejoin="round"/>'
            f'<text x="{(L+W-R)/2}" y="{Hh-6}" font-size="12" text-anchor="middle" fill="#5d6580">False positive rate</text>'
            f'<text transform="translate(12 {(T+Hh-B)/2}) rotate(-90)" font-size="12" text-anchor="middle" fill="#5d6580">True positive rate</text>'
            f'<text x="{px(.97)}" y="{py(.12)}" font-size="14" font-weight="700" text-anchor="end" fill="#0f1a3a">AUC = {EV["auc"]:.4f}</text></svg>')


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
      <p class="hint" style="margin-top:.8rem">Dark bar: initial model. Blue bar: tuned model.</p>
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
    st.session_state["scan"] = None


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


# ---- URL feature extraction ---------------------------------------------------------------
import re
import socket
import ipaddress
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin

import requests

SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly", "bit.do", "cutt.ly",
              "rebrand.ly", "shorturl.at", "tiny.cc", "lnkd.in", "t.ly", "rb.gy", "v.gd", "shorte.st", "tr.im", "x.co"}
# These seven indicators need WHOIS, traffic-rank, search-index or blacklist data that the app cannot look up.
UNMEASURABLE = ["Domain_registeration_length", "age_of_domain", "web_traffic", "Page_Rank",
                "Google_Index", "Links_pointing_to_page", "Statistical_report"]
UA = {"User-Agent": "Mozilla/5.0 (compatible; PhishingDetectorResearch/1.0)"}


def base_domain(host: str) -> str:
    parts = host.lower().split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "gov", "edu", "ac") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def is_public_host(host: str) -> bool:
    """Block localhost / private networks so the app cannot be pointed at internal addresses."""
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False


def lexical_features(url: str) -> dict:
    u = urlparse(url)
    host = (u.hostname or "").lower()
    out = {}
    out["having_IPhaving_IP_Address"] = -1 if (re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host) or re.fullmatch(r"0x[0-9a-f]+", host) or ":" in host) else 1
    n = len(url)
    out["URLURL_Length"] = 1 if n < 54 else (0 if n <= 75 else -1)
    out["Shortining_Service"] = -1 if base_domain(host) in SHORTENERS or host in SHORTENERS else 1
    out["having_At_Symbol"] = -1 if "@" in url else 1
    out["double_slash_redirecting"] = -1 if url.rfind("//") > 7 else 1
    out["Prefix_Suffix"] = -1 if "-" in host else 1
    h = host[4:] if host.startswith("www.") else host
    dots = h.count(".")
    if re.search(r"\.(co|com|org|net|gov|edu|ac)\.[a-z]{2}$", h):
        dots -= 1
    out["having_Sub_Domain"] = 1 if dots <= 1 else (0 if dots == 2 else -1)
    out["HTTPS_token"] = -1 if "https" in host else 1
    out["port"] = 1 if u.port in (None, 80, 443) else -1
    out["Abnormal_URL"] = 1 if host and host in url.lower() else -1
    return out


class _Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.anchors, self.media, self.tagLinks, self.forms, self.iframes, self.icons = [], [], [], [], [], []

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "a" and "href" in a:
            self.anchors.append(a["href"].strip())
        elif tag in ("img", "audio", "video", "embed", "source") and a.get("src"):
            self.media.append(a["src"].strip())
        elif tag == "script" and a.get("src"):
            self.tagLinks.append(a["src"].strip())
        elif tag == "link" and a.get("href"):
            self.tagLinks.append(a["href"].strip())
            if "icon" in a.get("rel", "").lower():
                self.icons.append(a["href"].strip())
        elif tag == "meta" and a.get("content", "").lower().startswith(("http://", "https://")):
            self.tagLinks.append(a["content"].strip())
        elif tag == "form":
            self.forms.append(a.get("action", "").strip())
        elif tag == "iframe":
            style = a.get("style", "").replace(" ", "").lower()
            hidden = (a.get("frameborder") == "0" or a.get("width") in ("0", "1") or a.get("height") in ("0", "1")
                      or "display:none" in style or "visibility:hidden" in style)
            self.iframes.append(hidden)


def _safe_get(url: str, max_hops: int = 6):
    """Follow redirects by hand so every hop can be checked. Returns (response, redirects, cert_state)."""
    cert_state, hops, cur = 1, 0, url
    for _ in range(max_hops + 1):
        host = urlparse(cur).hostname or ""
        if not is_public_host(host):
            raise ValueError("blocked")
        try:
            r = requests.get(cur, headers=UA, timeout=8, allow_redirects=False, verify=True, stream=True)
        except requests.exceptions.SSLError:
            cert_state = 0
            r = requests.get(cur, headers=UA, timeout=8, allow_redirects=False, verify=False, stream=True)
        if r.is_redirect and r.headers.get("location"):
            cur = urljoin(cur, r.headers["location"])
            hops += 1
            r.close()
            continue
        return r, hops, cert_state, cur
    raise ValueError("too many redirects")


def _pct(items, bad):
    items = [i for i in items if i]
    if not items:
        return 0.0
    return 100 * sum(1 for i in items if bad(i)) / len(items)


def analyse_url(raw: str, defaults: dict):
    """Return (values, status, notes). status[f] is 'measured', 'estimated' or 'default'."""
    url = raw.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    notes = []
    vals = lexical_features(url)
    status = {f: "measured" for f in vals}
    host = (urlparse(url).hostname or "").lower()
    if not host or "." not in host and ":" not in host:
        raise ValueError("That does not look like a valid web address.")

    try:
        socket.getaddrinfo(host, None)
        vals["DNSRecord"] = 1
    except Exception:
        vals["DNSRecord"] = -1
    status["DNSRecord"] = "measured"

    vals["SSLfinal_State"] = -1 if url.startswith("http://") else 1
    status["SSLfinal_State"] = "measured"

    page_features = ["Favicon", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
                     "Iframe", "Redirect", "on_mouseover", "RightClick", "popUpWidnow"]
    try:
        r, hops, cert_state, final = _safe_get(url)
        if url.startswith("https://"):
            vals["SSLfinal_State"] = cert_state
        body = r.raw.read(1_500_000, decode_content=True).decode(r.encoding or "utf-8", errors="ignore")
        r.close()
        page_host = urlparse(final).hostname or host
        page_base = base_domain(page_host)
        parser = _Page()
        parser.feed(body)
        low = body.lower()

        def ext(link):
            if link.startswith(("mailto:", "tel:")):
                return False
            h = urlparse(urljoin(final, link)).hostname or ""
            return bool(h) and base_domain(h) != page_base

        vals["Favicon"] = -1 if parser.icons and ext(parser.icons[0]) else 1
        p = _pct(parser.media, ext)
        vals["Request_URL"] = 1 if p < 22 else (0 if p <= 61 else -1)
        p = _pct(parser.anchors, lambda a: a.startswith(("#", "javascript")) or ext(a))
        vals["URL_of_Anchor"] = 1 if p < 31 else (0 if p <= 67 else -1)
        p = _pct(parser.tagLinks, ext)
        vals["Links_in_tags"] = 1 if p < 17 else (0 if p <= 81 else -1)
        sfh = 1
        for act in parser.forms:
            if act in ("", "about:blank"):
                sfh = -1
                break
            if ext(act):
                sfh = 0
        vals["SFH"] = sfh
        vals["Submitting_to_email"] = -1 if any(a.lower().startswith("mailto:") for a in parser.forms) else 1
        vals["Iframe"] = -1 if any(parser.iframes) else 1
        vals["Redirect"] = 0 if hops <= 1 else 1
        vals["on_mouseover"] = -1 if re.search(r"onmouseover\s*=\s*[\"'][^\"']*window\.status", low) else 1
        vals["RightClick"] = -1 if re.search(r"event\.button\s*==\s*2|contextmenu[^;]{0,80}(preventdefault|return\s+false)", low) else 1
        vals["popUpWidnow"] = -1 if re.search(r"\bprompt\s*\(", low) else 1
        for f in page_features:
            status[f] = "estimated"
        status["Redirect"] = "measured"
        status["SSLfinal_State"] = "measured"
        if final != url:
            notes.append(f"The address redirected to {final[:80]}. Page indicators describe the final page.")
    except ValueError as e:
        notes.append("That address points to a private or internal network, so the page was not fetched." if str(e) == "blocked"
                     else "The page could not be followed (too many redirects).")
    except Exception:
        notes.append("The page could not be fetched (offline, blocked or timed out), so only address-based indicators were measured.")

    for f in FEATURES:
        if f not in vals or f in UNMEASURABLE:
            vals[f] = int(defaults[f])
            status[f] = "default"
        elif f not in status:
            status[f] = "default"
        if vals[f] not in META[f][2]:
            vals[f] = int(defaults[f])
            status[f] = "default"
    return {f: int(vals[f]) for f in FEATURES}, status, notes


def scan_url(raw: str):
    defaults = DF[FEATURES].mode().iloc[0].to_dict()
    vals, status, notes = analyse_url(raw, defaults)
    for f in FEATURES:
        st.session_state[ss_key(f)] = vals[f]
    st.session_state["actual"] = None
    st.session_state["scan"] = {"url": raw.strip(), "status": status, "notes": notes}


def page_check():
    head("Check a Website", "Paste a URL, or describe a site with the 30 indicators, and see what the model says")
    if "init" not in st.session_state:
        st.session_state["init"] = True
        pick(1)
    H('<div class="urlbox"><div class="ub-title">Scan a web address</div>'
      '<div class="ub-sub">The app reads what it can from the address and the live page, then asks the model for a verdict.</div></div>')
    uc1, uc2 = st.columns([5, 1.2], vertical_alignment="bottom")
    url_in = uc1.text_input("Website URL", placeholder="https://example.com/login", label_visibility="collapsed", key="url_in")
    go = uc2.button("Scan website", type="primary", width="stretch")
    if go:
        if not url_in.strip():
            st.warning("Type or paste a URL first.")
        else:
            try:
                with st.spinner("Reading the address and the page…"):
                    scan_url(url_in)
            except ValueError as e:
                st.error(str(e))
    scan = st.session_state.get("scan")
    if scan:
        stt = scan["status"]
        n_meas = sum(1 for v in stt.values() if v == "measured")
        n_est = sum(1 for v in stt.values() if v == "estimated")
        dflt = [META[f][0] for f, v in stt.items() if v == "default"]
        notes = "".join(f"<div>{n}</div>" for n in scan["notes"])
        H(f'<div class="scanbar"><b>{scan["url"][:90]}</b>'
          f'<div class="chips"><span class="chip c-ok">{n_meas} measured</span><span class="chip c-est">{n_est} estimated from page code</span>'
          f'<span class="chip c-def">{len(dflt)} typical values</span></div>'
          f'<div class="sb-note">Typical values were used for: {", ".join(dflt) if dflt else "nothing"}. '
          f'These need WHOIS, traffic-rank or blacklist data the app cannot look up, so set them yourself in the tabs below if you know them.</div>{notes}</div>')
    with st.expander("Or load an example record from the test set"):
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
              f'<b style="color:{"#138a5b" if ok else "#d1344f"}">{"The model agrees." if ok else "The model disagrees."}</b></div>')
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
            f'<div class="tile"><div class="n" style="color:#d1344f">{flagged}</div><div class="l">Flagged as phishing</div></div>'
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
      f'<div class="tile"><div class="n" style="color:#d1344f">{nph:,}</div><div class="l">Flagged as phishing</div></div>'
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
    H('<div class="brand"><div class="mark">🎣</div><div class="side-title">Phishing<br>Detector</div></div>'
      '<div class="side-sub">Website classification · Random Forest</div>')
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    H(f"""
    <div class="colo">
      <div class="co-kicker">Research project · S.Y. 2026–2027</div>
      <div class="co-title">Phishing Website Detection<br>Using Random Forest Classification</div>
      <div class="co-author">
        <div class="co-av">MG</div>
        <div class="co-who"><b>Mc Gabriel C. Limbojan</b><span>CST9L · University of Mindanao</span><span>College of Computing Education</span></div>
      </div>
      <div class="co-stats">
        <div><b>{MODEL.n_estimators}</b><span>trees</span></div>
        <div><b>{len(FEATURES)}</b><span>features</span></div>
        <div><b>{MODEL.random_state}</b><span>seed</span></div>
      </div>
    </div>""")

PAGES[choice]()
