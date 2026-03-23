"""
COT ML Rapor Üreticisi
══════════════════════════════════════════════════════════════
Çalıştırınca:
  1. CFTC API'den tüm emtiaların COT verisini çeker (26 hafta)
  2. Yahoo Finance'ten haftalık fiyat geçmişini çeker (6 ay)
  3. Her emtia için Logistic Regression ML modeli eğitir
  4. BIAS, yön olasılığı, feature importance hesaplar
  5. Tek sayfalık interaktif HTML raporu üretir → COT_Raporu.html

Kurulum:   pip install requests numpy scikit-learn
Çalıştır:  python generate_cot_report.py
Zamanlama: Her Cuma 19:00 (CFTC 15:30 ET'den sonra)
"""

import requests
import json
import numpy as np
import datetime
import sys
import os
import time
from pathlib import Path

# sklearn opsiyonel — yoksa kendi implementasyonu kullanılır
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN = True
except ImportError:
    SKLEARN = False
    print("scikit-learn bulunamadı, dahili ML kullanılacak")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
OUTPUT_FILE  = Path(__file__).parent / "COT_Raporu.html"
LOG_FILE     = Path(__file__).parent / "update_log.txt"
LEGACY_API   = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
TFF_API      = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
YF_BASE      = "https://query1.finance.yahoo.com/v8/finance/chart"

COMMODITIES = [
    dict(key="gold",   label="Altın",      emoji="🥇", api="legacy",
         mkt="GOLD - COMMODITY EXCHANGE INC.",
         yf="GC=F",   hist_range=(-20000, 280000), unit="100 troy oz · COMEX"),
    dict(key="silver", label="Gümüş",      emoji="🥈", api="legacy",
         mkt="SILVER - COMMODITY EXCHANGE INC.",
         yf="SI=F",   hist_range=(-15000, 80000),  unit="5000 troy oz · COMEX"),
    dict(key="crude",  label="Ham Petrol", emoji="🛢",  api="legacy",
         mkt="CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
         yf="CL=F",   hist_range=(-50000, 400000), unit="1000 varil · NYMEX"),
    dict(key="natgas", label="Doğal Gaz",  emoji="⛽", api="legacy",
         mkt="NATURAL GAS (HENRY HUB) - NEW YORK MERCANTILE EXCHANGE",
         yf="NG=F",   hist_range=(-150000, 80000), unit="10000 mmBtu · NYMEX"),
    dict(key="copper", label="Bakır",      emoji="🔶", api="legacy",
         mkt="COPPER- #1 - COMMODITY EXCHANGE INC.",
         yf="HG=F",   hist_range=(-30000, 100000), unit="25000 lbs · COMEX"),
    dict(key="wheat",  label="Buğday",     emoji="🌾", api="legacy",
         mkt="WHEAT - CHICAGO BOARD OF TRADE",
         yf="ZW=F",   hist_range=(-80000, 120000), unit="5000 bu · CBOT"),
    dict(key="corn",   label="Mısır",      emoji="🌽", api="legacy",
         mkt="CORN - CHICAGO BOARD OF TRADE",
         yf="ZC=F",   hist_range=(-200000, 400000),unit="5000 bu · CBOT"),
    dict(key="sp500",  label="S&P 500",    emoji="📈", api="tff",
         mkt="E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
         yf="^GSPC",  hist_range=(-200000, 600000),unit="E-mini · CME"),
    dict(key="nasdaq", label="Nasdaq 100", emoji="💻", api="tff",
         mkt="E-MINI NASDAQ-100 - CHICAGO MERCANTILE EXCHANGE",
         yf="^NDX",   hist_range=(-80000, 250000), unit="E-mini · CME"),
    dict(key="eurusd", label="EUR/USD",    emoji="💶", api="tff",
         mkt="EURO FX - CHICAGO MERCANTILE EXCHANGE",
         yf="EURUSD=X",hist_range=(-80000, 200000),unit="125000 EUR · CME"),
    dict(key="jpyusd", label="JPY/USD",    emoji="🇯🇵", api="tff",
         mkt="JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
         yf="JPY=X",  hist_range=(-80000, 60000),  unit="12.5M ¥ · CME"),
    dict(key="btc",    label="Bitcoin",    emoji="₿",  api="tff",
         mkt="BITCOIN - CHICAGO MERCANTILE EXCHANGE",
         yf="BTC-USD",hist_range=(-5000, 30000),   unit="5 BTC · CME"),
    dict(key="eth",    label="Ethereum",   emoji="Ξ",  api="tff",
         mkt="ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE",
         yf="ETH-USD",hist_range=(-3000, 18000),   unit="50 ETH · CME"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; COT-Reporter/1.0)"}

# ══════════════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════════════
def log(msg):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def fetch_cot(com, retries=3):
    api  = LEGACY_API if com["api"] == "legacy" else TFF_API
    name = com["mkt"].replace("'", "''")
    params = {
        "$where":  f"market_and_exchange_names='{name}'",
        "$order":  "report_date_as_yyyy_mm_dd DESC",
        "$limit":  "26",
    }
    for attempt in range(retries):
        try:
            r = requests.get(api, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            rows = r.json()
            if not rows:
                raise ValueError("Boş yanıt")
            rows.sort(key=lambda x: x.get("report_date_as_yyyy_mm_dd", ""), reverse=True)
            return rows
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise

def gf(d, *keys):
    for k in keys:
        v = d.get(k)
        if v not in (None, "", "0.0"):
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return 0.0

def parse_legacy(d):
    return dict(
        date      = d.get("report_date_as_yyyy_mm_dd", "?"),
        specLong  = int(gf(d, "noncomm_positions_long_all")),
        specShort = int(gf(d, "noncomm_positions_short_all")),
        specSpread= int(gf(d, "noncomm_postions_spread_all", "noncomm_positions_spread_all")),
        commLong  = int(gf(d, "comm_positions_long_all")),
        commShort = int(gf(d, "comm_positions_short_all")),
        totLong   = int(gf(d, "tot_rept_positions_long_all",  "tot_rpt_positions_long_all")),
        totShort  = int(gf(d, "tot_rept_positions_short_all", "tot_rpt_positions_short_all")),
        oi        = int(gf(d, "open_interest_all")),
        chgSL     = int(gf(d, "change_in_noncomm_long_all")),
        chgSS     = int(gf(d, "change_in_noncomm_short_all")),
        chgCL     = int(gf(d, "change_in_comm_long_all")),
        chgCS     = int(gf(d, "change_in_comm_short_all")),
    )

def parse_tff(d):
    sl = gf(d, "lev_money_positions_long_all")
    ss = gf(d, "lev_money_positions_short_all")
    cl = gf(d, "asset_mgr_positions_long_all")
    cs = gf(d, "asset_mgr_positions_short_all")
    return dict(
        date      = d.get("report_date_as_yyyy_mm_dd", "?"),
        specLong  = int(sl),
        specShort = int(ss),
        specSpread= int(gf(d, "lev_money_positions_spread_all")),
        commLong  = int(cl),
        commShort = int(cs),
        totLong   = int(sl + cl + gf(d,"dealer_positions_long_all") + gf(d,"other_rept_positions_long_all")),
        totShort  = int(ss + cs + gf(d,"dealer_positions_short_all") + gf(d,"other_rept_positions_short_all")),
        oi        = int(gf(d, "open_interest_all")),
        chgSL     = int(gf(d, "change_in_lev_money_long_all")),
        chgSS     = int(gf(d, "change_in_lev_money_short_all")),
        chgCL     = int(gf(d, "change_in_asset_mgr_long_all")),
        chgCS     = int(gf(d, "change_in_asset_mgr_short_all")),
    )

def fetch_price(yf_symbol, retries=3):
    url = f"{YF_BASE}/{requests.utils.quote(yf_symbol)}"
    params = {"interval": "1wk", "range": "1y"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            data   = r.json()
            result = data["chart"]["result"][0]
            ts     = result["timestamp"]
            q      = result["indicators"]["quote"][0]
            closes = q.get("close", [])
            highs  = q.get("high",  [])
            lows   = q.get("low",   [])
            vols   = q.get("volume",[])
            rows = []
            for i, t in enumerate(ts):
                c = closes[i] if i < len(closes) else None
                if c is None:
                    continue
                rows.append(dict(
                    date   = datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"),
                    close  = round(float(c), 4),
                    high   = round(float(highs[i]),4) if i<len(highs) and highs[i] else None,
                    low    = round(float(lows[i]),4)  if i<len(lows)  and lows[i]  else None,
                    volume = int(vols[i])              if i<len(vols)  and vols[i]  else None,
                ))
            rows.sort(key=lambda x: x["date"], reverse=True)
            return rows
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise

# ══════════════════════════════════════════════════════════════════
# ML ENGINE
# ══════════════════════════════════════════════════════════════════
FEAT_NAMES = [
    "Spec Net",
    "Net Delta (1H)",
    "Net Delta (2H)",
    "L/S Oran",
    "Ticari Net",
    "Net / OI",
    "Chg Net",
    "Fiyat Getiri",
    "Stochastic %K",
    "Vol Değişimi",
]

def build_features(cot, price):
    """Her hafta için özellik vektörü üret."""
    rows = []
    n = len(cot) - 2
    price_map = {p["date"]: p for p in price} if price else {}

    for i in range(n):
        w   = cot[i]
        w1  = cot[i + 1]
        w2  = cot[i + 2] if i + 2 < len(cot) else w1

        net   = w["specLong"]  - w["specShort"]
        net1  = w1["specLong"] - w1["specShort"]
        net2  = w2["specLong"] - w2["specShort"]
        comm  = w["commLong"]  - w["commShort"]

        f0 = float(net)
        f1 = float(net - net1)
        f2 = float(net1 - net2)
        f3 = float(w["specLong"] / w["specShort"]) if w["specShort"] > 0 else 1.0
        f4 = float(comm)
        f5 = float(net / w["oi"]) if w["oi"] > 0 else 0.0
        f6 = float(w["chgSL"] - w["chgSS"])

        # Fiyat özellikleri
        f7 = f8 = f9 = 0.0
        p_curr = price_map.get(w["date"])
        p_prev = price_map.get(w1["date"])
        if p_curr and p_prev and p_prev["close"]:
            f7 = (p_curr["close"] - p_prev["close"]) / p_prev["close"]
        if p_curr and p_curr.get("high") and p_curr.get("low"):
            rng = p_curr["high"] - p_curr["low"]
            f8  = (p_curr["close"] - p_curr["low"]) / rng if rng > 0 else 0.5
        if p_curr and p_prev:
            v0, v1 = p_curr.get("volume"), p_prev.get("volume")
            if v0 and v1 and v1 > 0:
                f9 = (v0 - v1) / v1

        # Hedef: bir sonraki hafta fiyat yön (1=yukari, 0=asagi)
        target = None
        if i > 0:
            p_next = price_map.get(cot[i - 1]["date"])
            if p_curr and p_next:
                target = 1 if p_next["close"] > p_curr["close"] else 0

        rows.append(dict(
            features = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9],
            target   = target,
        ))

    return rows

def run_ml(cot, price):
    rows       = build_features(cot, price)
    train_rows = [r for r in rows if r["target"] is not None]
    has_price  = price and len(price) >= 4

    if len(train_rows) < 5:
        return rule_based(cot)

    X = np.array([r["features"] for r in train_rows])
    y = np.array([r["target"]   for r in train_rows])

    # Predict için güncel hafta (index 0)
    X_pred = np.array([rows[0]["features"]])

    if SKLEARN:
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        X_pred_sc = scaler.transform(X_pred)

        model = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        model.fit(X_sc, y)
        prob_up = float(model.predict_proba(X_pred_sc)[0][1])

        # Feature importance (|coef| normalize)
        coefs = np.abs(model.coef_[0])
        importance = (coefs / (coefs.max() + 1e-9)).tolist()

        # Geriye dönük doğruluk (son 6 hafta hariç, kalanla eğit)
        accuracy = None
        if len(train_rows) >= 10:
            split   = max(4, len(train_rows) - 8)
            X_tr_sc = scaler.transform(X[:split])
            X_val_sc= scaler.transform(X[split:])
            m2 = LogisticRegression(C=0.5, max_iter=500, random_state=42)
            m2.fit(X_tr_sc, y[:split])
            preds    = m2.predict(X_val_sc)
            accuracy = int(np.mean(preds == y[split:]) * 100)

    else:
        # Dahili normalleştirme + gradyan iniş
        means = X.mean(axis=0)
        stds  = X.std(axis=0)
        stds[stds == 0] = 1
        X_sc      = (X      - means) / stds
        X_pred_sc = (X_pred - means) / stds

        w_vec = np.zeros(X.shape[1])
        bias  = 0.0
        lr    = 0.05
        for _ in range(500):
            z    = X_sc @ w_vec + bias
            pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            err  = pred - y
            w_vec -= lr * (X_sc.T @ err / len(y) + 0.01 * w_vec / len(y))
            bias  -= lr * err.mean()

        z_pred  = X_pred_sc @ w_vec + bias
        prob_up = float(1 / (1 + np.exp(-float(z_pred[0]))))
        coefs   = np.abs(w_vec)
        importance = (coefs / (coefs.max() + 1e-9)).tolist()
        accuracy = None

    prob_down = 1 - prob_up
    bias_label, bias_arrow, bias_bg, bias_color = _bias_label(prob_up)

    return dict(
        prob_up   = prob_up,
        prob_down = prob_down,
        bias      = bias_label,
        arrow     = bias_arrow,
        bg        = bias_bg,
        color     = bias_color,
        importance= importance,
        accuracy  = accuracy,
        train_size= len(train_rows),
        has_price = has_price,
        method    = "sklearn" if SKLEARN else "builtin",
    )

def rule_based(cot):
    d, d1 = cot[0], cot[1] if len(cot) > 1 else cot[0]
    net  = d["specLong"] - d["specShort"]
    net1 = d1["specLong"] - d1["specShort"]
    comm = d["commLong"] - d["commShort"]
    score = 0.5
    score += 0.08  if net  > net1 else -0.08
    score += 0.10  if comm < 0 and net > 0 else (-0.10 if comm > 0 and net < 0 else 0)
    score += 0.07  if d["chgSL"] > d["chgSS"] else -0.07
    score += 0.05  if d["oi"] > d1["oi"] and net > net1 else 0
    score = max(0.1, min(0.9, score))
    bl, ba, bb, bc = _bias_label(score)
    return dict(prob_up=score, prob_down=1-score, bias=bl, arrow=ba, bg=bb, color=bc,
                importance=[0.9,0.8,0.6,0.7,0.4,0.5,0.3,0,0,0],
                accuracy=None, train_size=0, has_price=False, method="rule")

def _bias_label(p):
    if   p >= 0.68: return "LONG",  "▲", "#1a7a4a", "#fff"
    elif p >= 0.55: return "LONG",  "△", "#2e7d32", "#fff"
    elif p >= 0.45: return "NÖTR",  "◆", "#92400e", "#fff"
    elif p >= 0.32: return "SHORT", "▽", "#b45309", "#fff"
    else:           return "SHORT", "▼", "#b91c1c", "#fff"

# ══════════════════════════════════════════════════════════════════
# YORUM MOTORU
# ══════════════════════════════════════════════════════════════════
def build_reasons(cot, price, ml_res, com):
    d, d1 = cot[0], cot[1] if len(cot) > 1 else cot[0]
    d2 = cot[2] if len(cot) > 2 else d1
    net  = d["specLong"]  - d["specShort"]
    net1 = d1["specLong"] - d1["specShort"]
    comm = d["commLong"]  - d["commShort"]
    h_min, h_max = com["hist_range"]
    h_span = h_max - h_min
    net_pct = int(max(0, min(100, (net - h_min) / h_span * 100))) if h_span else 50

    reasons = []

    # 1. Net pozisyon
    dir_str = "net long" if net > 0 else "net short"
    reasons.append(("📊", f"Spekülatörler <strong>{fmt(abs(net))}</strong> kontrat <strong>{dir_str}</strong> pozisyonda. "
                          f"Haftalık değişim: <strong>{'+' if net>=net1 else ''}{fmt(net-net1)}</strong>."))

    # 2. Momentum streak
    diffs = [net - net1, net1 - (d2["specLong"] - d2["specShort"])]
    streak = sum(1 for x in diffs if (x > 0) == (diffs[0] > 0))
    if streak >= 2:
        mom_dir = "birikim artıyor" if diffs[0] > 0 else "birikim azalıyor"
        reasons.append(("📅", f"Son <strong>{streak} hafta</strong> üst üste {mom_dir}."))

    # 3. Ticari kutuplaşma
    if comm < 0 and net > 0:
        reasons.append(("🏦", f"Ticari oyuncular <strong>{fmt(abs(comm))} net short</strong> — klasik üretici hedge. "
                              "Spekülatörlerle karşı pozisyon = güçlü kutuplaşma."))
    elif comm > 0 and net < 0:
        reasons.append(("🏦", f"Ticari oyuncular <strong>{fmt(abs(comm))} net long</strong> — spekülatörlerle zıt yönde."))
    else:
        reasons.append(("🏦", f"Ticari oyuncular <strong>{fmt(abs(comm))} net {'long' if comm >= 0 else 'short'}</strong> "
                              "— her iki taraf aynı yönde dikkatli olun."))

    # 4. Extreme seviye
    if net_pct > 85:
        tag = f'<span class="tag tag-short">AŞIRI LONG %{net_pct}</span>'
    elif net_pct < 15:
        tag = f'<span class="tag tag-long">AŞIRI SHORT %{net_pct}</span>'
    else:
        tag = f'<span class="tag tag-neutral">NORMAL %{net_pct}</span>'
    reasons.append(("📏", f"Tarihsel konum: {tag} — spec net pozisyon tarihsel aralığın %{net_pct}'inde."))

    # 5. Fiyat örtüşme
    if price and len(price) >= 2:
        p0, p1 = price[0], price[1]
        ret    = (p0["close"] - p1["close"]) / p1["close"] * 100
        p_dir  = p0["close"] > p1["close"]
        cot_up = net > net1
        aligned = (p_dir == cot_up)
        color  = "#1a7a4a" if p_dir else "#c0392b"
        reasons.append(("✅" if aligned else "⚠",
            f"Fiyat haftalık <strong style='color:{color}'>{'yükseldi' if p_dir else 'düştü'} "
            f"%{abs(ret):.2f}</strong>, COT birikim {'artıyor' if cot_up else 'azalıyor'}. "
            f"<strong>{'Örtüşme onaylandı — trend güçlü.' if aligned else 'Divergence! — zıt yön, kırılım riski.'}</strong>"))

        # Destek/direnç
        last5  = price[:5]
        sup    = min(p["low"] or p["close"] for p in last5)
        res    = max(p["high"] or p["close"] for p in last5)
        rng    = res - sup
        pos_pct = int((p0["close"] - sup) / rng * 100) if rng > 0 else 50
        reasons.append(("🎯",
            f"Güncel fiyat <strong>{p0['close']:.2f}</strong> — 5 haftalık destek/direnç aralığının "
            f"<strong>%{pos_pct}'inde</strong> (Destek: {sup:.2f} · Direnç: {res:.2f})."))

    # BIAS bozucu koşullar
    breakers = []
    if ml_res["bias"] == "LONG":
        breakers.append("Spec short bu haftadan itibaren hızla artarsa (chgSS > 5000) → bias tersine döner")
        if net_pct > 80:
            breakers.append("Aşırı long bölgede — beklenmedik negatif haber büyük satış tetikleyebilir")
        breakers.append("OI düşerken fiyat yükselirse sahte kırılım, pozisyon boyutunu küçük tut")
        if comm > 0:
            breakers.append("Ticari oyuncular short pozisyona geçerse kutuplaşma bozulur")
    elif ml_res["bias"] == "SHORT":
        breakers.append("Spec long ani artışı (chgSL > 5000) görülürse → bias tersine döner")
        if net_pct < 20:
            breakers.append("Aşırı short bölgede — short squeeze riski yüksek, stop sıkı tut")
        breakers.append("Fiyat güçlü destek üzerinde kapanırsa COT dönüşü başlayabilir")
    else:
        breakers.append("Net pozisyon bu hafta > +3000 olursa LONG bias devreye girer")
        breakers.append("Net pozisyon bu hafta < -3000 olursa SHORT bias devreye girer")
        breakers.append("Belirsizlik döneminde pozisyon boyutunu küçük tut")

    return reasons, breakers, net_pct

def fmt(n):
    return f"{abs(int(round(n))):,}".replace(",", ".")

def fmts(n):
    return ("+" if n >= 0 else "-") + fmt(n)

# ══════════════════════════════════════════════════════════════════
# HTML ÜRETICI
# ══════════════════════════════════════════════════════════════════
CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#f4f5f7;color:#1a2332;font-size:14px}
.header{background:#1a2332;color:#fff;padding:16px 28px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.header h1{font-size:16px;font-weight:600}.header-sub{font-size:11px;color:#7a9cbf;margin-top:2px}
.gen-date{font-size:11px;color:#c9a84c;font-weight:500}
.tabs{background:#fff;border-bottom:1px solid #e2e6ec;padding:0 20px;display:flex;overflow-x:auto;scrollbar-width:none}
.tab{padding:12px 15px;font-size:13px;cursor:pointer;color:#6b7c93;border-bottom:2px solid transparent;white-space:nowrap;background:none;border-top:none;border-left:none;border-right:none;font-family:'Inter',sans-serif}
.tab:hover{color:#1a2332}.tab.active{color:#1a2332;border-bottom-color:#c9a84c;font-weight:600}
.content{max-width:1200px;margin:0 auto;padding:20px}
.section{display:none}.section.active{display:block}
.bias-hero{background:#fff;border:1px solid #e2e6ec;border-radius:10px;overflow:hidden;margin-bottom:14px}
.bias-top{padding:13px 20px;border-bottom:1px solid #e2e6ec;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
.bias-top-title{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;color:#8a9bb0}
.bias-date{font-size:11px;color:#8a9bb0}
.bias-body{display:grid;grid-template-columns:160px 1fr}
@media(max-width:600px){.bias-body{grid-template-columns:1fr}}
.verdict{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:22px 14px;gap:6px;border-right:1px solid #e2e6ec;text-align:center}
.v-arrow{font-size:40px;line-height:1}.v-label{font-size:16px;font-weight:700;letter-spacing:.5px}
.v-pct{font-size:30px;font-weight:700;line-height:1}.v-pct-lbl{font-size:10px;color:#8a9bb0}
.v-bar{width:80px;height:5px;background:#edf0f5;border-radius:3px;margin-top:5px;overflow:hidden}
.v-bar-fill{height:100%;border-radius:3px}
.v-meta{font-size:10px;color:#b0bec5;margin-top:5px}
.analysis{padding:18px 20px;display:flex;flex-direction:column;gap:9px}
.reason-row{display:flex;gap:9px;font-size:12px;line-height:1.6}
.r-icon{flex-shrink:0;font-size:13px;margin-top:1px}.r-text{color:#4a5568}
.r-text strong{color:#1a2332;font-weight:600}
.tag{display:inline-block;font-size:10px;font-weight:600;padding:1px 7px;border-radius:3px;margin-left:3px;vertical-align:middle}
.tag-long{background:#e6f4ea;color:#1a7a4a}.tag-short{background:#fce8e8;color:#b91c1c}.tag-neutral{background:#fef9e7;color:#92400e}
.breaker-box{background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:10px 14px;margin-top:2px}
.breaker-title{font-size:10px;font-weight:600;color:#92400e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}
.breaker-item{font-size:11px;color:#78350f;line-height:1.7;display:flex;gap:5px}
.charts-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}
@media(max-width:680px){.charts-row{grid-template-columns:1fr}}
.chart-box{background:#fff;border:1px solid #e2e6ec;border-radius:8px;padding:14px 16px}
.chart-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#8a9bb0;margin-bottom:10px}
.bar-chart{display:flex;align-items:flex-end;gap:4px;height:90px}
.bar-col{flex:1;display:flex;flex-direction:column;align-items:center;gap:2px}
.bar-body{width:100%;border-radius:2px 2px 0 0;min-height:2px}
.bar-lbl{font-size:9px;color:#8a9bb0;text-align:center}.bar-val{font-size:9px;font-weight:500;text-align:center}
.prob-section{background:#fff;border:1px solid #e2e6ec;border-radius:8px;padding:14px 18px;margin-bottom:14px}
.prob-row{display:flex;align-items:center;gap:10px;font-size:12px;margin-bottom:7px}
.prob-lbl{width:65px;flex-shrink:0;font-weight:500}
.prob-track{flex:1;height:20px;background:#f0f2f5;border-radius:4px;overflow:hidden;position:relative}
.prob-fill{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:8px;font-size:11px;font-weight:600;color:#fff}
.factor-row{display:flex;align-items:center;gap:8px;font-size:11px;margin-bottom:5px}
.factor-name{width:145px;flex-shrink:0;color:#6b7c93}
.factor-track{flex:1;height:6px;background:#edf0f5;border-radius:3px;overflow:hidden}
.factor-fill{height:100%;border-radius:3px}
.factor-val{width:28px;text-align:right;font-weight:600}
.overlap-box{background:#fff;border:1px solid #e2e6ec;border-radius:8px;padding:14px 18px;margin-bottom:14px}
.overlap-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin:10px 0}
.ostat{background:#f8f9fc;border:1px solid #e2e6ec;border-radius:6px;padding:10px 12px}
.ostat-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:#8a9bb0;margin-bottom:3px}
.ostat-val{font-size:17px;font-weight:600}.ostat-sub{font-size:10px;color:#8a9bb0;margin-top:2px}
.overlap-result{padding:10px 14px;border-radius:6px;font-size:12px;line-height:1.6}
.extreme-box{background:#fff;border:1px solid #e2e6ec;border-radius:8px;padding:14px 16px;margin-bottom:14px}
.ext-track{height:12px;background:#edf0f5;border-radius:6px;overflow:hidden;margin:6px 0}
.ext-fill{height:100%;border-radius:6px}
.ext-zones{display:flex;justify-content:space-between;font-size:9px;color:#b0bec5}
.table-box{background:#fff;border:1px solid #e2e6ec;border-radius:8px;overflow:hidden;margin-bottom:14px}
.table-hdr{padding:10px 16px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e2e6ec}
.table-ttl{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#8a9bb0}
.table-badge{font-size:11px;color:#8a9bb0}
table{width:100%;border-collapse:collapse}
th{padding:8px 13px;text-align:right;font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:#8a9bb0;font-weight:500;border-bottom:1px solid #e2e6ec}
th:first-child{text-align:left}
td{padding:9px 13px;text-align:right;border-bottom:1px solid #f0f2f5;font-size:12px}
td:first-child{text-align:left;font-weight:500;color:#4a5568}
tr:last-child td{border-bottom:none}
tr.sub td{background:#f8f9fc;font-size:11px;color:#8a9bb0}
tr.sub td:first-child{padding-left:26px}
.cp{color:#1a7a4a;font-weight:500}.cn{color:#c0392b;font-weight:500}
.pos{color:#1a7a4a}.neg{color:#c0392b}.gold{color:#c9a84c}
.note{font-size:11px;color:#8a9bb0;padding-top:14px;line-height:1.7;border-top:1px solid #e2e6ec;margin-top:4px}
.note a{color:#c9a84c;text-decoration:none}
"""

def bar_chart_html(bars_data, height=90, value_fmt=None):
    """bars_data: list of (value, color, label)"""
    max_abs = max(abs(v) for v, _, _ in bars_data) or 1
    cols = []
    for val, color, label in bars_data:
        h = max(int(abs(val) / max_abs * (height - 12)), 3)
        vstr = value_fmt(val) if value_fmt else str(int(val))
        cols.append(
            f'<div class="bar-col">'
            f'<div class="bar-val" style="color:{color}">{vstr}</div>'
            f'<div class="bar-body" style="height:{h}px;background:{color};opacity:.75"></div>'
            f'<div class="bar-lbl">{label}</div>'
            f'</div>'
        )
    return f'<div class="bar-chart" style="height:{height}px">{"".join(cols)}</div>'

def render_commodity_html(com, cot, price, ml_res, reasons, breakers, net_pct):
    d  = cot[0]
    d1 = cot[1] if len(cot) > 1 else cot[0]
    spec_net = d["specLong"] - d["specShort"]
    comm_net = d["commLong"] - d["commShort"]
    tot_net  = d["totLong"]  - d["totShort"]
    h_min, h_max = com["hist_range"]
    extreme_color = "#c0392b" if net_pct > 85 else "#1a7a4a" if net_pct < 15 else "#c9a84c"
    pct_up   = int(ml_res["prob_up"] * 100)
    pct_down = int(ml_res["prob_down"] * 100)

    # ── Net trend bars ──
    bar_data = [(w["specLong"] - w["specShort"],
                 "#1a7a4a" if w["specLong"] - w["specShort"] >= 0 else "#c0392b",
                 (w["date"] or "")[-5:].replace("-", "."))
                for w in cot[:8][::-1]]
    net_bars = bar_chart_html(bar_data, value_fmt=lambda v: f"{'+'if v>=0 else ''}{int(v/1000)}K")

    # ── Momentum bars ──
    nets  = [w["specLong"] - w["specShort"] for w in cot]
    diffs = [nets[i] - nets[i+1] for i in range(min(7, len(nets)-1))]
    diffs_rev = diffs[::-1]
    mom_max = max(abs(x) for x in diffs_rev) or 1
    mom_cols = []
    for i, val in enumerate(diffs_rev):
        h   = max(int(abs(val) / mom_max * 34), 2)
        col = "#1a7a4a" if val >= 0 else "#c0392b"
        lbl = (cot[len(diffs)-1-i]["date"] or "")[-5:].replace("-",".")
        if val >= 0:
            inner = (f'<div style="height:{h}px;width:100%;background:{col};opacity:.75;border-radius:2px 2px 0 0"></div>'
                     f'<div style="height:34px"></div>')
        else:
            inner = (f'<div style="height:34px"></div>'
                     f'<div style="height:{h}px;width:100%;background:{col};opacity:.75;border-radius:0 0 2px 2px"></div>')
        mom_cols.append(f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;height:70px;justify-content:center">'
                        f'{inner}<div class="bar-lbl">{lbl}</div></div>')
    mom_chart = f'<div style="display:flex;align-items:center;gap:3px;height:70px">{"".join(mom_cols)}</div>'

    # ── Price bars ──
    price_section = ""
    overlap_section = ""
    if price and len(price) >= 2:
        p0, p1 = price[0], price[1]
        p_bars = [(p["close"], "#60a5fa", (p["date"] or "")[-5:].replace("-","."))
                  for p in price[:8][::-1]]
        price_section = f'''
        <div class="charts-row">
          <div class="chart-box" style="grid-column:1/-1">
            <div class="chart-title">Fiyat Seyri (Haftalık Kapanış)</div>
            {bar_chart_html(p_bars, value_fmt=lambda v: f"{v:.0f}")}
          </div>
        </div>'''

        ret     = (p0["close"] - p1["close"]) / p1["close"] * 100
        p_dir   = p0["close"] > p1["close"]
        cot_up  = nets[0] > nets[1]
        aligned = (p_dir == cot_up)
        p_color = "#1a7a4a" if p_dir else "#c0392b"

        last5 = price[:5]
        sup   = min(p.get("low") or p["close"] for p in last5)
        res   = max(p.get("high") or p["close"] for p in last5)
        rng   = res - sup

        overlap_section = f'''
        <div class="overlap-box">
          <div class="chart-title" style="margin-bottom:8px">📈 Fiyat + COT Örtüşme Analizi</div>
          <div class="overlap-grid">
            <div class="ostat"><div class="ostat-lbl">Güncel Fiyat</div>
              <div class="ostat-val gold">{p0["close"]:.2f}</div>
              <div class="ostat-sub">{p0["date"]}</div></div>
            <div class="ostat"><div class="ostat-lbl">Haftalık Getiri</div>
              <div class="ostat-val {'pos' if p_dir else 'neg'}">{'+' if p_dir else ''}{ret:.2f}%</div>
              <div class="ostat-sub">Kapanış bazlı</div></div>
            <div class="ostat"><div class="ostat-lbl">COT Momentum</div>
              <div class="ostat-val {'pos' if cot_up else 'neg'}">{'Birikim ↑' if cot_up else 'Azalış ↓'}</div>
              <div class="ostat-sub">Δ {fmts(nets[0]-nets[1])}</div></div>
            <div class="ostat"><div class="ostat-lbl">Örtüşme</div>
              <div class="ostat-val" style="color:{'#1a7a4a' if aligned else '#c0392b'}">{'✅ Onay' if aligned else '⚠ Diverg.'}</div>
              <div class="ostat-sub">{'Aynı yön' if aligned else 'Zıt yön'}</div></div>
          </div>
          <div class="overlap-result" style="background:{'#f0faf4' if aligned else '#fffbeb'};border:1px solid {'#c6e6d1' if aligned else '#fde68a'}">
            <strong style="color:{'#1a7a4a' if aligned else '#92400e'}">{'Örtüşme Onaylandı — Güçlü Sinyal' if aligned else 'Divergence Tespit Edildi — Dikkatli Ol'}</strong><br>
            <span style="font-size:12px;color:#4a5568">Fiyat haftalık <strong style="color:{p_color}">{'yükseldi' if p_dir else 'düştü'} %{abs(ret):.2f}</strong>,
            spekülatör birikim {'arttı' if cot_up else 'azaldı'} ({fmts(nets[0]-nets[1])} kontrat).
            {'Her iki sinyal aynı yönde — bias güvenilirliği yüksek.' if aligned else 'Fiyat ve COT zıt yönde — trend kırılımı olabilir, pozisyon boyutunu küçük tut.'}</span>
          </div>
        </div>'''

    # ── Reasons / Breakers ──
    reason_html = "".join(
        f'<div class="reason-row"><span class="r-icon">{icon}</span><span class="r-text">{text}</span></div>'
        for icon, text in reasons
    )
    breaker_html = "".join(
        f'<div class="breaker-item"><span>•</span><span>{b}</span></div>'
        for b in breakers
    )

    # ── Feature importance ──
    feat_rows = "".join(
        f'<div class="factor-row">'
        f'<span class="factor-name">{FEAT_NAMES[i]}</span>'
        f'<div class="factor-track"><div class="factor-fill" style="width:{int(imp*100)}%;background:#60a5fa"></div></div>'
        f'<span class="factor-val" style="color:#60a5fa">{int(imp*100)}</span>'
        f'</div>'
        for i, imp in enumerate(ml_res["importance"])
    )

    # ── COT table ──
    def chg_td(n):
        cls = "cp" if n >= 0 else "cn"
        return f'<td class="{cls}">{fmts(n)}</td>'

    hist_rows = ""
    for i, w in enumerate(cot[:12]):
        net_w = w["specLong"] - w["specShort"]
        bg    = "" if i % 2 == 0 else "background:#fafbfc"
        hist_rows += (
            f'<tr style="{bg}">'
            f'<td style="color:#8a9bb0;font-weight:400">{w["date"]}</td>'
            f'<td>{fmt(w["specLong"])}</td><td>{fmt(w["specShort"])}</td>'
            f'<td class="{"pos" if net_w>=0 else "neg"}">{fmts(net_w)}</td>'
            f'{chg_td(w["chgSL"])}{chg_td(w["chgSS"])}'
            f'<td style="color:#8a9bb0">{fmt(w["oi"])}</td></tr>'
        )

    acc_str = f"Geriye dönük doğruluk: %{ml_res['accuracy']}" if ml_res["accuracy"] else "COT-only mod"
    method_str = {"sklearn": "scikit-learn LR", "builtin": "Dahili LR", "rule": "Kural bazlı"}.get(ml_res["method"], "")

    return f"""
    <!-- BIAS HERO -->
    <div class="bias-hero">
      <div class="bias-top">
        <span class="bias-top-title">Haftalık BIAS — {com['emoji']} {com['label']}</span>
        <span class="bias-date">COT: {d['date']}{f" · Fiyat: {price[0]['date']}" if price else ""}</span>
      </div>
      <div class="bias-body">
        <div class="verdict" style="background:{'#f8fff9' if ml_res['bias']=='LONG' else '#fff8f8' if ml_res['bias']=='SHORT' else '#fffdf0'}">
          <div class="v-arrow" style="color:{ml_res['bg']}">{ml_res['arrow']}</div>
          <div class="v-label" style="color:{ml_res['bg']}">{ml_res['bias']}</div>
          <div style="height:8px"></div>
          <div class="v-pct" style="color:{ml_res['bg']}">{pct_up}%</div>
          <div class="v-pct-lbl">YUKARI OLASILIK</div>
          <div class="v-bar"><div class="v-bar-fill" style="width:{pct_up}%;background:{ml_res['bg']}"></div></div>
          {f'<div class="v-meta">Doğruluk: %{ml_res["accuracy"]}</div>' if ml_res["accuracy"] else ''}
          <div class="v-meta">{method_str}{f" · {ml_res['train_size']}h" if ml_res["train_size"] else ""}</div>
        </div>
        <div class="analysis">
          {reason_html}
          <div class="breaker-box">
            <div class="breaker-title">⛔ Bu koşullar BIAS değiştirir</div>
            {breaker_html}
          </div>
        </div>
      </div>
    </div>

    <!-- OLASILIK -->
    <div class="prob-section">
      <div class="chart-title" style="margin-bottom:10px">ML Yön Olasılıkları</div>
      <div class="prob-row"><span class="prob-lbl pos">Yukarı</span>
        <div class="prob-track"><div class="prob-fill" style="width:{pct_up}%;background:#1a7a4a">{pct_up}%</div></div></div>
      <div class="prob-row"><span class="prob-lbl neg">Aşağı</span>
        <div class="prob-track"><div class="prob-fill" style="width:{pct_down}%;background:#c0392b">{pct_down}%</div></div></div>
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid #e2e6ec">
        <div class="chart-title" style="margin-bottom:8px">Feature Importance</div>
        {feat_rows}
        <div style="font-size:10px;color:#8a9bb0;margin-top:6px">{method_str} · Eğitim: {ml_res["train_size"]} hafta · {acc_str}</div>
      </div>
    </div>

    {overlap_section}

    <!-- GRAFİKLER -->
    <div class="charts-row">
      <div class="chart-box">
        <div class="chart-title">Spekülatör Net Trend</div>
        {net_bars}
      </div>
      <div class="chart-box">
        <div class="chart-title">Momentum (Haftalık Δ)</div>
        {mom_chart}
      </div>
    </div>

    {price_section}

    <!-- EXTREME GAUGE -->
    <div class="extreme-box">
      <div class="chart-title" style="margin-bottom:8px">Tarihsel Extreme Konum</div>
      <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px">
        <span style="color:#6b7c93">Spec Net · {fmt(h_min)} → {fmt(h_max)}</span>
        <span style="font-weight:600;color:{extreme_color}">%{net_pct} · {'+' if spec_net>=0 else ''}{fmt(spec_net)}</span>
      </div>
      <div class="ext-track"><div class="ext-fill" style="width:{net_pct}%;background:{extreme_color}"></div></div>
      <div class="ext-zones"><span>📉 Aşırı Short</span><span>Normal</span><span>Aşırı Long 📈</span></div>
    </div>

    <!-- COT TABLO -->
    <div class="table-box">
      <div class="table-hdr"><span class="table-ttl">Güncel Pozisyonlar</span><span class="table-badge">{d['date']}</span></div>
      <table><thead><tr>
        <th style="text-align:left">Kategori</th>
        <th>Long</th><th>Short</th><th>Net</th><th>Δ Long</th><th>Δ Short</th>
      </tr></thead><tbody>
        <tr><td>Büyük Spekülatörler</td><td>{fmt(d['specLong'])}</td><td>{fmt(d['specShort'])}</td>
          <td class="{'pos' if spec_net>=0 else 'neg'}">{fmts(spec_net)}</td>
          {chg_td(d['chgSL'])}{chg_td(d['chgSS'])}</tr>
        <tr class="sub"><td>Haftalık Δ</td>{chg_td(d['chgSL'])}{chg_td(d['chgSS'])}<td>—</td><td>—</td><td>—</td></tr>
        <tr><td>Ticari Oyuncular</td><td>{fmt(d['commLong'])}</td><td>{fmt(d['commShort'])}</td>
          <td class="{'pos' if comm_net>=0 else 'neg'}">{fmts(comm_net)}</td>
          {chg_td(d['chgCL'])}{chg_td(d['chgCS'])}</tr>
        <tr><td>Toplam</td><td>{fmt(d['totLong'])}</td><td>{fmt(d['totShort'])}</td>
          <td class="{'pos' if tot_net>=0 else 'neg'}">{fmts(tot_net)}</td><td>—</td><td>—</td></tr>
      </tbody></table>
    </div>

    <!-- GEÇMİŞ TABLO -->
    <div class="table-box">
      <div class="table-hdr"><span class="table-ttl">Geçmiş — Son {min(len(cot),12)} Hafta</span>
        <span class="table-badge">Spec pozisyonları</span></div>
      <table><thead><tr>
        <th style="text-align:left">Tarih</th>
        <th>Long</th><th>Short</th><th>Net</th><th>Δ Long</th><th>Δ Short</th><th>OI</th>
      </tr></thead><tbody>{hist_rows}</tbody></table>
    </div>

    <div class="note">
      Kaynak: <a href="https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm" target="_blank">CFTC</a>
      + Yahoo Finance · Üretim tarihi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} ·
      ML: {method_str} · {acc_str} · Yatırım tavsiyesi değildir.
    </div>"""


def generate_html(results):
    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    tabs_html = ""
    sections_html = ""
    for i, (com, section) in enumerate(results):
        active = "active" if i == 0 else ""
        tabs_html += (
            f'<button class="tab {active}" '
            f'onclick="show(\'{com["key"]}\')">'
            f'{com["emoji"]} {com["label"]}</button>'
        )
        sections_html += f'<div class="section {active}" id="sec-{com["key"]}">{section}</div>'

    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>COT ML Raporu — {gen_time}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="header">
  <div>
    <h1>COT ML Rapor Sistemi</h1>
    <div class="header-sub">CFTC + Yahoo Finance · Otomatik ML BIAS + Yön Tahmini</div>
  </div>
  <div class="gen-date">Üretim: {gen_time}</div>
</div>
<div class="tabs" id="tabs">{tabs_html}</div>
<div class="content" id="content">{sections_html}</div>
<script>
function show(key) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('sec-' + key).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    log("=" * 60)
    log(f"COT ML Rapor üretimi başladı — {datetime.datetime.now()}")

    results = []
    ok = fail = 0

    for com in COMMODITIES:
        log(f"\n[{ok+fail+1}/{len(COMMODITIES)}] {com['emoji']} {com['label']}")

        # COT verisi
        try:
            raw_rows = fetch_cot(com)
            if com["api"] == "legacy":
                cot = [parse_legacy(r) for r in raw_rows]
            else:
                cot = [parse_tff(r) for r in raw_rows]
            log(f"  ✓ COT: {len(cot)} hafta · Son tarih: {cot[0]['date']}")
        except Exception as e:
            log(f"  ✗ COT başarısız: {e}")
            fail += 1
            continue

        # Fiyat verisi
        price = None
        try:
            price = fetch_price(com["yf"])
            log(f"  ✓ Fiyat: {len(price)} hafta · Son: {price[0]['close']} ({price[0]['date']})")
        except Exception as e:
            log(f"  ⚠ Fiyat çekilemedi ({e}) — COT-only ML kullanılacak")

        # ML
        try:
            ml_res  = run_ml(cot, price)
            reasons, breakers, net_pct = build_reasons(cot, price, ml_res, com)
            log(f"  ✓ ML: BIAS={ml_res['bias']} · Yukarı=%{int(ml_res['prob_up']*100)} · "
                f"Eğitim={ml_res['train_size']}h · Doğruluk={'%'+str(ml_res['accuracy']) if ml_res['accuracy'] else 'N/A'}")

            section = render_commodity_html(com, cot, price, ml_res, reasons, breakers, net_pct)
            results.append((com, section))
            ok += 1
        except Exception as e:
            log(f"  ✗ ML/Render başarısız: {e}")
            import traceback; traceback.print_exc()
            fail += 1

    if not results:
        log("HATA: Hiç emtia işlenemedi!")
        sys.exit(1)

    # HTML yaz
    html = generate_html(results)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_FILE.stat().st_size // 1024
    log(f"\n✅ Rapor üretildi: {OUTPUT_FILE} ({size_kb} KB)")
    log(f"   {ok} emtia başarılı · {fail} başarısız")
    log("=" * 60)

if __name__ == "__main__":
    main()
