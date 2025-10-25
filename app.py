import io, os, re, json, base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageFile
import pytesseract

# ---------------- Tesseract path ----------------
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="AWK – Equivalent In-Situ CBR", layout="centered")
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- Responsive layout / mobile polish ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main { padding-top: .6rem; }
.block-container {
  max-width: 1000px;
  padding-left: 1.0rem; padding-right: 1.0rem;
  margin-left: auto; margin-right: auto;
}
.stButton > button { border-radius: 10px; padding: .65rem 1rem; }
@media (max-width: 640px) { .stButton > button { width: 100%; } }
h2 { text-align:center; letter-spacing:.3px; }
@media (max-width: 640px) {
  .block-container { padding-left:.6rem; padding-right:.6rem; }
  .stDataFrame { font-size: .9rem; }
}
.awk-banner { margin: 10px 0 14px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------- Logo ----------------
try:
    logo_path = r"C:\Users\AWK 4\AppData\Local\Programs\Python\Python312\Round LOGO AWK (1).jpg"
    with Image.open(logo_path) as img:
        max_width = 350
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        logo = img.resize((max_width, new_height))
        buffer = io.BytesIO()
        logo.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        st.markdown(
            f"<div style='text-align:center;'><img src='data:image/png;base64,{img_b64}' width='{max_width}'></div>",
            unsafe_allow_html=True,
        )
except Exception as e:
    st.warning(f"⚠️ Could not load logo: {e}")

# ---------------- Constants ----------------
CF_DEFAULTS = {300: 0.442, 455: 0.629, 610: 0.816}
LOAD_STEPS = [0, 10, 20, 30, 40, 50]
PRESSURE_UNITS = "kN/m²"
PRESSURE_LIBRARY = {
    300: {0: 0.00, 10: 145.97, 20: 288.01, 30: 435.70, 40: 577.17, 50: 718.64},
    455: {0: 0.00, 10:  64.32, 20: 126.07, 30: 190.27, 40: 251.78, 50: 313.28},
    610: {0: 0.00, 10:  36.43, 20:  71.08, 30: 106.51, 40: 140.73, 50: 174.95},
}

# ---------------- Helpers ----------------
def clean_num(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

def parse_dials_from_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        nums = [float(n.replace(",", ".")) for n in re.findall(r"[-+]?\d*\.?\d+", ln)]
        if len(nums) >= 3:
            rows.append(nums[:3])
    if len(rows) >= 6:
        return {L: rows[i] for i, L in enumerate(LOAD_STEPS)}
    return None

def pretty_cbr_plot(settle, pressure, bp125=None, compact=False):
    x = np.asarray(settle, float)
    y = np.asarray(pressure, float)

    x_max = max(3.0, np.ceil((np.nanmax(x) if x.size else 3.0) / 0.5) * 0.5)
    y_max = np.nanmax(y) if y.size else 350.0
    y_max = np.ceil((y_max * 1.05) / 50.0) * 50.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                             line=dict(color="red", width=3),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(symbol="diamond", size=10,
                    color="#1f77b4", line=dict(width=1.5, color="#173a6a")),
        hovertemplate="Settlement: %{x:.2f} mm<br>Pressure: %{y:.2f} kN/m²<extra></extra>",
        showlegend=False
    ))
    if bp125 is not None:
        fig.add_shape(type="line", x0=1.25, x1=1.25, y0=0, y1=y_max,
                      line=dict(color="rgba(255,0,0,0.35)", width=2, dash="dash"))

    # --- make it visually centered & tidy ---
    pad_lr = 60 if compact else 70  # equal left/right padding
    fig.update_layout(
        title=dict(
            text="MEAN BEARING PRESSURE VS SETTLEMENT",
            x=0.5, xanchor="center", y=0.95,  # center over figure
            font=dict(size=24 if compact else 28, family="Segoe UI, Arial Black")
        ),
        plot_bgcolor="white",
        margin=dict(l=pad_lr, r=pad_lr, t=90 if not compact else 70, b=60 if compact else 70),
        height=380 if compact else 520,
        font=dict(family="Segoe UI, Arial, Helvetica, sans-serif",
                  size=13 if compact else 14, color="#333"),
    )
    fig.update_xaxes(
        title="SETTLEMENT (MM)", range=[0, x_max], dtick=0.5,
        ticks="outside", ticklen=6, tickwidth=1,
        showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False,
        title_standoff=8
    )
    fig.update_yaxes(
        title="MEAN BEARING PRESSURE (kN/m²)", range=[0, y_max], dtick=50,
        ticks="outside", ticklen=6, tickwidth=1,
        showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False,
        title_standoff=8
    )
    return fig


def render_cbr_banner(cbr_pct: float, layer: str):
    """PASS/FAIL banner with round-up rule (14.5→15 pass, 29.5→30 pass)."""
    thresholds = {"Capping Layer": 15.0, "Sub-Base Layer": 30.0}
    rounded_value = round(float(cbr_pct), 1)      # display rounding
    # comparison uses whole-number rounding
    def is_pass(req): return round(rounded_value) >= round(req)

    if layer in thresholds:
        req = thresholds[layer]
        passed = is_pass(req)
        if passed:
            bg, border, text, label = "#e8f5e9", "#2e7d32", "#1b5e20", "PASS"
        else:
            bg, border, text, label = "#fdecea", "#f44336", "#b71c1c", "FAIL"
        status_line = f"{label} • Requirement: ≥{req:.0f}%"
    else:
        bg, border, text = "#e8f5e9", "#2e7d32", "#1b5e20"
        status_line = "Informational (no threshold)"

    st.markdown(
        f"""
        <div class="awk-banner" style="
            padding:14px 16px;
            border:1px solid {border};
            background:{bg};
            border-radius:12px;
            color:{text};
        ">
            <div style="font-size:14px; font-weight:600; letter-spacing:0.2px;">
                Equivalent In-Situ CBR
            </div>
            <div style="font-size:30px; font-weight:800; line-height:1.1; margin-top:4px;">
                {rounded_value:.1f}%
            </div>
            <div style="font-size:13px; margin-top:6px;">
                Layer: {layer} &nbsp;&nbsp; {status_line}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- OCR Functions ----------------
def easyocr_image_to_text(pil_img):
    """Offline Free AI OCR using EasyOCR (optional)."""
    try:
        import easyocr
    except Exception:
        return "__OCR_ERROR__: EasyOCR not installed"

    reader = easyocr.Reader(['en'], gpu=False)
    img_np = np.array(pil_img.convert("RGB"))
    result = reader.readtext(img_np, detail=0)
    nums = []
    for r in result:
        r = r.replace(",", ".").replace("O", "0").replace("o", "0")
        if re.fullmatch(r"\d+(\.\d+)?", r):
            nums.append(r)
    if len(nums) < 18:
        return "__OCR_ERROR__: Not enough numeric tokens"
    lines = [f"{nums[i*3]} {nums[i*3+1]} {nums[i*3+2]}" for i in range(6)]
    return "\n".join(lines)

def ocr_image_to_text(img):
    """Tesseract fallback OCR."""
    from pytesseract import Output
    img = img.convert("L").point(lambda x: 0 if x < 160 else 255, "1")
    cfg = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789.,-"
    df = pytesseract.image_to_data(img, config=cfg, output_type=Output.DATAFRAME)
    if df is None or "text" not in df.columns:
        return "__OCR_ERROR__: No text found"
    df["text"] = df["text"].astype(str).str.strip()
    tokens = (
        df["text"].str.replace(",", ".", regex=False)
        .str.findall(r"[-+]?\d*\.?\d+")
        .explode().dropna().tolist()
    )
    if len(tokens) < 18:
        return "__OCR_ERROR__: Not enough numeric tokens"
    nums = tokens[:18]
    lines = [f"{nums[i*3]} {nums[i*3+1]} {nums[i*3+2]}" for i in range(6)]
    return "\n".join(lines)

# ---------------- Math Functions ----------------
def interp_bp_at_1p25(settle, press):
    x, y = np.asarray(settle, float), np.asarray(press, float)
    above = np.where(x > 1.25)[0]
    below = np.where(x < 1.25)[0]
    if len(above) == 0 or len(below) == 0:
        return None, None, None, None, None
    i2 = above[0]; i1 = i2 - 1
    x1, y1, x2, y2 = x[i1], y[i1], x[i2], y[i2]
    m = (y2 - y1) / (x2 - x1); b = y1 - m * x1
    bp = m * 1.25 + b
    return bp, (x1, y1), (x2, y2), m, b

def k762(bp_at_1p25, cf): return int(np.rint((bp_at_1p25 / 0.00125) * cf))
def cbr_from_k(kint): return 6.1e-8 * (float(kint) ** 1.733)

# ---------------- UI ----------------
st.markdown("<h2>AWK – Equivalent In-Situ CBR</h2>", unsafe_allow_html=True)

colA, colB = st.columns(2)
with colA:
    plate = st.selectbox("Plate size (mm)", [300, 455, 610], index=0)
with colB:
    cf = st.number_input("Correction Factor (CF)", value=float(CF_DEFAULTS[plate]),
                         step=0.001, format="%.3f")

# Layer selector + mobile compact toggle
colC, colD = st.columns([2,1])
with colC:
    layer = st.selectbox("Layer / Test Type",
                         ["Formation", "Capping Layer", "Sub-Base Layer", "Other"], index=0)
with colD:
    mobile = st.toggle("📱 Compact layout", value=True,
                       help="Optimised height/margins for phone/tablet.")

method = st.radio("Input method", ["Manual entry", "Paste 6×3 dials", "Photo OCR (beta)"], index=0)
rows = []

if method == "Manual entry":
    for L in LOAD_STEPS:
        with st.expander(f"Load {L} kN", expanded=(L in [0, 10])):
            c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
            p_val = float(PRESSURE_LIBRARY[plate][L])
            c1.number_input(f"Bearing pressure @ {L} kN ({PRESSURE_UNITS})",
                            value=p_val, disabled=True, key=f"p_{L}")
            d1 = clean_num(c2.text_input("Dial 1 (mm)", "", key=f"d1_{L}"))
            d2 = clean_num(c3.text_input("Dial 2 (mm)", "", key=f"d2_{L}"))
            d3 = clean_num(c4.text_input("Dial 3 (mm)", "", key=f"d3_{L}"))
        rows.append([L, p_val, d1, d2, d3])

elif method == "Paste 6×3 dials":
    pasted = st.text_area("Paste 6×3 table", height=160)
    if st.button("Submit"):
        dmap = parse_dials_from_text(pasted)
        if not dmap:
            st.error("Could not parse 6×3 numbers."); st.stop()
        for L in LOAD_STEPS:
            rows.append([L, PRESSURE_LIBRARY[plate][L]] + dmap[L])

else:
    upl = st.file_uploader("Upload photo", type=["png", "jpg", "jpeg", "webp"])
    if not upl:
        st.info("Upload a clear photo of the dial readings.")
        st.stop()
    img = Image.open(io.BytesIO(upl.read()))
    st.image(img, caption="Uploaded image", use_column_width=True)
    use_easyocr = st.toggle("Use EasyOCR (Free AI OCR)", value=True)
    if use_easyocr:
        with st.spinner("Reading numbers with EasyOCR..."):
            txt = easyocr_image_to_text(img)
        if "__OCR_ERROR__" in txt:
            st.warning("EasyOCR struggled; falling back to Tesseract.")
            txt = ocr_image_to_text(img)
    else:
        with st.spinner("Running local OCR (Tesseract)..."):
            txt = ocr_image_to_text(img)
    st.text_area("OCR text (editable)", value=txt, height=200, key="ocr_text")
    if st.button("Submit OCR Data"):
        dmap = parse_dials_from_text(st.session_state["ocr_text"])
        if not dmap:
            st.error("❌ Couldn’t parse 6×3 readings."); st.stop()
        for L in LOAD_STEPS:
            rows.append([L, PRESSURE_LIBRARY[plate][L]] + dmap[L])

# ---------------- DataFrame ----------------
if not rows: st.stop()
df = pd.DataFrame(rows, columns=[
    "Load (kN)", f"Bearing Pressure ({PRESSURE_UNITS})", "Dial 1 (mm)", "Dial 2 (mm)", "Dial 3 (mm)"
])
df["Avg Dial (mm)"] = df[["Dial 1 (mm)", "Dial 2 (mm)", "Dial 3 (mm)"]].mean(axis=1)
avg0 = df.loc[df["Load (kN)"] == 0, "Avg Dial (mm)"].iloc[0]
df["Settlement (mm)"] = avg0 - df["Avg Dial (mm)"]

# ---------------- Calculations ----------------
bp125, p1a, p1b, m, b = interp_bp_at_1p25(
    df["Settlement (mm)"], df[f"Bearing Pressure ({PRESSURE_UNITS})"]
)
if bp125 is None:
    st.warning("1.25 mm outside range."); st.stop()

kint = k762(bp125, cf)
cbr_pct = round(cbr_from_k(kint), 1)

# ---------------- Results banner ----------------
render_cbr_banner(cbr_pct, layer)

# Compact numeric summary
st.write(
    f"**BP@1.25mm (interpolated):** {bp125:.2f} {PRESSURE_UNITS}  "
    f"**k₇₆₂:** {kint:,}  **CF:** {cf:.3f}  **Plate:** {plate} mm"
)

# ---------------- Plot (right under banner) ----------------
fig = pretty_cbr_plot(
    df["Settlement (mm)"],
    df[f"Bearing Pressure ({PRESSURE_UNITS})"],
    bp125=bp125,
    compact=mobile,
)
st.plotly_chart(fig, use_container_width=True, key="cbr_main_plot")

# ---------------- Tables ----------------
bp_map = pd.DataFrame({
    "Load (kN)": LOAD_STEPS,
    f"Bearing Pressure ({PRESSURE_UNITS})": [PRESSURE_LIBRARY[plate][L] for L in LOAD_STEPS],
})
with st.expander("Load → Bearing Pressure used", expanded=True):
    st.dataframe(bp_map, use_container_width=True, height=260)

with st.expander("Readings table", expanded=False):
    st.dataframe(
        df[["Load (kN)", "Avg Dial (mm)", "Settlement (mm)", f"Bearing Pressure ({PRESSURE_UNITS})"]],
        use_container_width=True, height=280
    )

# Points used for interpolation
df_sorted = df.sort_values("Settlement (mm)")
lower = df_sorted[df_sorted["Settlement (mm)"] <= 1.25].tail(1)
upper = df_sorted[df_sorted["Settlement (mm)"] >= 1.25].head(1)
if not lower.empty and not upper.empty:
    bracket = pd.concat([lower, upper], ignore_index=True)[
        ["Load (kN)", "Settlement (mm)", f"Bearing Pressure ({PRESSURE_UNITS})"]
    ]
    bracket.insert(0, "Role", ["Before 1.25 mm", "After 1.25 mm"])
    with st.expander("Points used for interpolation at 1.25 mm", expanded=True):
        st.dataframe(bracket, use_container_width=True, height=200)
