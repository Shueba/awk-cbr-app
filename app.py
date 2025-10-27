import io, os, re, base64, shutil, platform
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import pytesseract
import streamlit.components.v1 as components

# =========================
# App & environment
# =========================
st.set_page_config(page_title="AWK - Equivalent In-Situ CBR", layout="centered")

def set_tesseract_path():
    if platform.system().lower().startswith("win"):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        cmd = shutil.which("tesseract")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
set_tesseract_path()

# =========================
# HTML injector (top-level preferred; fallback to iframe)
# =========================
def _html(snippet: str):
    # Prefer top-level st.html so JS can reach Streamlit widgets
    if hasattr(st, "html"):
        try:
            st.html(snippet)  # no height kwarg for max compatibility
            return
        except TypeError:
            pass
    # Fallback (iframe) - JS may not reach Streamlit inputs, but app won't crash
    components.html(snippet, height=0, scrolling=False)

# =========================
# Password protection
# =========================
def check_password():
    pw = st.secrets.get("APP_PASSWORD")

    if st.session_state.get("auth_ok"):
        return True

    st.markdown(
        """
        <div style='text-align:center; padding:28px 8px 6px;'>
            <h3 style='color:#003366; margin:0;'>WELCOME TO AWK GROUND TESTING APP</h3>
            <p style='font-size:15px; color:#333; max-width:540px; margin:8px auto 20px;'>
                Use this tool on site to enter dial readings and view the Equivalent In-Situ CBR,
                plots, and tables.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("login_main", clear_on_submit=False):
        entered = st.text_input("Password", type="password", placeholder="Enter password here")
        ok = st.form_submit_button("Sign in", use_container_width=True)

    if ok:
        if pw and entered == pw:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    if not pw:
        st.info("APP_PASSWORD not set; app is currently unlocked.")
        return True

    return False

if not check_password():
    st.stop()

# =========================
# Mobile polish CSS
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main { padding-top: .6rem; }
.block-container { max-width: 1000px; padding-left: 1.0rem; padding-right: 1.0rem; margin: 0 auto; }
.stButton > button { border-radius: 10px; padding: .65rem 1rem; }
@media (max-width: 640px) { .stButton > button { width: 100%; } }
h2 { text-align:center; letter-spacing:.3px; }
@media (max-width: 640px) {
  .block-container { padding-left:.6rem; padding-right:.6rem; }
  .stDataFrame { font-size: .9rem; }
}
.awk-banner { margin: 10px 0 14px 0; }

/* If any number_input remains, hide steppers just in case */
[data-testid="stNumberInput"] button[aria-label="Increment"],
[data-testid="stNumberInput"] button[aria-label="Decrement"] { display: none !important; }
[data-testid="stNumberInput"] label { display: none !important; }
[data-testid="stNumberInput"] input { padding: 10px 12px !important; text-align: left !important; }
[data-testid="stNumberInput"] input::placeholder { color: #bbb; }
</style>
""", unsafe_allow_html=True)

# =========================
# Logo
# =========================
def render_logo(max_width=350):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = ["Round AWK Logo.jpg", "awk_logo.jpg", "Round LOGO AWK (1).jpg"]
    logo_path = next((os.path.join(base_dir, name) for name in candidates
                      if os.path.isfile(os.path.join(base_dir, name))), None)
    if not logo_path:
        st.warning(f"Could not load logo: tried {candidates} in {base_dir}")
        return
    try:
        with Image.open(logo_path) as img:
            ratio = max_width / float(img.width)
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        st.markdown(
            f"""<div style="text-align:center;">
                    <img src="data:image/png;base64,{img_b64}" width="{max_width}" />
                </div>""",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not load logo: {e}")

render_logo()

# =========================
# Constants
# =========================
CF_DEFAULTS = {300: 0.442, 455: 0.629, 610: 0.816}
LOAD_STEPS = [0, 10, 20, 30, 40, 50]
PRESSURE_UNITS = "kN/m^2"  # ASCII only
PRESSURE_LIBRARY = {
    300: {0: 0.00, 10: 145.97, 20: 288.01, 30: 435.70, 40: 577.17, 50: 718.64},
    455: {0: 0.00, 10:  64.32, 20: 126.07, 30: 190.27, 40: 251.78, 50: 313.28},
    610: {0: 0.00, 10:  36.43, 20:  71.08, 30: 106.51, 40: 140.73, 50: 174.95},
}

# =========================
# Helpers & math
# =========================
def _to_float(x) -> float:
    """Safely convert a string (allowing ',' or '.') to float. Empty/None -> 0.0."""
    if x is None:
        return 0.0
    x = str(x).strip().replace(",", ".")
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0

def ios_number_input(label: str, key: str, value: str = "", allow_decimal: bool = True, placeholder: str = "") -> str:
    """
    Streamlit text_input that forces iPhone's numeric keypad.
    Uses type="tel" + inputmode and re-applies after rerenders.
    Keep label UNIQUE per widget.
    """
    s = st.text_input(label, value=value, key=key, placeholder=placeholder, label_visibility="visible")

    # Build the JS once per field
    js = f"""
<script>
(function(){{
  // one global observer to re-apply patches on Streamlit rerenders
  if(!window._awk_obs){{
    window._awk_patchers = [];
    window._awk_obs = new MutationObserver(function(){{
      window._awk_patchers.forEach(fn => fn());
    }});
    window._awk_obs.observe(document.body, {{ childList: true, subtree: true }});
  }}

  const wanted = {label!r};
  const allowDecimal = {str(allow_decimal).lower()};

  function patchThisField(){{
    // locate this input by its aria-label (Streamlit sets it from the label)
    const el = Array.from(document.querySelectorAll('input[aria-label]'))
      .find(i => (i.getAttribute('aria-label') || '') === wanted);
    if(!el) return;

    // Force iOS keypad: type=tel + inputmode (iOS respects at least one of these)
    try {{ el.type = 'tel'; }} catch(e) {{}}
    el.setAttribute('inputmode', allowDecimal ? 'decimal' : 'numeric');
    el.setAttribute('autocapitalize','off');
    el.setAttribute('autocomplete','off');
    el.setAttribute('autocorrect','off');
    el.setAttribute('spellcheck','false');
    el.setAttribute('enterkeyhint','done');
    el.setAttribute('pattern','[0-9]*');

    // Live sanitise value
    if(!el._awk_patched){{
      el.addEventListener('input', function(){{
        let v = el.value || "";
        v = v.replace(/[^0-9.]/g, "");
        if(!allowDecimal) v = v.replace(/[.]/g, "");
        const p = v.indexOf(".");
        if(p !== -1) v = v.slice(0, p+1) + v.slice(p+1).replace(/[.]/g, "");
        el.value = v;
      }});
      el._awk_patched = true;
    }}
  }}

  // register this field's patcher and run it now and shortly after
  window._awk_patchers.push(patchThisField);
  patchThisField();
  setTimeout(patchThisField, 0);
  setTimeout(patchThisField, 120);
}})();
</script>
"""
    _html(js)  # uses st.html if available; otherwise components.html fallback

    # Python-side safety net
    s = (s or "").replace(",", ".")
    s = re.sub(r"[^0-9.]", "", s)
    if not allow_decimal:
        s = s.replace(".", "")
    if s.count(".") > 1:
        i = s.find("."); s = s[:i+1] + s[i+1:].replace(".", "")
    return s

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
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="red", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(symbol="diamond", size=10, color="#1f77b4", line=dict(width=1.5, color="#173a6a")),
        hovertemplate="Settlement: %{x:.2f} mm<br>Pressure: %{y:.2f} kN/m^2<extra></extra>",
        showlegend=False
    ))
    if bp125 is not None:
        fig.add_shape(type="line", x0=1.25, x1=1.25, y0=0, y1=y_max,
                      line=dict(color="rgba(255,0,0,0.35)", width=2, dash="dash"))
    pad_lr = 60 if compact else 70
    fig.update_layout(
        title=dict(text="MEAN BEARING PRESSURE VS SETTLEMENT", x=0.5, xanchor="center", y=0.95,
                   font=dict(size=24 if compact else 28, family="Segoe UI, Arial Black")),
        plot_bgcolor="white",
        margin=dict(l=pad_lr, r=pad_lr, t=90 if not compact else 70, b=60 if compact else 70),
        height=380 if compact else 520,
        font=dict(family="Segoe UI, Arial, Helvetica, sans-serif", size=13 if compact else 14, color="#333"),
    )
    fig.update_xaxes(title="SETTLEMENT (MM)", range=[0, x_max], dtick=0.5, ticks="outside",
                     ticklen=6, tickwidth=1, showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False)
    fig.update_yaxes(title="MEAN BEARING PRESSURE (kN/m^2)", range=[0, y_max], dtick=50, ticks="outside",
                     ticklen=6, tickwidth=1, showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False)
    return fig

def render_cbr_banner(cbr_pct: float, layer: str):
    thresholds = {"Capping Layer": 15.0, "Sub-Base Layer": 30.0}
    rounded_value = round(float(cbr_pct), 1)
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

def ocr_image_to_text(img):
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

def easyocr_image_to_text(pil_img):
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

def k762(bp_at_1p25, cf):
    return int(np.rint((bp_at_1p25 / 0.00125) * cf))

def cbr_from_k(kint):
    return 6.1e-8 * (float(kint) ** 1.733)

# =========================
# UI - Controls
# =========================
st.markdown("<h2>AWK - Equivalent In-Situ CBR</h2>", unsafe_allow_html=True)

colA, colB = st.columns(2)
with colA:
    plate = st.selectbox("Plate size (mm)", [300, 455, 610], index=0)
with colB:
    cf = st.number_input("Correction Factor (CF)", value=float(CF_DEFAULTS[plate]), step=0.001, format="%.3f")

colC, colD = st.columns([2, 1])
with colC:
    layer = st.selectbox("Layer / Test Type", ["Formation", "Capping Layer", "Sub-Base Layer", "Other"], index=0)
with colD:
    mobile = st.toggle("Compact layout", value=True, help="Optimised height/margins for phone/tablet.")

method = st.radio("Input method", ["Manual entry", "Paste 6x3 dials", "Photo OCR (beta)"], index=0)

rows = []

# =========================
# Manual entry (raw HTML with iPhone keypad) -> pipes into "Paste 6x3 dials"
# =========================
if method == "Manual entry":
    st.markdown("#### Dial gauge inputs (mm)")

    # 1) Render the target textarea+button (hidden) that our HTML will fill & submit
    #    We reuse your existing "Paste 6x3 dials" parser by targeting these widgets.
    with st.container():
        pasted = st.text_area("Paste 6x3 table", key="hidden_paste_box", height=160, placeholder="", label_visibility="collapsed")
        col_hidden = st.columns([1, 3])[0]
        with col_hidden:
            submit_hidden = st.button("Submit", key="hidden_submit_button", use_container_width=False)

    # 2) Draw raw HTML numeric inputs (6 rows x 3 columns), iPhone keypad guaranteed
    #    On "Use these values", JS packs values as 6 lines x 3 numbers and
    #    writes them into the hidden textarea, then clicks the hidden submit.
    #    Labels are ASCII-only; no fancy characters.
    inputs_html = """
<style>
  .awk-wrap{max-width:860px;margin:0 auto}
  table.awk {width:100%; border-collapse:collapse; margin-top:8px}
  table.awk th, table.awk td {border-bottom:1px solid #eee; padding:8px 6px; text-align:center}
  table.awk th{background:#f7f9fc; font-weight:700}
  input.dial {
    width:100%; font-size:18px; padding:10px 12px; border-radius:10px;
    border:2px solid #d5d9e0; outline:none; text-align:right; background:#fff;
  }
  input.dial:focus{border-color:#3b82f6; box-shadow:0 0 0 3px rgba(59,130,246,.15)}
  .actions{display:flex; gap:10px; margin-top:12px; justify-content:flex-end}
  .btn {
    padding:10px 14px; border-radius:10px; border:0; background:#2563eb; color:#fff; font-weight:700; cursor:pointer;
  }
  .btn.secondary { background:#6b7280; }
</style>

<div class="awk-wrap">
  <table class="awk">
    <thead>
      <tr>
        <th>Load (kN)</th>
        <th>Dial 1 (mm)</th>
        <th>Dial 2 (mm)</th>
        <th>Dial 3 (mm)</th>
      </tr>
    </thead>
    <tbody>
      <!-- rows for 0,10,20,30,40,50 kN -->
      <tr><td>0</td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_0_1" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_0_2" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_0_3" placeholder=""></td>
      </tr>
      <tr><td>10</td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_10_1" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_10_2" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_10_3" placeholder=""></td>
      </tr>
      <tr><td>20</td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_20_1" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_20_2" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_20_3" placeholder=""></td>
      </tr>
      <tr><td>30</td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_30_1" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_30_2" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_30_3" placeholder=""></td>
      </tr>
      <tr><td>40</td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_40_1" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_40_2" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_40_3" placeholder=""></td>
      </tr>
      <tr><td>50</td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_50_1" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_50_2" placeholder=""></td>
        <td><input class="dial" inputmode="decimal" pattern="[0-9]*" id="d_50_3" placeholder=""></td>
      </tr>
    </tbody>
  </table>

  <div class="actions">
    <button class="btn secondary" type="button" id="clear_all">Clear</button>
    <button class="btn" type="button" id="use_values">Use these values</button>
  </div>
</div>

<script>
(function(){
  function clean(v, allowDecimal=true){
    v = (v||"").replace(/,/g,".");
    v = v.replace(/[^0-9.]/g,"");
    if(!allowDecimal) v = v.replace(/[.]/g,"");
    const p = v.indexOf(".");
    if (p !== -1) v = v.slice(0, p+1) + v.slice(p+1).replace(/[.]/g,"");
    return v;
  }

  // live sanitize
  document.querySelectorAll("input.dial").forEach(el=>{
    el.addEventListener("input", ()=>{ el.value = clean(el.value, true); });
  });

  document.getElementById("clear_all").addEventListener("click", ()=>{
    document.querySelectorAll("input.dial").forEach(el=>{ el.value=""; });
  });

  document.getElementById("use_values").addEventListener("click", ()=>{
    const loads = [0,10,20,30,40,50];
    const lines = [];
    for(const L of loads){
      const a = clean(document.getElementById(`d_${L}_1`).value, true);
      const b = clean(document.getElementById(`d_${L}_2`).value, true);
      const c = clean(document.getElementById(`d_${L}_3`).value, true);
      if(a==="" || b==="" || c===""){ alert("Please fill all three dials for each load."); return; }
      lines.push(`${a} ${b} ${c}`);
    }
    const blob = lines.join("\\n");

    // Find Streamlit's hidden textarea and submit button by their labels/keys
    // The "hidden_paste_box" has aria-label equal to "Paste 6x3 table"
    const ta = Array.from(document.querySelectorAll('textarea[aria-label]'))
      .find(t => (t.getAttribute('aria-label')||'') === 'Paste 6x3 table');
    if(!ta){ alert("Could not locate the Paste 6x3 box in the page."); return; }
    ta.value = blob;
    ta.dispatchEvent(new Event('input', { bubbles: true }));

    // Click the hidden Submit button
    const btn = Array.from(document.querySelectorAll('button'))
      .find(b => (b.innerText||'').trim() === 'Submit');
    if(btn){ btn.click(); }
  });
})();
</script>
"""

    st.html(inputs_html)  # top-level injection so JS sees Streamlit widgets

    # If user clicked the hidden submit, build rows from the filled textarea
    if submit_hidden:
        dmap = parse_dials_from_text(st.session_state.get("hidden_paste_box", ""))
        if not dmap:
            st.error("Could not parse values from the manual keypad.")
            st.stop()
        for L in LOAD_STEPS:
            rows.append([L, PRESSURE_LIBRARY[plate][L]] + dmap[L])

# =========================
# Paste 6x3 (unchanged)
# =========================
elif method == "Paste 6x3 dials":
    pasted = st.text_area("Paste 6x3 table", height=160,
                          placeholder="e.g.\n25.00 25.00 25.00\n24.00 24.00 23.56\n...")
    if st.button("Submit"):
        dmap = parse_dials_from_text(pasted)
        if not dmap:
            st.error("Could not parse 6x3 numbers.")
            st.stop()
        for L in LOAD_STEPS:
            rows.append([L, PRESSURE_LIBRARY[plate][L]] + dmap[L])
    else:
        st.stop()

# =========================
# OCR (unchanged)
# =========================
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
            st.error("Couldn't parse 6x3 readings.")
            st.stop()
        for L in LOAD_STEPS:
            rows.append([L, PRESSURE_LIBRARY[plate][L]] + dmap[L])
    else:
        st.stop()


# =========================
# Build DataFrame
# =========================
if not rows:
    st.stop()

df = pd.DataFrame(rows, columns=[
    "Load (kN)", f"Bearing Pressure ({PRESSURE_UNITS})", "Dial 1 (mm)", "Dial 2 (mm)", "Dial 3 (mm)"
])
df["Avg Dial (mm)"] = df[["Dial 1 (mm)", "Dial 2 (mm)", "Dial 3 (mm)"]].mean(axis=1)

# Settlement relative to the 0 kN average dial
avg0 = df.loc[df["Load (kN)"] == 0, "Avg Dial (mm)"].iloc[0]
df["Settlement (mm)"] = avg0 - df["Avg Dial (mm)"]

# =========================
# Calculations
# =========================
bp125, p1a, p1b, m, b = interp_bp_at_1p25(df["Settlement (mm)"], df[f"Bearing Pressure ({PRESSURE_UNITS})"])
if bp125 is None:
    st.warning("1.25 mm outside range. Ensure 1.25 mm is bracketed by your readings.")
    st.stop()

kint = k762(bp125, cf)
cbr_pct = round(cbr_from_k(kint), 1)

# =========================
# Results & Plot
# =========================
render_cbr_banner(cbr_pct, layer)

max_settlement = df["Settlement (mm)"].max()
st.write(
    f"**BP@1.25mm (interpolated):** {bp125:.2f} {PRESSURE_UNITS}    "
    f"**k762:** {kint:,}    **CF:** {cf:.3f}    **Plate:** {plate} mm    "
    f"**Max Settlement:** {max_settlement:.2f} mm"
)

fig = pretty_cbr_plot(
    df["Settlement (mm)"], df[f"Bearing Pressure ({PRESSURE_UNITS})"],
    bp125=bp125, compact=mobile,
)
st.plotly_chart(fig, use_container_width=True, key="cbr_main_plot")

# =========================
# Tables
# =========================
def format_df(df_in, decimals=2):
    df2 = df_in.copy()
    for col in df2.select_dtypes(include=["float", "float64", "int"]):
        df2[col] = df2[col].apply(lambda x: f"{x:.{decimals}f}")
    return df2

bp_map = pd.DataFrame({
    "Load (kN)": LOAD_STEPS,
    f"Bearing Pressure ({PRESSURE_UNITS})": [PRESSURE_LIBRARY[plate][L] for L in LOAD_STEPS],
})
with st.expander("Load -> Bearing Pressure used", expanded=False):
    st.dataframe(format_df(bp_map), use_container_width=True, height=260)

with st.expander("Readings table", expanded=True):
    st.dataframe(
        format_df(df[["Load (kN)", "Avg Dial (mm)", "Settlement (mm)", f"Bearing Pressure ({PRESSURE_UNITS})"]]),
        use_container_width=True, height=280
    )

df_sorted = df.sort_values("Settlement (mm)")
lower = df_sorted[df_sorted["Settlement (mm)"] <= 1.25].tail(1)
upper = df_sorted[df_sorted["Settlement (mm)"] >= 1.25].head(1)
if not lower.empty and not upper.empty:
    bracket = pd.concat([lower, upper], ignore_index=True)[
        ["Load (kN)", "Settlement (mm)", f"Bearing Pressure ({PRESSURE_UNITS})"]
    ]
    bracket.insert(0, "Role", ["Before 1.25 mm", "After 1.25 mm"])
    with st.expander("Points used for interpolation at 1.25 mm", expanded=True):
        st.dataframe(format_df(bracket), use_container_width=True, height=200)
