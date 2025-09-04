# Import Standard Libraries
import vars
import modeller
import forecaster
import numpy as np
import pandas as pd
import warnings
import sys, io
import html
from collections import deque
from concurrent.futures import ThreadPoolExecutor
warnings.simplefilter('ignore', category=pd.errors.SettingWithCopyWarning)

# Import Bokeh Modules
from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import ColorPicker, CheckboxGroup
from bokeh.models import (
    ColumnDataSource, TextInput, TextAreaInput, Slider, Button, Div,
    HoverTool, Span, RangeSlider, CheckboxGroup, Segment
)
from bokeh.plotting import figure

# Globals to allow unbinding if needed
_ORIG_STDOUT = None
_ORIG_STDERR = None
_NOTES_LOGGER = None

_executor = ThreadPoolExecutor(max_workers=1)
_training_future = None
_training_busy = False
_likelihood_cache = None

# Holds the visible forecast window endpoints for the segments
_forecast_x0 = None  # datetime at first forecast month (after last historic)
_forecast_x1 = None  # datetime at last forecast month

# --- consistent styling ---
THR_DASH = "dotted"     # match the boundary_rule style
HI_COLOR = "red"        # high-risk
STR_COLOR = "green"     # strong
THR_WIDTH = 2           # bump if you want bolder lines

# Default thresholds taken from vars.py (keep these static for the rectangles)
DEFAULT_THR_PRICE_LOW  = vars.high_risk_price_change
DEFAULT_THR_PRICE_HIGH = vars.strong_market_price_change

DEFAULT_THR_DOM_LOW    = vars.strong_market_days_on_market
DEFAULT_THR_DOM_HIGH   = vars.high_risk_days_on_market

DEFAULT_THR_VIEW_LOW   = vars.high_risk_view_count_change
DEFAULT_THR_VIEW_HIGH  = vars.strong_market_view_count_change

# Rectangle default heights per series
RECT_H_PRICE = 0.10
RECT_H_DOM   = 30.0
RECT_H_VIEW  = 0.15

# Load Dataset
supply = pd.read_csv(vars.supply_data_path)
demand = pd.read_csv(vars.demand_data_path)

supply_zips = set(supply['postal_code'].unique())
demand_zips = set(demand['postal_code'].unique())

# Clean up dataframes such that we only keep records where zip codes are in all three: supply, demand, and crosswalk.
zip_intersection = supply_zips & demand_zips
sup_df = supply[supply['postal_code'].isin(zip_intersection)].copy()
dem_df = demand[demand['postal_code'].isin(zip_intersection)].copy()

# Drop Overlapping columns in dem
join_keys = ['month_date_yyyymm', 'postal_code']
overlap = [c for c in dem_df.columns if c in sup_df.columns and c not in join_keys]

# Drop them from demand side before merge
dem_df_nodup = dem_df.drop(columns=overlap)

# Decision Point: Lets make the decision to only keep zip codes and months that are present in both supply and demand.
df = sup_df.merge(dem_df_nodup, 'inner', on=join_keys)

# Generate final_df for pulling records
anchor = pd.Period(vars.start_date, freq='M')
last_date = pd.Period(vars.end_date, freq='M')
required_months = {anchor - i for i in range(vars.check_window_months)}  # set for fast membership

# Add a Period month column
df2 = df.copy()
df2['month'] = pd.to_datetime(df['month_date_yyyymm'], format='%Y%m').dt.to_period('M')
# Keep only rows in the required 36-month window
df_win = df2[df2['month'].isin(required_months)]

# Count distinct months per postal_code within the window
counts = df_win.groupby('postal_code')['month'].nunique()

# Postal codes with all vars.min_records months present (consecutive by construction of the window)
valid_postals = counts[counts >= vars.min_acceptable_records].index

# Final result: only those postal_codes and only those 36 months
final_df = (
    df2[
        (df2['postal_code'].isin(valid_postals)) &
        (df2['month'] >= last_date)]
    .sort_values(['postal_code', 'month'], ascending=[True, False])
)

# ==========================================
# Helpers: build the outcome_trend dataframe
# ==========================================
def _bridge_series(hist_df: pd.DataFrame, fcst_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Prepend the last non-NaN historic point to the forecast series for column `col`.
    Ensures the forecast line connects to history.
    """
    hv = hist_df.dropna(subset=[col])
    if hv.empty:
        return fcst_df[["month", col]].copy()

    last_m = pd.to_datetime(hv["month"].iloc[-1])
    last_y = float(hv[col].iloc[-1])
    bridge = pd.DataFrame({"month": [last_m], col: [last_y]})

    # Keep forecast months as-is (they already start at next month)
    fc = fcst_df[["month", col]].copy()
    return pd.concat([bridge, fc], ignore_index=True)

# --- Logging-to-Notes binding ---
def _set_notes_widget_content(widget, msg: str):
    """Set text on either TextAreaInput(.value) or Div(.text)."""
    if isinstance(widget, Div):
        # preserve newlines & use monospace; escape HTML
        widget.text = (
            "<div style='white-space:pre-wrap; font-family:monospace;'>"
            f"{html.escape(msg)}"
            "</div>"
        )
    elif isinstance(widget, TextAreaInput):
        widget.value = msg
    else:
        # fallback for any other widget with .value
        if hasattr(widget, "value"):
            widget.value = msg

class NotesTee(io.TextIOBase):
    """
    File-like object that:
      - tees to the original stream (optional)
      - appends lines to a Bokeh TextAreaInput (notes) with a max line cap
    """
    def __init__(self, widget, max_lines=500, tee_stream=None, timestamp=True):
        self.widget = widget
        self.lines = deque(maxlen=max_lines)
        self.buffer = ""   # hold partial line fragments
        self.tee = tee_stream
        self.timestamp = timestamp

    def write(self, data):
        if self.tee is not None:
            try: self.tee.write(data)
            except Exception: pass

        if not data:
            return 0
        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if self.timestamp:
                ts = pd.Timestamp.now().strftime("%H:%M:%S")
                line = f"[{ts}] {line}"
            self.lines.append(line)

        msg = "\n".join(self.lines)
        curdoc().add_next_tick_callback(lambda m=msg: _set_notes_widget_content(self.widget, m))
        return len(data)

    def flush(self):
        if self.tee is not None:
            try:
                self.tee.flush()
            except Exception:
                pass

def bind_prints_to_notes(widget, max_lines=500, timestamp=True, tee=True):
    """
    Redirect sys.stdout/sys.stderr to the Notes widget (textarea).
    Set tee=False to suppress console output.
    """
    global _ORIG_STDOUT, _ORIG_STDERR, _NOTES_LOGGER
    _ORIG_STDOUT, _ORIG_STDERR = sys.stdout, sys.stderr
    tee_out = _ORIG_STDOUT if tee else None
    # Use ONE NotesTee for both stdout and stderr so ordering stays consistent
    logger = NotesTee(widget, max_lines=max_lines, tee_stream=tee_out, timestamp=timestamp)
    sys.stdout = logger
    sys.stderr = logger
    _NOTES_LOGGER = logger
    return logger

def unbind_prints_from_notes():
    global _ORIG_STDOUT, _ORIG_STDERR, _NOTES_LOGGER
    if _ORIG_STDOUT is not None:
        sys.stdout = _ORIG_STDOUT
    if _ORIG_STDERR is not None:
        sys.stderr = _ORIG_STDERR
    _NOTES_LOGGER = None

def clear_notes():
    global _NOTES_LOGGER
    if _NOTES_LOGGER is not None:
        _NOTES_LOGGER.lines.clear()
        _NOTES_LOGGER.buffer = ""
    notes.value = ""


#### BEGIN BOKEH APP ####
# Defaults
ALL_ZIPS = sorted(list(final_df["postal_code"].unique()))
DEFAULT_ZIP = str(ALL_ZIPS[0]) if ALL_ZIPS else "00000"
DEFAULT_HORIZON = 12
DEFAULT_K = 3

def empty_source():
    return ColumnDataSource(dict(month=[], y=[]))

src_hist_price = empty_source()
src_fore_price = empty_source()
src_hist_dom = empty_source()
src_fore_dom = empty_source()
src_hist_view = empty_source()
src_fore_view = empty_source()
# threshold line sources (each as a single horizontal Segment over the forecast window)
src_thr_price_hi   = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[]))  # high risk (price_change)
src_thr_price_str  = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[]))  # strong market (price_change)

src_thr_dom_hi     = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[]))  # high risk (days_on_market)
src_thr_dom_str    = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[]))  # strong market (days_on_market)

src_thr_view_hi    = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[]))  # high risk (view_count_change)
src_thr_view_str   = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[]))  # strong market (view_count_change)

# Each source holds 12 quads per side (upper/lower) over the forecast window
src_rect_price_up = ColumnDataSource(dict(left=[], right=[], top=[], bottom=[], fill_color=[], fill_alpha=[], line_color=[], line_alpha=[]))
src_rect_price_lo = ColumnDataSource(dict(left=[], right=[], top=[], bottom=[], fill_color=[], fill_alpha=[], line_color=[], line_alpha=[]))

src_rect_dom_up   = ColumnDataSource(dict(left=[], right=[], top=[], bottom=[], fill_color=[], fill_alpha=[], line_color=[], line_alpha=[]))
src_rect_dom_lo   = ColumnDataSource(dict(left=[], right=[], top=[], bottom=[], fill_color=[], fill_alpha=[], line_color=[], line_alpha=[]))

src_rect_view_up  = ColumnDataSource(dict(left=[], right=[], top=[], bottom=[], fill_color=[], fill_alpha=[], line_color=[], line_alpha=[]))
src_rect_view_lo  = ColumnDataSource(dict(left=[], right=[], top=[], bottom=[], fill_color=[], fill_alpha=[], line_color=[], line_alpha=[]))


TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p1 = figure(height=220, sizing_mode="stretch_width", x_axis_type="datetime",
            tools=TOOLS, title="Price Change (historic & projected)")
p2 = figure(height=220, sizing_mode="stretch_width", x_axis_type="datetime",
            tools=TOOLS, title="Days on Market (historic & projected)")
p3 = figure(height=220, sizing_mode="stretch_width", x_axis_type="datetime",
            tools=TOOLS, title="View Count Change (historic & projected)")

# Share x-range
p2.x_range = p1.x_range
p3.x_range = p1.x_range

# Lines
h1 = p1.line("month", "y", source=src_hist_price, line_width=2, legend_label="historic")
f1 = p1.line("month", "y", source=src_fore_price, line_width=2, line_dash="dashed", legend_label="forecast")
h2 = p2.line("month", "y", source=src_hist_dom, line_width=2, legend_label="historic")
f2 = p2.line("month", "y", source=src_fore_dom, line_width=2, line_dash="dashed", legend_label="forecast")
h3 = p3.line("month", "y", source=src_hist_view, line_width=2, legend_label="historic")
f3 = p3.line("month", "y", source=src_fore_view, line_width=2, line_dash="dashed", legend_label="forecast")

# Price change thresholds
p1.segment('x0','y0','x1','y1', source=src_thr_price_hi,
           line_dash=THR_DASH, line_color=HI_COLOR, line_width=THR_WIDTH)
p1.segment('x0','y0','x1','y1', source=src_thr_price_str,
           line_dash=THR_DASH, line_color=STR_COLOR, line_width=THR_WIDTH)

# Days on market thresholds
p2.segment('x0','y0','x1','y1', source=src_thr_dom_hi,
           line_dash=THR_DASH, line_color=HI_COLOR, line_width=THR_WIDTH)
p2.segment('x0','y0','x1','y1', source=src_thr_dom_str,
           line_dash=THR_DASH, line_color=STR_COLOR, line_width=THR_WIDTH)

# View count change thresholds
p3.segment('x0','y0','x1','y1', source=src_thr_view_hi,
           line_dash=THR_DASH, line_color=HI_COLOR, line_width=THR_WIDTH)
p3.segment('x0','y0','x1','y1', source=src_thr_view_str,
           line_dash=THR_DASH, line_color=STR_COLOR, line_width=THR_WIDTH)

# Price rectangles (independent quads; 12 up + 12 down)
r_price_up = p1.quad(left='left', right='right', top='top', bottom='bottom',
                     source=src_rect_price_up,  fill_color="green", fill_alpha=0.20, line_color=None)
r_price_lo = p1.quad(left='left', right='right', top='top', bottom='bottom',
                     source=src_rect_price_lo,  fill_color="red",   fill_alpha=0.20, line_color=None)

# DOM rectangles
r_dom_up = p2.quad(left='left', right='right', top='top', bottom='bottom',
                   source=src_rect_dom_up,     fill_color="red",   fill_alpha=0.20, line_color=None)
r_dom_lo = p2.quad(left='left', right='right', top='top', bottom='bottom',
                   source=src_rect_dom_lo,     fill_color="green", fill_alpha=0.20, line_color=None)

# View-change rectangles
r_view_up = p3.quad(left='left', right='right', top='top', bottom='bottom',
                    source=src_rect_view_up,    fill_color="green", fill_alpha=0.20, line_color=None)
r_view_lo = p3.quad(left='left', right='right', top='top', bottom='bottom',
                    source=src_rect_view_lo,    fill_color="red",   fill_alpha=0.20, line_color=None)

# Start invisible (you can flip these to True when you’re ready to show them)
for r in (r_price_up, r_price_lo, r_dom_up, r_dom_lo, r_view_up, r_view_lo):
    r.visible = False


for p in (p1, p2, p3):
    p.add_tools(HoverTool(
        tooltips=[("month", "@month{%Y-%m}"), ("y", "@y{0.000}")],
        formatters={"@month": "datetime"}
    ))
    p.legend.location = "top_left"

# Forecast boundary marker (added dynamically per update)
boundary_rule = Span(location=np.nan, dimension="height", line_color="gray", line_dash="dotted", line_width=1)
for p in (p1, p2, p3):
    p.renderers.append(boundary_rule)

# =========================
# Controls (right column)
# =========================
zip_input = TextInput(title="ZIP code", value=DEFAULT_ZIP, placeholder="Type ZIP (e.g., 92620)")
horizon_slider = Slider(title="Forecast horizon (months)", start=3, end=36, step=1, value=DEFAULT_HORIZON)
k_slider = Slider(title="Fourier order K (seasonality)", start=1, end=8, step=1, value=DEFAULT_K)

# Defaults taken from vars.py
price_slider = RangeSlider(
    title="Price change thresholds (high risk  ⇆  strong)",
    start=-0.5, end=0.5, step=0.005,
    value=(vars.high_risk_price_change, vars.strong_market_price_change),
    format="0.000"
)

dom_slider = RangeSlider(
    title="Days on market thresholds (strong  ⇆  high risk)",
    start=0, end=200, step=1,
    value=(vars.strong_market_days_on_market, vars.high_risk_days_on_market),
    format="0"
)

view_slider = RangeSlider(
    title="View change thresholds (high risk  ⇆  strong)",
    start=-0.5, end=0.5, step=0.005,
    value=(vars.high_risk_view_count_change, vars.strong_market_view_count_change),
    format="0.000"
)

actions = CheckboxGroup(
    labels=[
        "Generate forecast",
        "Show forecast performance",
        "Generate likelihoods",
        "Show likelihood performance",
    ],
    active=[0],  # default: forecast visible
)

update_btn = Button(label="Update", button_type="primary")
status_div = Div(text="", width=320, height=140, sizing_mode="fixed")
notes = Div(text="", width=340, height=220,
            styles={"overflow-y": "auto", "white-space": "pre-wrap", "font-family": "monospace",
                    "border": "1px solid #ddd", "padding": "6px", "border-radius": "8px"})
# after: notes = TextAreaInput(title="Notes", value="", rows=6, cols=40)
bind_prints_to_notes(notes, max_lines=600, timestamp=True, tee=True)

# =========================
# Update Logic Helpers
# =========================
def make_outcome_trend_for_zip(zip_code):
    pr_low, pr_high = price_slider.value
    dom_low, dom_high = dom_slider.value
    vw_low, vw_high = view_slider.value
    outcome_thresholds = {
        'high_risk': {
            'price_change': pr_low,
            'days_on_market': dom_high,
            'view_count_change': vw_low
        },
        'strong_market': {
            'price_change': pr_high,
            'days_on_market': dom_low,
            'view_count_change': vw_high
        }
    }
    training_sets, outcome_sets, outcome_trend, inference_point = modeller.grab_training_records(final_df, zip_code, outcome_thresholds)
    # Normalize month to Timestamp @ MS (for plotting) and drop all-NaN rows
    out = forecaster._normalize_month_index(outcome_trend).reset_index().rename(columns={"index": "month"})
    
    # if 2 in actions.active:
    #     predictions, results = modeller.train_and_predict(training_sets, outcome_sets, inference_point)
    # else:
    #     predictions, results = None, None
    return out, training_sets, outcome_sets, inference_point

def _update_threshold_lines():
    """Update dotted threshold segments over the forecast window using current slider values."""
    global _forecast_x0, _forecast_x1
    if _forecast_x0 is None or _forecast_x1 is None:
        # No data yet
        for s in (src_thr_price_hi, src_thr_price_str, src_thr_dom_hi, src_thr_dom_str, src_thr_view_hi, src_thr_view_str):
            s.data = dict(x0=[], y0=[], x1=[], y1=[])
        return

    x0 = _forecast_x0
    x1 = _forecast_x1

    # Price change: (low=high risk, high=strong)
    pr_low, pr_high = price_slider.value
    src_thr_price_hi.data  = dict(x0=[x0], y0=[pr_low],  x1=[x1], y1=[pr_low])
    src_thr_price_str.data = dict(x0=[x0], y0=[pr_high], x1=[x1], y1=[pr_high])

    # DOM: (low=strong, high=high risk)
    dom_str, dom_hi = dom_slider.value
    src_thr_dom_hi.data    = dict(x0=[x0], y0=[dom_hi],  x1=[x1], y1=[dom_hi])
    src_thr_dom_str.data   = dict(x0=[x0], y0=[dom_str], x1=[x1], y1=[dom_str])

    # View change: (low=high risk, high=strong)
    vw_low, vw_high = view_slider.value
    src_thr_view_hi.data   = dict(x0=[x0], y0=[vw_low],  x1=[x1], y1=[vw_low])
    src_thr_view_str.data  = dict(x0=[x0], y0=[vw_high], x1=[x1], y1=[vw_high])

for sld in (price_slider, dom_slider, view_slider):
    sld.on_change("value", lambda attr, old, new: _update_threshold_lines())

def _month_edges(boundary_start: pd.Timestamp, months: int):
    """Return arrays of left/right edges for each forecast month."""
    m = pd.date_range(boundary_start, periods=months, freq='MS')
    left  = m
    right = m + pd.offsets.MonthBegin(1)
    return left, right

def _populate_series_rects(src_up: ColumnDataSource, src_lo: ColumnDataSource,
                           boundary_start: pd.Timestamp, months: int,
                           thr_low: float, thr_high: float, height: float,
                           semantics: str):
    """
    Fill two sources with 12 'upper' and 12 'lower' rectangles for one series.
    semantics: 'price' or 'view' => upper above STRONG, lower below HIGH-RISK
               'dom'             => upper above HIGH-RISK, lower below STRONG
    """
    left, right = _month_edges(boundary_start, months)
    n = len(left)

    if semantics in ("price", "view"):
        # Upper band: above STRONG (green); Lower band: below HIGH-RISK (red)
        up_bottom = np.repeat(thr_high,     n)
        up_top    = np.repeat(thr_high+height, n)
        lo_bottom = np.repeat(thr_low-height, n)
        lo_top    = np.repeat(thr_low,      n)
    else:  # 'dom' semantics inverted
        # Upper band: above HIGH-RISK (red); Lower band: below STRONG (green)
        up_bottom = np.repeat(thr_high,     n)
        up_top    = np.repeat(thr_high+height, n)
        lo_bottom = np.repeat(thr_low-height, n)
        lo_top    = np.repeat(thr_low,      n)

    src_up.data = dict(left=left, right=right, bottom=up_bottom, top=up_top)
    src_lo.data = dict(left=left, right=right, bottom=lo_bottom, top=lo_top)

def _kickoff_training(training_sets, outcome_sets, inference_point):
    """Run training in a background thread, stash results in module-level cache."""
    global _training_future, _training_busy

    if _training_busy:
        print("Training already running; ignoring request.")
        return
    _training_busy = True

    def _work():
        import traceback
        try:
            # Do your heavy work here:
            # from modeller import train_and_predict
            predictions, results = modeller.train_and_predict(training_sets, outcome_sets, inference_point)

            
            # Stash result into module-level cache
            def _done_ui():
                global _likelihood_cache, _training_busy
                _likelihood_cache = (predictions, results)
                status_div.text = f"<b>Training complete.</b>"
                # Optionally render perf right away:
                # _show_likelihood_perf_to_div()
                _training_busy = False

            curdoc().add_next_tick_callback(_done_ui)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Training error: {e}\n{tb}")
            def _fail_ui():
                global _training_busy
                status_div.text = f"<b style='color:#b00'>Training failed:</b> {e}"
                _training_busy = False
            curdoc().add_next_tick_callback(_fail_ui)

    _training_future = _executor.submit(_work)


# =========================
# Update logic
# =========================
def update_all(event=None):
    zip_code = zip_input.value.strip()
    try:
        zip_code = int(zip_code)
    except ValueError:
        status_div.text = f"<b style='color:#b00'>Error:</b> Invalid ZIP code '{zip_code}'"
        return
    horizon = int(horizon_slider.value)
    K = int(k_slider.value)

    try:
        hist, prd, res = make_outcome_trend_for_zip(zip_code)
        # Forecast using your Fourier forecaster (exposes K)
        fcst = forecaster.forecast_12_fourier(hist, horizon=horizon, K=K, period=12)

        # Keep a clean join
        hist = forecaster._normalize_month_index(hist).reset_index().rename(columns={"index": "month"})
        fcst = forecaster._normalize_month_index(fcst).reset_index().rename(columns={"index": "month"})

        # Update sources (historic)
        src_hist_price.data = {"month": hist["month"], "y": hist["price_change"]}
        src_hist_dom.data   = {"month": hist["month"], "y": hist["days_on_market"]}
        src_hist_view.data  = {"month": hist["month"], "y": hist["view_count_change"]}

        # Bridge forecast to last historic point
        fc_price = _bridge_series(hist, fcst, "price_change")
        fc_dom   = _bridge_series(hist, fcst, "days_on_market")
        fc_view  = _bridge_series(hist, fcst, "view_count_change")

        if 0 in actions.active:
            src_fore_price.data = {"month": fc_price["month"], "y": fc_price["price_change"]}
            src_fore_dom.data   = {"month": fc_dom["month"],   "y": fc_dom["days_on_market"]}
            src_fore_view.data  = {"month": fc_view["month"],  "y": fc_view["view_count_change"]}
        else:
            src_fore_price.data = dict(empty_source().data)
            src_fore_dom.data   = dict(empty_source().data)
            src_fore_view.data  = dict(empty_source().data)

        # Set forecast boundary at (last historic month + 1 month)
        global _forecast_x0, _forecast_x1
        if not hist.empty:
            boundary_x = pd.to_datetime(hist["month"].iloc[-1]) + pd.offsets.MonthBegin(1)
            boundary_rule.location = boundary_x.value / 1e6  # datetime -> ms epoch

            # Save window for threshold segments
            _forecast_x0 = pd.to_datetime(boundary_x)
            _forecast_x1 = pd.to_datetime(fcst["month"].iloc[-1])
        else:
            boundary_rule.location = np.nan
            _forecast_x0 = _forecast_x1 = None

        # Titles
        p1.title.text = f"Price Change – ZIP {zip_code}"
        p2.title.text = f"Days on Market – ZIP {zip_code}"
        p3.title.text = f"View Count Change – ZIP {zip_code}"

        # Set Rectangles
        print('actions active', actions.active)
        if 2 in actions.active:
            # Show rectangles
            for r in (r_price_up, r_price_lo, r_dom_up, r_dom_lo, r_view_up, r_view_lo):
                r.visible = True
            
            boundary_start = pd.to_datetime(hist["month"].iloc[-1]) + pd.offsets.MonthBegin(1)
            boundary_start = pd.to_datetime(boundary_start)

            # 24 rectangles per graph (12 up + 12 down), static thresholds, fixed heights
            pr_low, pr_high = price_slider.value
            dom_low, dom_high = dom_slider.value
            vw_low, vw_high = view_slider.value

            _populate_series_rects(
                src_rect_price_up, src_rect_price_lo,
                boundary_start, vars.max_months_predicted,
                thr_low=pr_low, thr_high=pr_high,
                height=RECT_H_PRICE, semantics="price"
            )
            _populate_series_rects(
                src_rect_dom_up, src_rect_dom_lo,
                boundary_start, vars.max_months_predicted,
                thr_low=dom_low, thr_high=dom_high,
                height=RECT_H_DOM, semantics="dom"
            )
            _populate_series_rects(
                src_rect_view_up, src_rect_view_lo,
                boundary_start, vars.max_months_predicted,
                thr_low=vw_low, thr_high=vw_high,
                height=RECT_H_VIEW, semantics="view"
            )
        else:
            for r in (r_price_up, r_price_lo, r_dom_up, r_dom_lo, r_view_up, r_view_lo):
                r.visible = False

            # clear if no data
            for s in (src_rect_price_up, src_rect_price_lo, src_rect_dom_up, src_rect_dom_lo, src_rect_view_up, src_rect_view_lo):
                s.data = dict(left=[], right=[], top=[], bottom=[])


        status_div.text = f"<b>Loaded ZIP:</b> {zip_code} | <b>Horizon:</b> {horizon} | <b>K:</b> {K}"
        
        # draw threshold lines now that x0/x1 are known
        _update_threshold_lines()

    except Exception as e:
        # Clear sources and show error
        for s in (src_hist_price, src_hist_dom, src_hist_view, src_fore_price, src_fore_dom, src_fore_view):
            s.data = {"month": [], "y": []}
        boundary_rule.location = np.nan
        status_div.text = f"<b style='color:#b00'>Error:</b> {str(e)}"

# Wire callbacks
update_btn.on_click(update_all)
zip_input.on_change("value", lambda attr, old, new: None)  # keep manual; update via button
zip_input.on_change("value_input", lambda attr, old, new: None)
zip_input.on_change("value", lambda attr, old, new: None)
# Also allow pressing Enter to submit in most Bokeh builds:
def _on_enter(attr, old, new):
    # When user presses Enter, value is committed; trigger update
    update_all()
zip_input.on_change("value", _on_enter)

# First render
update_all()

# =========================
# Layout
# =========================
left = column(p1, p2, p3, sizing_mode="stretch_both")

right = column(
    zip_input,
    Spacer(height=10),
    horizon_slider,
    k_slider,
    price_slider,
    dom_slider,
    view_slider,
    actions,
    update_btn,
    Spacer(height=10),
    status_div,
    Spacer(height=10),
    notes,        # or your Div-based log
    width=360,
    sizing_mode="fixed"
)


curdoc().add_root(row(left, right, sizing_mode="stretch_both"))
curdoc().title = "Zip Trend Forecaster"