"""Local web UI: enter an orbit (or pick a known asteroid / upload a CSV) and get
its hazard probability, collision proxy and charts.

    uv run streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from neo_hazard.features import SIGMA_COLS
from neo_hazard.moid import EARTH, orbit_points
from neo_hazard.predict import REQUIRED, score
from neo_hazard.rank import SCORED_PATH

EARTH_COLOR = "#2a78d6"   # categorical slot 1
OBJECT_COLOR = "#eb6834"  # categorical slot 2
MUTED = "rgba(128,128,128,0.35)"
LD_PER_AU = 389.17

st.set_page_config(page_title="NEO Impact Hazard", page_icon="☄️", layout="wide")


@st.cache_data
def load_neos() -> pd.DataFrame:
    df = pd.read_parquet(SCORED_PATH)
    return df[df["neo"] == 1].sort_values("hazard_score", ascending=False).reset_index(drop=True)


if not SCORED_PATH.exists():
    st.error(f"Scored data not found at `{SCORED_PATH}`.")
    st.markdown(
        "This file is generated, not shipped in the repo. Create it on this machine with:\n\n"
        "```bash\n"
        "uv run python -m neo_hazard.train   # needs dataset.csv in the project root\n"
        "uv run python -m neo_hazard.rank    # writes outputs/scored.parquet\n"
        "```\n\n"
        "or copy `models/pha_hgb.joblib` and `outputs/scored.parquet` from a machine "
        "that already has them. See the README section *Running on another Mac*."
    )
    st.stop()

neos = load_neos()
DEFAULTS = neos[SIGMA_COLS + ["rms"]].median().to_dict()


# ---------- charts ----------
def orbit_figure(row: pd.Series) -> go.Figure:
    f = np.linspace(0, 2 * np.pi, 721)
    earth = orbit_points(**EARTH, f=f)
    obj = orbit_points(row.a, row.e, row.i, row.om, row.w, f)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=earth[:, 0], y=earth[:, 1], name="Earth orbit", mode="lines",
                             line=dict(color=EARTH_COLOR, width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=obj[:, 0], y=obj[:, 1], name="Object orbit", mode="lines",
        line=dict(color=OBJECT_COLOR, width=2),
        customdata=np.c_[obj[:, 2], np.linalg.norm(obj, axis=1)],
        hovertemplate="x %{x:.3f} AU<br>y %{y:.3f} AU<br>z %{customdata[0]:.3f} AU"
                      "<br>r %{customdata[1]:.3f} AU<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0], y=[0], name="Sun", mode="markers",
                             marker=dict(size=12, color="#eda100"), hoverinfo="name"))
    lim = float(min(max(np.abs(obj[:, :2]).max(), 1.2), 6)) * 1.08
    fig.update_layout(
        title="Orbits seen from above the ecliptic", height=460, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(range=[-lim, lim], title="x (AU)", zeroline=False),
        yaxis=dict(range=[-lim, lim], title="y (AU)", scaleanchor="x", zeroline=False),
        legend=dict(orientation="h", y=-0.15))
    return fig


def population_figure(points: pd.DataFrame) -> go.Figure:
    """Where the input sits among known NEOs in MOID–H space (PHA zone shaded)."""
    fig = go.Figure()
    fig.add_shape(type="rect", x0=1e-6, x1=0.05, y0=10, y1=22, layer="below",
                  fillcolor="rgba(128,128,128,0.15)", line=dict(width=0))
    fig.add_annotation(x=np.log10(0.05), y=10.6, text="PHA zone (MOID ≤ 0.05 AU, H ≤ 22)",
                       showarrow=False, xanchor="right", font=dict(size=11))
    fig.add_trace(go.Scattergl(
        x=neos["moid"].clip(lower=1e-6), y=neos["H"], mode="markers", name="Known NEOs",
        marker=dict(size=3, color=MUTED), text=neos["full_name"],
        hovertemplate="%{text}<br>MOID %{x:.4g} AU<br>H %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=points["moid"].clip(lower=1e-6), y=points["H"], mode="markers", name="Your input",
        marker=dict(size=12, color=OBJECT_COLOR, line=dict(width=2, color="white")),
        text=points.get("name", pd.Series(["input"] * len(points))),
        hovertemplate="%{text}<br>MOID %{x:.4g} AU<br>H %{y}<extra></extra>"))
    fig.update_layout(
        title="Input vs. known near-Earth objects", height=460, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(type="log", title="MOID (AU, log scale)"),
        yaxis=dict(title="Absolute magnitude H (smaller = bigger object)", autorange="reversed",
                   range=[33, 10]),
        legend=dict(orientation="h", y=-0.15))
    return fig


def show_result(r: pd.Series) -> None:
    capture_ratio = r.moid / r.capture_radius_au
    closer_than = (neos["moid"] > r.moid).mean() * 100
    c = st.columns(3)
    c[0].metric("Hazard probability (ML)", f"{r.p_pha_ml:.1%}",
                help="Model's probability that this is a Potentially Hazardous Asteroid")
    c[1].metric("MOID", f"{r.moid:.2g} AU", f"{r.moid * LD_PER_AU:.3g} lunar distances",
                delta_color="off", help=f"Closest distance between the two orbits. Source: {r.get('moid_source', 'JPL')}")
    c[2].metric("Closer orbit than", f"{closer_than:.1f}% of NEOs")
    c = st.columns(3)
    c[0].metric("Miss distance / capture radius", f"{capture_ratio:,.3g}×",
                help="≤ 1× means the orbit geometrically intersects Earth's gravitationally focused cross-section")
    c[1].metric("P(orbit intersects Earth)", f"{r.p_moid_within_capture:.2g}",
                help="Probability the true MOID is within Earth's capture radius, given the orbit uncertainty")
    c[2].metric("Hazard score", f"{r.hazard_score:.2g}",
                help="P(hazardous) × P(intersects) × orbits per year; used for ranking")

    if r.get("moid_source") == "computed":
        st.info("MOID was computed locally (accuracy about ±0.0003 AU, larger than Earth's 0.00005 AU "
                "capture radius), and that error is included in the intersection probability. "
                "For a precise answer, supply the JPL MOID in a CSV `moid` column.")
    if r.p_moid_within_capture > 1e-3:
        st.error("An intersection with Earth's capture zone cannot be ruled out at this orbit precision. Orbit propagation (as Sentry does) would be needed to assess a real impact.")
    elif r.p_pha_ml >= 0.5:
        st.warning("Classified as potentially hazardous: large enough and on an orbit that comes close to Earth's.")
    else:
        st.success("Not classified as hazardous by the model.")

    g = st.columns(2)
    g[0].plotly_chart(orbit_figure(r), width="stretch")
    g[1].plotly_chart(population_figure(r.to_frame().T.astype({"moid": float, "H": float})),
                      width="stretch")


# ---------- page ----------
st.title("☄️ Near-Earth Object Impact Hazard")
st.caption("Gradient-boosted model trained on the JPL Small-Body Database (no MOID used as input), "
           "plus a geometric collision proxy. This is a **relative hazard ranking**, not an official "
           "impact probability; for that, see NASA CNEOS Sentry.")

mode = st.sidebar.radio("Input", ["Known asteroid", "Enter orbit", "Upload CSV"])

if mode == "Known asteroid":
    name = st.sidebar.selectbox("Search NEOs (sorted by hazard score)", neos["full_name"])
    row = neos[neos["full_name"] == name].iloc[0].copy()
    row["moid_source"] = "JPL catalogue"
    st.subheader(name)
    show_result(row)
    st.dataframe(neos.head(25)[["full_name", "H", "diameter", "moid", "moid_ld",
                                "p_moid_within_capture", "p_pha_ml", "hazard_score"]],
                 hide_index=True, width="stretch")
    st.caption("Top 25 NEOs by hazard score")

elif mode == "Enter orbit":
    sb = st.sidebar
    sb.caption("Defaults are Apophis (99942). Angles in degrees.")
    vals = dict(
        a=sb.number_input("a: semi-major axis (AU)", 0.1, 100.0, 0.922438, format="%.6f"),
        e=sb.number_input("e: eccentricity", 0.0, 0.999, 0.191195, format="%.6f"),
        i=sb.number_input("i: inclination", 0.0, 180.0, 3.33137, format="%.5f"),
        om=sb.number_input("Ω: longitude of ascending node", 0.0, 360.0, 204.44603, format="%.5f"),
        w=sb.number_input("ω: argument of perihelion", 0.0, 360.0, 126.40188, format="%.5f"),
        H=sb.number_input("H: absolute magnitude", -2.0, 35.0, 19.7, step=0.1),
    )
    with sb.expander("Orbit uncertainty (optional)"):
        vals["sigma_q"] = st.number_input("σ q (AU)", 0.0, 1.0, float(DEFAULTS["sigma_q"]), format="%.2e")
        vals["sigma_e"] = st.number_input("σ e", 0.0, 1.0, float(DEFAULTS["sigma_e"]), format="%.2e")
    r = score(pd.DataFrame([vals]), DEFAULTS).iloc[0]
    st.subheader("Your orbit")
    show_result(r)

else:
    st.sidebar.markdown(f"CSV columns required: `{', '.join(REQUIRED)}`. "
                        "Optional: `name`, `moid`, `sigma_q`, `sigma_e`, other `sigma_*`, `rms`.")
    template = pd.DataFrame([dict(name="Apophis-like", a=0.9224, e=0.1912, i=3.331, om=204.446,
                                  w=126.402, H=19.7)])
    st.sidebar.download_button("Download template CSV", template.to_csv(index=False), "orbits_template.csv")
    up = st.sidebar.file_uploader("Upload orbits CSV", type="csv")
    if up is None:
        st.info("Upload a CSV of orbits in the sidebar to score them in bulk.")
    else:
        inp = pd.read_csv(up)
        missing = [c for c in REQUIRED if c not in inp]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            with st.spinner(f"Scoring {len(inp)} orbits…"):
                out = score(inp, DEFAULTS).sort_values("hazard_score", ascending=False)
            cols = [c for c in ["name"] if c in out] + REQUIRED + [
                "moid", "moid_source", "moid_ld", "p_pha_ml", "p_moid_within_capture",
                "impact_proxy_per_yr", "hazard_score"]
            st.dataframe(out[cols], hide_index=True, width="stretch")
            st.download_button("Download results", out[cols].to_csv(index=False), "hazard_results.csv")
            st.plotly_chart(population_figure(out), width="stretch")
            pick = st.selectbox("Inspect one orbit", out.index,
                                format_func=lambda k: str(out.loc[k].get("name", k)))
            show_result(out.loc[pick])
