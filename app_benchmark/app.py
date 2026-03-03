"""
AI Model Benchmark
Compare GPT-4.1-mini, o3-mini, and Phi-4 on HR questions.
Evaluated by GPT-4o on Groundedness, Relevance, and Safety.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from core import (
    get_settings, COMPETITOR_MODELS,
    RAGService, MultiModelService, EvaluatorService,
)

st.set_page_config(
    page_title="AI Evaluation Studio",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean minimal styling
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* Primary button — Azure blue */
    .stButton > button {
        background-color: #0078d4 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background-color: #106ebe !important;
        color: white !important;
    }

    /* Sidebar background */
    div[data-testid="stSidebarContent"] {
        background-color: #fafafa !important;
    }

    /* Sliders — Azure blue */
    .stSlider [data-testid="stSlider"] > div > div > div {
        background-color: #0078d4 !important;
    }

    /* Checkboxes */
    .stCheckbox [data-testid="stCheckbox"] span {
        border-color: #0078d4 !important;
    }
</style>
""", unsafe_allow_html=True)

SAMPLE_QUESTIONS = [
    "What is the out-of-pocket maximum for the Northwind Standard plan?",
    "What is the deductible for the Northwind Standard plan?",
    "What are the monthly premiums for the Northwind Standard plan?",
    "Does the Northwind Standard plan cover mental health services?",
    "How do I find an in-network provider?",
    "Does the plan cover dental implants?",
    "What is the co-insurance rate?",
    "Is gym membership reimbursement covered?",
]


@st.cache_resource
def get_services():
    settings = get_settings()
    return (
        RAGService(settings),
        MultiModelService(settings),
        EvaluatorService(settings),
    )


# ── Header ───────────────────────────────────────────────────────────────────
st.title("AI Evaluation Studio")
st.caption("Compare AI models on your HR knowledge base. Evaluated by GPT-4o using Microsoft Azure AI Evaluation SDK.")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    st.subheader("Select Models")
    st.caption("Choose which models to benchmark. At least 1 required.")
    selected_models = []
    for model in COMPETITOR_MODELS:
        checked = st.checkbox(
            model.name,
            value=model.default_selected,
            key=f"sel_{model.name}",
            help=f"{model.provider} — ${model.input_cost_per_1m}/1M input tokens",
        )
        if checked:
            selected_models.append(model)

    st.divider()
    st.subheader("Sample Questions")
    sample = st.selectbox("Pick a sample", [""] + SAMPLE_QUESTIONS, label_visibility="collapsed")
    if sample:
        st.session_state.question = sample

    st.divider()
    st.subheader("Evaluator")
    st.info("GPT-4o judges all model answers in a single call for consistency.")
    st.subheader("Eval Thresholds")
    ground_thresh = st.slider("Groundedness", 1, 5, 4)
    relev_thresh = st.slider("Relevance", 1, 5, 4)

# ── Main ─────────────────────────────────────────────────────────────────────
if "question" not in st.session_state:
    st.session_state.question = ""

col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "Question",
        value=st.session_state.question,
        placeholder="Ask an HR benefits question...",
        label_visibility="collapsed",
    )
with col_btn:
    run_clicked = st.button("Run", use_container_width=True, type="primary")

if not selected_models:
    st.warning("Please select at least one model from the sidebar.")
    st.stop()

if run_clicked and question:
    try:
        rag_service, multi_service, eval_service = get_services()
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        st.stop()

    # Step 1: Safety check
    with st.spinner("Checking question safety..."):
        is_safe, safety_reason = eval_service.check_safety(question)

    if not is_safe:
        st.error(f"Question blocked: {safety_reason}")
        st.info("Please ask a question related to HR benefits and policies.")
        st.stop()

    # Step 2: Retrieve documents (shared across all models)
    with st.spinner("Retrieving relevant documents..."):
        retrieval = rag_service.retrieve(question)

    if not retrieval.documents:
        st.warning("No relevant documents found. Please try a different question.")
        st.stop()

    with st.expander(f"Retrieved {len(retrieval.documents)} source chunks", expanded=False):
        for i, doc in enumerate(retrieval.documents, 1):
            st.markdown(f"**Chunk {i}** — {doc.source or doc.title}")
            st.caption(doc.content[:250] + "...")

    st.divider()

    # Step 3: Run all models in parallel
    with st.spinner(f"Running {len(selected_models)} model(s) in parallel..."):
        system_prompt = rag_service.system_prompt(retrieval.context)
        responses = multi_service.run_parallel(selected_models, system_prompt, question)

    # Step 4: Evaluate all answers in one GPT-4o call
    with st.spinner("GPT-4o evaluating all responses..."):
        model_answer_pairs = [
            (r.model_name, r.answer if not r.error else "Error generating answer")
            for r in responses
        ]
        eval_results, summary = eval_service.evaluate(
            question=question,
            model_answers=model_answer_pairs,
            numbered_chunks=retrieval.numbered_chunks,
            context=retrieval.context,
        )

    # Build unified results table
    model_map = {m.name: m for m in COMPETITOR_MODELS}
    response_map = {r.model_name: r for r in responses}

    rows = []
    for er in eval_results:
        resp = response_map.get(er.model_name)
        model_cfg = model_map.get(er.model_name)
        cost = resp.cost(model_cfg) if resp and model_cfg else 0.0
        quality = er.overall_score
        value_score = round(quality / max(cost * 1000, 0.001), 2) if cost > 0 else 0.0
        rows.append({
            "model_name": er.model_name,
            "groundedness": er.groundedness,
            "relevance": er.relevance,
            "safety": "Pass" if er.safety else "Fail",
            "overall": er.overall_score,
            "verdict": er.verdict,
            "latency": resp.latency_seconds if resp else 0,
            "input_tokens": resp.input_tokens if resp else 0,
            "output_tokens": resp.output_tokens if resp else 0,
            "total_tokens": (resp.input_tokens + resp.output_tokens) if resp else 0,
            "cost": cost,
            "value_score": value_score,
            "coherence": er.coherence,
            "fluency": er.fluency,
            "eval_result": er,
            "response": resp,
        })

    # ── Summary Banner ────────────────────────────────────────────────────────
    st.subheader("Results")
    if summary:
        st.info(f"**GPT-4o Judge Summary:** {summary}")

    # ── Comparison Table ──────────────────────────────────────────────────────
    st.markdown("#### Comparison Table")
    table_data = []
    for row in rows:
        verdict_label = row["verdict"]
        table_data.append({
            "Model": row["model_name"],
            "Groundedness": f"{row['groundedness']:.1f}/5",
            "Relevance": f"{row['relevance']:.1f}/5",
            "Safety": row["safety"],
            "Groundedness": f"{row['groundedness']:.1f}/5",
            "Relevance": f"{row['relevance']:.1f}/5",
            "Coherence": f"{row['coherence']:.1f}/5",
            "Fluency": f"{row['fluency']:.1f}/5",
            "Overall Score": f"{row['overall']:.2f}/5",
            "Verdict": verdict_label,
            "Latency (s)": f"{row['latency']:.2f}s",
            "Tokens": row["total_tokens"],
            "Est. Cost (USD)": f"${row['cost']:.6f}",
            "Value Score": row["value_score"],
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    model_names = [r["model_name"] for r in rows]

    # Azure / Microsoft official color palette
    AZURE_COLORS = {
        "groundedness": "#0078d4",  # Azure blue
        "relevance":    "#50e6ff",  # Azure cyan
        "coherence":    "#773adc",  # Azure purple
        "fluency":      "#00b294",  # Azure teal
        "overall":      "#005a9e",  # Azure dark blue
    }
    # Per-model colors for latency/cost/value charts
    MODEL_COLORS = ["#0078d4", "#005a9e", "#50e6ff", "#773adc", "#00b294"]
    colors = MODEL_COLORS[:len(model_names)]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Scores by Model")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Groundedness", x=model_names, y=[r["groundedness"] for r in rows], marker_color=AZURE_COLORS["groundedness"]))
        fig.add_trace(go.Bar(name="Relevance",    x=model_names, y=[r["relevance"] for r in rows],    marker_color=AZURE_COLORS["relevance"]))
        fig.add_trace(go.Bar(name="Coherence",    x=model_names, y=[r.get("coherence", 0) for r in rows], marker_color=AZURE_COLORS["coherence"]))
        fig.add_trace(go.Bar(name="Fluency",      x=model_names, y=[r.get("fluency", 0) for r in rows],   marker_color=AZURE_COLORS["fluency"]))
        fig.add_trace(go.Bar(name="Overall",      x=model_names, y=[r["overall"] for r in rows],      marker_color=AZURE_COLORS["overall"]))
        fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 5.5], title="Score", gridcolor="#f1f5f9"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
            margin=dict(t=40, b=20),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Latency by Model (seconds)")
        fig2 = go.Figure(go.Bar(
            x=model_names,
            y=[r["latency"] for r in rows],
            marker_color=colors,
            text=[f"{r['latency']:.2f}s" for r in rows],
            textposition="outside",
        ))
        fig2.update_layout(
            yaxis=dict(title="Seconds", gridcolor="#f1f5f9"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(t=20, b=20),
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Estimated Cost per Query (USD)")
        fig3 = go.Figure(go.Bar(
            x=model_names,
            y=[r["cost"] for r in rows],
            marker_color=colors,
            text=[f"${r['cost']:.6f}" for r in rows],
            textposition="outside",
        ))
        fig3.update_layout(
            yaxis=dict(title="USD", gridcolor="#f1f5f9"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(t=20, b=20),
            height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Value Score (Quality / Cost)")
        st.caption("Higher is better — measures quality delivered per dollar spent.")
        fig4 = go.Figure(go.Bar(
            x=model_names,
            y=[r["value_score"] for r in rows],
            marker_color=colors,
            text=[f"{r['value_score']:.1f}" for r in rows],
            textposition="outside",
        ))
        fig4.update_layout(
            yaxis=dict(title="Value Score", gridcolor="#f1f5f9"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(t=20, b=20),
            height=380,
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Per Model Detail ──────────────────────────────────────────────────────
    st.markdown("#### Model Responses and Justifications")
    for row in rows:
        er = row["eval_result"]
        resp = row["response"]
        verdict_icon = "PASS" if row["verdict"] == "PASS" else "BLOCK"
        with st.expander(
            f"[{verdict_icon}]  {row['model_name']}  |  Overall: {row['overall']:.2f}/5  |  {row['latency']:.2f}s  |  {row['total_tokens']} tokens  |  ${row['cost']:.6f}",
            expanded=(er.overall_score == max(r["overall"] for r in rows)),
        ):
            # Scores
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Groundedness", f"{er.groundedness}/5",
                      delta=float(er.groundedness - ground_thresh))
            c2.metric("Relevance", f"{er.relevance}/5",
                      delta=float(er.relevance - relev_thresh))
            c3.metric("Safety", "Safe" if er.safety else "Risk")
            c4.metric("Overall Score", f"{er.overall_score:.2f}/5")

            st.divider()

            # Answer
            st.markdown("**Answer:**")
            if resp and resp.error:
                st.error(f"Error: {resp.error}")
            else:
                st.markdown(resp.answer if resp else "_No response_")

            st.divider()

            # Justification
            st.markdown("**Evaluation Justification:**")

            st.markdown("**Groundedness**")
            st.write(er.groundedness_reason)
            st.markdown("**Relevance**")
            st.write(er.relevance_reason)

            st.markdown("**Coherence**")
            st.write(er.coherence_reason)

            st.markdown("**Fluency**")
            st.write(er.fluency_reason)

    # ── Winner ────────────────────────────────────────────────────────────────
    st.divider()
    best = rows[0]
    if best["verdict"] == "PASS":
        st.success(f"Best model for this question: **{best['model_name']}** — Overall {best['overall']:.2f}/5, {best['latency']:.2f}s, ${best['cost']:.6f}")
    else:
        st.warning("No model achieved a PASS verdict for this question. Consider reviewing source documents or adjusting eval thresholds.")

elif not question:
    st.caption("Enter a question above and click Run to start the benchmark.")