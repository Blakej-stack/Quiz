import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional
from scipy.stats import norm

st.set_page_config(page_title="Histogram Quick Quiz", page_icon="curiousStat_icon.png", layout="centered")

# ---------- Data model ----------
@dataclass
class MCQ:
    prompt: str
    options: List[str]
    correct_idx: int
    explain: str

QUESTIONS = [
    MCQ(
        prompt="What type of data is best represented by a histogram?",
        options=["Categorical data (e.g., eye color)", "Continuous numerical data (e.g., height)","Names of students","Phone numbers"],
        correct_idx=1,
        explain="Histograms are for continuous (or large-range discrete) data; categories use bar charts."
    ),
    MCQ(
        prompt="What’s the best description of this distribution?",
        options=["Symmetric", "Right-skewed", "Bimodal", "Uniform"],
        correct_idx=0,
        explain="Imagine a vertical line down the center: both sides roughly match (symmetric)."
    ),
    MCQ(
        prompt="In a histogram with unequal bin widths, what corresponds to frequency?",
        options=["Bar height","Bar area (height × bin width)","Number of bins","Axis labels"],
        correct_idx=1,
        explain="When bin widths differ, the area is proportional to frequency; height alone can mislead."
    ),
    MCQ(
        prompt="Most bars are on the left with a long tail to the right. The distribution is:",
        options=["Symmetric","Left-skewed (negative)","Right-skewed (positive)","Uniform"],
        correct_idx=2,
        explain="Pile on the left + tail to the right → right/positive skew."
    ),
    MCQ(
        prompt="If you double the bin width (fewer, wider bins), the histogram will usually:",
        options=["Show more fine-grained detail","Hide small clusters/gaps","Become perfectly normal","Not change at all"],
        correct_idx=1,
        explain="Wider bins smooth the shape and can hide structure."
    ),
    MCQ(
        prompt="Two classes’ histograms: A is tall/narrow around the mean; B is flat/spread out. What’s true?",
        options=["A has lower variability than B","B has smaller standard deviation","They must have equal variability","B’s mean is higher than A’s"],
        correct_idx=0,
        explain="Tight clustering → lower variability; flat/spread → higher variability."
    ),
    MCQ(
        prompt="Why use a histogram instead of the raw data list?",
        options=["To display exact values","To summarize the distribution’s shape and spread","To compute the mean exactly","Because it’s required by default"],
        correct_idx=1,
        explain="Histograms reveal shape, center, spread, gaps, and outliers at a glance."
    ),
    MCQ(
        prompt="You sample 1,000,000 values from a Normal(μ=50, σ=10) and plot many bins. Which is accurate?",
        options=["It will be a perfect bell curve","It will closely approximate the bell curve","It will be skewed","Only bin width matters, not data"],
        correct_idx=1,
        explain="Large samples approximate the underlying distribution well, though not perfectly."
    ),
    MCQ(
        prompt="Gaps between bars in a histogram usually indicate:",
        options=["Missing values in the CSV","Intervals with few or no observations","A plotting bug","Wrong x-axis scale"],
        correct_idx=1,
        explain="Gaps typically mean no data fell in those intervals."
    ),
]

# ---------- Helpers ----------
def init_state():
    if "q_order" not in st.session_state:
        st.session_state.q_order = list(range(len(QUESTIONS)))
        #random.shuffle(st.session_state.q_order)  # light randomization
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "answers" not in st.session_state:
        st.session_state.answers = [None] * len(QUESTIONS)
    if "revealed" not in st.session_state:
        st.session_state.revealed = [False] * len(QUESTIONS)
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "submit_enabled" not in st.session_state:
        st.session_state.submit_enabled = True
    if "next_enabled" not in st.session_state:
        st.session_state.next_enabled = False
    if "finished" not in st.session_state:
        st.session_state.finished = False

def click_start():
    st.session_state.started = True

def click_submit():
    record_answer(choice[0])
    st.session_state.submit_enabled = False
    st.session_state.next_enabled = True if not quiz_finished() else False

def click_next():
    st.session_state.next_enabled = False
    st.session_state.submit_enabled = True
    st.session_state.idx += 1

def click_finish():
    st.session_state.next_enabled = False
    st.session_state.submit_enabled = True
    st.session_state.finished = True

def reset_quiz():
    for k in ["q_order","idx","answers","revealed","score", "submit_enabled", "next_enabled", "finished"]:
        if k in st.session_state: del st.session_state[k]
    init_state()

def current_question() -> MCQ:
    return QUESTIONS[st.session_state.q_order[st.session_state.idx]]

def record_answer(choice_idx: int):
    qpos = st.session_state.idx
    if st.session_state.revealed[qpos]:
        return  # already graded
    st.session_state.answers[qpos] = choice_idx
    correct = (choice_idx == current_question().correct_idx)
    if correct:
        st.session_state.score += 1
    st.session_state.revealed[qpos] = True

def next_question():
    if st.session_state.idx < len(QUESTIONS) - 1:
        st.session_state.idx += 1

def quiz_last_quest() -> bool:
    return st.session_state.idx == len(QUESTIONS) - 1 and st.session_state.revealed[-2]

def quiz_finished() -> bool:
    return st.session_state.finished

# def generate_question(type=None):
#     data = np.random.normal(loc=par_mean, scale=par_std, size=1000)
#     return data

def generate_data(par_mean=None, par_std=None):
    data = np.random.normal(loc=par_mean, scale=par_std, size=1000)
    return data

# ---------- Charts ----------

# Q0
# Q1
mean = np.random.uniform(-2, 2)
std = np.random.uniform(1, 2)
data1 = np.random.normal(loc=mean, scale=std, size=200)


# n_trials = 10
# p_succ = 0.5
# ss = 1000
# rng = np.random.default_rng(seed=42)
# data1 = rng.binomial(n_trials, p_succ, ss)

# fig1 = plt.hist(data1, bins=30, edgecolor="black")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.title("Histogram of Normal(0,1) samples")

# Q2
# mean = st.slider("Mean", -3.0, 3.0, 0.0, step=0.05)
# std = st.slider("Standard Deviation", 0.1, 3.0, 1.0, step=0.05)
# x = np.linspace(-3, 3, 1000)
# y = (1 / np.sqrt(2 * np.pi * (std**2))) * np.exp(-0.5 * ((x - mean) ** 2 / (std**2)))

# fig2, ax2 = plt.subplots()
# ax2.hist(data2, bins=30, density=True, alpha=0.6, color="green", label="Histogram")
# ax2.plot(x, y, 'r-', lw=2, label="Fitted Normal Distribution")
# ax2.set_ylim(0, 0.5)
# ax2.set_xlabel("Value")
# ax2.set_ylabel("Density")
# ax2.legend()
# ax2.set_title("Fitting a Normal Distribution")
# st.pyplot(fig)



# ---------- UI ----------
st.title("Stats Quiz: Histograms")
# st.caption("Test your knowledge about Histograms")

if "started" not in st.session_state:
    st.session_state.started = False

# Landing state (before first reveal)
if not st.session_state.started:
    st.write("""
         Histograms are a powerful way to visualize data. This quick quiz will test your understanding of histograms, their interpretation, and related concepts.
         Click **Start** when you are ready!
         """)
    cols = st.columns([1,1,1])
    with cols[1]:
        st.button('Start', on_click=click_start, width='stretch', type='primary')
    with st.container(border=True, ):
        st.info("If you need a refresher or if you would like to learn more about histograms first, check out this video:")
        st.video("https://youtu.be/hdUDyozbJpo")
    init_state()
else:
    # Progress
    progress = (st.session_state.idx+1) / len(QUESTIONS)
    st.progress(progress)
    st.write(f"**Question {st.session_state.idx + 1} of {len(QUESTIONS)}**")

    # Current question block
    q = current_question()
    with st.container(border=True, ):
        st.markdown(f"**{q.prompt}**")

        if st.session_state.idx==1:
            if "data" not in st.session_state:
                st.session_state.data = generate_data(par_mean = 2, par_std = 1.5)

            fig, ax = plt.subplots()
            ax.hist(st.session_state.data, bins=50, edgecolor="black", density=True, alpha=0.6)
            ax.set_xlim(-6, 6)
            ax.set_ylim(0, 0.5)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            st.pyplot(fig)

            choice = st.radio(
                "Choose one:",
                options=list(enumerate(q.options)),
                format_func=lambda x: x[1],
                index=None if st.session_state.answers[st.session_state.idx] is None else st.session_state.answers[st.session_state.idx],
                key=f"choice_{st.session_state.idx}",
            )

        # if st.session_state.idx==2:
        #     if "data" not in st.session_state:
        #         st.session_state.data = generate_data(par_mean = 2, par_std = 1.5)
        #     slider_mean = st.slider("Mean", -3.0, 3.0, 0.0, step=0.05)
        #     slider_std = st.slider("Standard Deviation", 0.5, 3.0, 1.75, step=0.05)
              
        #     x1 = np.linspace(-6, 6, 1000)
        #     y1 = norm.pdf(x1, slider_mean, slider_std)

        #     fig, ax = plt.subplots()
        #     ax.hist(st.session_state.data, bins=18, edgecolor="black", density=True, alpha=0.6)
        #     ax.plot(x1, y1, 'r-', lw=2, label="Fitted Normal Distribution")
        #     ax.set_xlim(-6, 6)
        #     ax.set_ylim(0, 0.5)
        #     ax.set_xlabel("Value")
        #     ax.set_ylabel("Density")
        #     st.pyplot(fig)

        #     choice = [slider_mean]

        if st.session_state.idx==0 or st.session_state.idx>=2:
            choice = st.radio(
                "Choose one:",
                options=list(enumerate(q.options)),
                format_func=lambda x: x[1],
                index=None if st.session_state.answers[st.session_state.idx] is None else st.session_state.answers[st.session_state.idx],
                key=f"choice_{st.session_state.idx}",
            )

        cols = st.columns([1,1])
        with cols[0]:
            submit = st.button(
                label="Check answer"
                , disabled=((choice is None) or (not st.session_state.submit_enabled))
                , on_click=click_submit
                )
        with cols[1]:
            if not quiz_last_quest():
                nxt = st.button(
                    label="Next ▶"
                    , disabled= not st.session_state.next_enabled
                    , on_click=click_next
                    )
            else:
                nxt = st.button(
                    label="Finish ✅"
                    , disabled= not st.session_state.next_enabled
                    , on_click=click_finish
                    # , type='primary'
                    )

        # Feedback
        if st.session_state.revealed[st.session_state.idx]:                                 # if already revealed
            your_idx = st.session_state.answers[st.session_state.idx]
            correct_idx = q.correct_idx
            if your_idx == correct_idx:
                st.success(f"✅ Correct: **{q.options[correct_idx]}**")
            else:
                st.error(f"❌ Not quite. Correct answer: **{q.options[correct_idx]}**")
            st.caption(q.explain)

        if nxt and st.session_state.revealed[st.session_state.idx]:
            next_question()

# Results screen
if quiz_finished():
    st.divider()
    st.subheader("Your results")
    st.metric("Score", f"{st.session_state.score} / {len(QUESTIONS)}")
    st.caption("Quick review of your responses:")

    # Build review table
    rows = []
    for i, qidx in enumerate(st.session_state.q_order):
        q = QUESTIONS[qidx]
        a = st.session_state.answers[i]
        rows.append({
            "Q": i+1,
            "Your answer": "-" if a is None else q.options[a],
            "Correct answer": q.options[q.correct_idx]
        })
    st.dataframe(rows, hide_index=True, width='stretch')

    st.success("Nice work! Want to keep your momentum?")
    c1, c2 = st.columns(2)
    with c1:
        st.button("🔁 Try again", on_click=reset_quiz)
    with c2:
        st.link_button("📚 Refresh your memory on histograms", "https://youtu.be/hdUDyozbJpo")