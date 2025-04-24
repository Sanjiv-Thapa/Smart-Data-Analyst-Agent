import streamlit as st
import pandas as pd
from utils import basic_eda,ask_llm,answer_question,run_generated_code,suggest_next_action
st.set_page_config(page_title="smart data analyst agent")
st.title("Smart Data Analyst Agent")
uploaded_file = st.file_uploader("upload a csv file",type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    st.write(df.head())

    user_goal = st.text_input("Ask the agent your goal (e.g., 'analyze dataset', 'suggest model')")
    if st.button("Ask the Agent") and user_goal:

        if 'eda_summary' not in st.session_state:
            eda = basic_eda(df)
            st.session_state['eda_summary'] = eda
            st.subheader("EDA Summary")
            st.text(eda)
            st.stop()

        if 'preprocessing_suggestions' not in st.session_state:
            suggestions = ask_llm(st.session_state['eda_summary'])
            st.session_state['preprocessing_suggestions'] = suggestions
            st.subheader("AI Suggestions for Preprocessing / Modeling")
            st.text(suggestions)
            st.stop()

        st.subheader("AI's Answer to Your Goal")
        result = answer_question(df, user_goal)
        st.text(result)

    if st.button("Suggest Next Action"):
        suggestion = suggest_next_action()
        st.session_state['next_action'] = suggestion
        st.subheader("Suggested Next Action by AI")
        st.text(suggestion)

    action_instruction = st.text_input("Let AI generate code for this task (e.g., 'clean missing values')")
    if st.button("Generate and Run Code") and action_instruction:
        df, output = run_generated_code(df, action_instruction)
        st.session_state['df'] = df
        st.subheader("Generated Code Output")
        st.text(output)
        st.write(df.head())

    if st.checkbox("Show Agent Memory"):
        st.write(st.session_state)

else:
    st.info("👈 Please upload a CSV to get started.")