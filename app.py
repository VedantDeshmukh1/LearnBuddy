import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import plotly.graph_objects as go
import random
import pandas as pd
from datetime import datetime, timedelta
from educhain.qna_engine import generate_mcq
from educhain.models import MCQList
import os
import sympy
from latex2sympy2 import latex2sympy

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key
# Set up OpenAI API (replace with your actual API key)
llm = ChatOpenAI(model = 'gpt-4o', api_key = openai_api_key )

st.set_page_config(page_title="StudyBuddy AI", page_icon="📚", layout="wide")

st.title("📚 StudyBuddy AI: Your Advanced Learning Companion")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a feature", ["Subject Tutor", "Homework Helper", "Quiz Generator", "Study Planner", "Concept Explainer", "Progress Tracker", "Flashcard Creator"])

# Function to get AI response using LangChain
def get_ai_response(prompt_template, **kwargs):
    chat_prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    if kwargs:
        response = chain.run(**kwargs)
    else:
        response = chain.run({"prompt": prompt_template})
    return response

if page == "Subject Tutor":
    st.header("🧑‍🏫 Subject-Specific Tutoring")
    subject = st.selectbox("Select a subject", ["Math", "Science", "History", "Literature", "Computer Science"])
    difficulty = st.select_slider("Select difficulty level", options=["Beginner", "Intermediate", "Advanced"])
    question = st.text_area("Ask your question:")
    if st.button("Get Help"):
        prompt = """
        As a {difficulty} level tutor for {subject}, please help with this question: {question}
        Provide a detailed explanation, including any relevant formulas, theories, or historical context.
        If applicable, suggest additional resources for further learning.
        """
        answer = get_ai_response(prompt, difficulty=difficulty, subject=subject, question=question)
        st.write(answer)

        # Visualization: Related topics word cloud
        st.subheader("Related Topics")
        words = answer.split()
        word_freq = {word: random.randint(10, 100) for word in set(words) if len(word) > 4}
        fig = go.Figure(data=[go.Scatter(x=list(word_freq.keys()), y=list(word_freq.values()),
                                         mode='text', text=list(word_freq.keys()),
                                         textfont={'size': [v/3 for v in word_freq.values()]})])
        fig.update_layout(title="Related Topics Word Cloud")
        st.plotly_chart(fig)

elif page == "Homework Helper":
    st.header("📝 Homework Helper")
    subject = st.selectbox("Select a subject", ["Math", "Science", "History", "Literature", "Computer Science"])
    problem = st.text_area("Describe your homework problem:")
    show_steps = st.checkbox("Show detailed steps")
    if st.button("Get Assistance"):
        # Pre-process the prompt based on the show_steps checkbox
        brevity = "Briefly " if not show_steps else ""
        detail = " with detailed steps" if show_steps else ""

        prompt = f"""
        Help solve this {{subject}} homework problem: {{problem}}
        {brevity}explain the solution{detail}.
        Include any relevant formulas or theories.
        """
        solution = get_ai_response(prompt, subject=subject, problem=problem)
        st.write(solution)

        # Visualization: Solution breakdown
        steps = solution.split('\n')
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Step', 'Description'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[list(range(1, len(steps)+1)), steps],
                       fill_color='black',
                       align='left'))
        ])
        fig.update_layout(title="Solution Breakdown")
        st.plotly_chart(fig)

if page == "Quiz Generator":
    st.header("🧠 Quiz Generator")
    subject = st.selectbox("Select a subject", ["Math", "Science", "History", "Literature", "Computer Science", "Custom"])
    
    if subject == "Custom":
        subject = st.text_input("Enter your custom topic:")
    
    num_questions = st.slider("Number of questions", 1, 10, 5)
    difficulty = st.select_slider("Select difficulty level", options=["Easy", "Medium", "Hard"])

    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
        st.session_state.quiz_data = None

    if st.button("Generate Quiz"):
        quiz = generate_mcq(
            topic=subject,
            num=num_questions,
            llm=llm,
            response_model=MCQList,
            prompt_template=f"Create a {num_questions}-question {difficulty} quiz on {subject}.\nFor each question, provide:\n1. The question\n2. Four multiple-choice options (A, B, C, D)\n3. The correct answer (A, B, C, or D)\n4. A brief explanation of the correct answer\nIf the question involves mathematical equations, use LaTeX format.",
        )

        st.session_state.quiz_data = quiz.questions
        st.session_state.quiz_generated = True
        st.session_state.user_answers = {}
        st.session_state.checked_answers = set()

    if st.session_state.quiz_generated:
        for i, q in enumerate(st.session_state.quiz_data, 1):
            question = q.question
            options = q.options
            correct_answer = q.correct_answer
            explanation = q.explanation

            st.subheader(f"Question {i}")
            # Use st.latex for rendering LaTeX equations
            st.latex(question)

            # Create a unique key for each radio button group
            user_answer = st.radio("Select your answer:", options, key=f"q{i}")
            st.session_state.user_answers[i] = user_answer

            if st.button("Check Answer", key=f"check_{i}"):
                st.session_state.checked_answers.add(i)

            if i in st.session_state.checked_answers:
                if st.session_state.user_answers[i] == correct_answer:
                    st.success("Correct!")
                else:
                    st.error(f"Incorrect. The correct answer is {correct_answer}")
                st.info(f"Explanation: {explanation}")

        if st.button("Submit Quiz"):
            score = sum(1 for i, q in enumerate(st.session_state.quiz_data, 1) 
                        if st.session_state.user_answers.get(i) == q.correct_answer)
            st.success(f"Quiz submitted! Your score: {score}/{len(st.session_state.quiz_data)}")

def parse_math_equation(equation):
    try:
        return latex2sympy(equation)
    except:
        return equation
        
elif page == "Study Planner":
    st.header("📅 Study Schedule Planner")
    subjects = st.multiselect("Select your subjects", ["Math", "Science", "History", "Literature", "Computer Science"])
    study_hours = st.number_input("How many hours can you study per day?", min_value=1, max_value=12, value=4)
    start_date = st.date_input("Start date of your study plan")
    duration = st.slider("Duration of study plan (weeks)", 1, 8, 2)
    if st.button("Create Study Plan"):
        prompt = f"""
        Create a {duration}-week study schedule starting from {start_date} for {', '.join(subjects)} with {study_hours} hours of study per day.
        Include:
        1. Daily breakdown of subjects
        2. Specific topics for each subject
        3. Suggested study techniques
        4. Short breaks between study sessions
        5. Weekly review sessions
        """
        schedule = get_ai_response(prompt)
        st.write(schedule)

        # Visualization: Study time distribution
        study_time = {subject: random.randint(1, study_hours) for subject in subjects}
        fig = go.Figure(data=[go.Pie(labels=list(study_time.keys()), values=list(study_time.values()))])
        fig.update_layout(title="Daily Study Time Distribution")
        st.plotly_chart(fig)

elif page == "Concept Explainer":
    st.header("💡 Concept Explainer")
    concept = st.text_input("Enter a concept you want to understand:")
    depth = st.select_slider("Explanation depth", options=["Basic", "Intermediate", "Advanced"])
    if st.button("Explain Concept"):
        prompt = f"""
        Explain the concept of '{concept}' at a {depth} level.
        Include:
        1. A clear definition
        2. Key principles or components
        3. Real-world applications or examples
        4. Any related concepts or theories
        5. Historical context or development of the concept
        """
        explanation = get_ai_response(prompt)
        st.write(explanation)

        # Visualization: Concept map
        concepts = explanation.split('\n')
        nodes = [concept] + [c.split(':')[0] for c in concepts if ':' in c]
        edges = [(concept, node) for node in nodes[1:]]
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = nodes,
              color = "blue"
            ),
            link = dict(
              source = [nodes.index(src) for src, _ in edges],
              target = [nodes.index(tgt) for _, tgt in edges],
              value = [1] * len(edges)
          ))])
        fig.update_layout(title_text="Concept Map", font_size=10)
        st.plotly_chart(fig)

elif page == "Progress Tracker":
    st.header("📊 Progress Tracker")
    subject = st.selectbox("Select a subject to track", ["Math", "Science", "History", "Literature", "Computer Science"])
    start_date = st.date_input("Start date of tracking")
    end_date = st.date_input("End date of tracking")

    # Simulated progress data
    date_range = pd.date_range(start=start_date, end=end_date)
    progress_data = pd.DataFrame({
        'Date': date_range,
        'Hours Studied': [random.randint(0, 5) for _ in range(len(date_range))],
        'Topics Covered': [random.randint(1, 3) for _ in range(len(date_range))],
        'Quiz Scores': [random.randint(60, 100) for _ in range(len(date_range))]
    })

    st.subheader("Study Progress")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=progress_data['Date'], y=progress_data['Hours Studied'], name='Hours Studied'))
    fig.add_trace(go.Scatter(x=progress_data['Date'], y=progress_data['Topics Covered'], name='Topics Covered'))
    fig.add_trace(go.Scatter(x=progress_data['Date'], y=progress_data['Quiz Scores'], name='Quiz Scores', yaxis='y2'))
    fig.update_layout(
        title='Study Progress Over Time',
        yaxis=dict(title='Hours / Topics'),
        yaxis2=dict(title='Quiz Scores', overlaying='y', side='right')
    )
    st.plotly_chart(fig)

    st.subheader("Study Statistics")
    total_hours = progress_data['Hours Studied'].sum()
    total_topics = progress_data['Topics Covered'].sum()
    avg_score = progress_data['Quiz Scores'].mean()
    st.write(f"Total hours studied: {total_hours}")
    st.write(f"Total topics covered: {total_topics}")
    st.write(f"Average quiz score: {avg_score:.2f}%")

elif page == "Flashcard Creator":
    st.header("🗂️ Flashcard Creator")
    subject = st.selectbox("Select a subject", ["Math", "Science", "History", "Literature", "Computer Science"])
    num_cards = st.slider("Number of flashcards", 1, 10, 5)
    if st.button("Generate Flashcards"):
        prompt = f"""
        Create {num_cards} flashcards for {subject}.
        For each flashcard, provide:
        1. The front side (question or term)
        2. The back side (answer or definition)
        Separate each flashcard with a newline.
        """
        flashcards = get_ai_response(prompt)
        cards = flashcards.split('\n\n')

        for i, card in enumerate(cards, 1):
            card_parts = card.split('\n')
            if len(card_parts) == 2:
                front, back = card_parts
                with st.expander(f"Flashcard {i} - {front}"):
                    st.success(back)

    st.subheader("Create Custom Flashcard")
    custom_front = st.text_input("Front of the card (question or term):")
    custom_back = st.text_input("Back of the card (answer or definition):")
    if st.button("Add Custom Flashcard"):
        with st.expander(custom_front):
            st.success(custom_back)
        st.success("Custom flashcard added successfully!")

st.sidebar.markdown("---")
st.sidebar.info("StudyBuddy AI uses artificial intelligence to assist with your learning. Always verify information and consult with your teachers or textbooks for authoritative answers.")
