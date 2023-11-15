import openai
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from gppod_utils_1 import SalesGPT
from gppod_utils_1 import conversation_stages
import random
openai.api_key  = st.secrets["OPENAI_API_KEY"]

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

verbose=True
llm = ChatOpenAI(model = "gpt-4-1106-preview",temperature = 0.5)
config = dict(
physician_name = "William",
company_name="InnoTech",
conversation_history=[],
conversation_stage = conversation_stages.get('3',"Recognize the signs and symptoms of common medical emergencies, such as Panic attack, Anxiety attack, Suicidal Thoughts, Homociadal Thoughts, Psychosis, Acute substance intoxication, Agitation and aggression, Extreme disorientation or confusion, Child or adolescent behavioral crises, Acute reactions to trauma, heart attack, stroke, allergic reaction, bleeding, poisoning, etc. Provide clear and concise instructions on how to perform first aid or CPR, or use an AED or an EpiPen, depending on the situation. Call the local emergency number for the user, or guide them to do so, and share their location and condition with the dispatcher Connect the user with a qualified medical professional, such as a doctor, nurse, or paramedic, who can offer further advice and support until help arrives. Consolidate the user if they need any mental health counseling in emergency situation, such as coping with stress, Panic attack, anxiety, trauma, or grief. *You must not refer the patients to any Mental Health professional. You must not say that i am unable to provide help*. You must provide help to the patient. You are Mental health professional help them accordingly"),
use_tools=False
)

def initialize_chain(human_input):
    if human_input!="":
        with st.spinner('Generating response...'):
            if st.session_state['count']==0:
                sales_agent = SalesGPT.from_llm(llm, verbose=False, **config)
                sales_agent.seed_agent() 
                st.session_state['sales_ag'] = sales_agent
                st.session_state['count'] = st.session_state['count'] +1
                # print(st.session_state['count'])
            else:
                sales_agent = st.session_state['sales_ag']
                # print(sales_agent)
            if human_input:
                sales_agent.human_step(human_input)
            print(sales_agent.determine_conversation_stage())
            ai_response = sales_agent.step()
            # print(sales_agent.conversation_history)
            ai_response =ai_response.replace("<END_OF_TURN>","").replace("<END_OF_CALL>","").replace("Ben: ","").replace('"',"")
            st.session_state['human'].append(human_input)
            st.session_state['ai'].append(ai_response)


if 'count' not in st.session_state:
    st.session_state['count'] = 0
if 'ai' not in st.session_state:
    st.session_state['ai'] = []
if 'human' not in st.session_state:
    st.session_state['human'] = []
if 'sales_ag' not in st.session_state:
    st.session_state['sales_ag'] = None
if 'user_msg' not in st.session_state:
    st.session_state['user_msg'] = None

def end_click():
    # st.session_state['prompts'] = [{"role": "system", "content": "You are a helpful assistant. Answer as concisely as possible with a little humor expression."}]
    st.session_state['count'] = 0
    st.session_state['ai'] = []
    st.session_state['human'] = []
    st.session_state['sales_ag'] = None

def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
  
    input_text = st.text_input(
        "You: ",
        # st.session_state['user'],
        key="user",
        placeholder="Type your message here...",
        # label_visibility="hidden",
    )
    # print("hello",input_text)
    return input_text


    


st.title("Virtual Assistant👨‍⚕️🩺👩‍⚕️")
st.markdown("Get medical advice and diagnosis from Virtual Assistant.")

user_input = get_text()
def clear_text():
    st.session_state["user"] = ""

col1, col2 = st.columns(2)
st.button("Clear text input", on_click=clear_text)


with col1:
    new_button = st.sidebar.button("New Chat", on_click=end_click, use_container_width=True)
with col2:
    end_button = st.sidebar.button("Clear History", on_click=end_click, use_container_width=True)

if user_input:
    initialize_chain(user_input)
if st.session_state['ai']:
    for i in range(len(st.session_state['ai'])-1, -1, -1):
        message(st.session_state['ai'][i], key=str(i))
        message(st.session_state['human'][i],
                is_user=True, key=str(i) + '_user')

