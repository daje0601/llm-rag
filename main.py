import streamlit as st 


st.set_page_config(
    page_title="Test Home",
    page_icon="🤣",
)

st.title("test Home")

with st.sidebar:
    st.title("sidbar tilte")
    st.text_input("xxx")



tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])