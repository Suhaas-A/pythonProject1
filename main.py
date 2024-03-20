from database import Session
import streamlit as st

session = Session()


if __name__ == '__main__':
    st.switch_page('pages/login.py')
