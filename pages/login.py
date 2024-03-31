import streamlit as st
from database import *

st.set_page_config(initial_sidebar_state="collapsed")

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

with st.sidebar:
    st.divider()
    st.header('some name')
    st.divider()
    st.page_link(page='pages/login.py', label='login')
    st.page_link(page='pages/register.py', label='register')

session = Session()

st.title('Login')

st.divider()

with st.form(key='login_form'):
    username = st.text_input('Username: ')
    st.divider()

    password = st.text_input('Password: ', type='password')
    st.divider()

    feature = st.selectbox('Choose the feature', ['Virtual mouse', 'Sign language translator', 'Video translator'])
    st.divider()

    st.page_link('pages/forget_password.py')

    submit = st.form_submit_button('Login')

st.divider()

if submit:
    if not username:
        st.error('Username field is required')
    elif not password:
        st.error('Password field is required')
    else:
        users = session.query(Users).all()
        usernames, passwords, ids = [], [], []
        for user in users:
            usernames.append(user.username)
            passwords.append(user.password)
            ids.append(user.id)
        if username not in usernames:
            st.error('Invalid username')
        elif password not in passwords:
            st.error('Invalid password')
        else:
            user_id = ids[usernames.index(username)]
            st.session_state.user = [user_id, username]
            st.switch_page(f'pages/{feature}.py')

signup = st.button('Create an account')
if signup:
    st.switch_page('pages/register.py')
