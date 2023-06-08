#--------------------------------------------------

import streamlit as st

from PIL import Image

#--------------------------------------------------

def nextPage():
    
    # Increment the page indicator
    st.session_state.page += 1

#--------------------------------------------------

def restart():
    
    # Reset the page indicator
    st.session_state.page = 1

    # Clear the image batch
    st.session_state.batch = None

#--------------------------------------------------

def page1():

    with page.container():

        st.title("Home screen")

        st.button("Go to upload", on_click=nextPage)

#--------------------------------------------------

def page2():

    with page.container():

        st.title("Upload an image batch")

        # Store images
        st.session_state.batch = st.file_uploader("", type=None, accept_multiple_files=True)

        if st.session_state.batch is not None:

            st.image(st.session_state.batch)

        st.button("Process images", on_click=nextPage)

#--------------------------------------------------

def page3():

    with page.container():

        st.title("Processing images")

        st.write(f"Number of images: {len(st.session_state.batch)}")
        
        st.balloons()

        st.button("Back to home", on_click=restart)


#--------------------------------------------------

# Note: Steamlit will run the code below
#       at the start and after button clicks

#--------------------------------------------------

# Only enters when the page indicator is not defined (at startup)
if "page" not in st.session_state:

    # Define the page indicator
    st.session_state.page = 1

    # To store image batch
    st.session_state.batch = None

#--------------------------------------------------

# Start with empty page
page = st.empty()

#--------------------------------------------------

# The flow of our app
if st.session_state.page == 1:

    page1()

elif st.session_state.page == 2:

    page2()

elif st.session_state.page == 3:

    page3()

#--------------------------------------------------