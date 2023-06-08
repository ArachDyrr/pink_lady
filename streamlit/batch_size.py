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

def WIP():
    # move to page 99
    st.session_state.page += 99
    # Clear the image batch
    st.session_state.batch = None



#--------------------------------------------------

def page1():

    with page.container():

        st.title("AQL Image Batch Processor")

        st.button("Select AQL criteria", on_click=nextPage)

#--------------------------------------------------

def page2():

    with page.container():

        st.title("select protocol")

        # Select AQL criteria
        # 
        st.button("Make AI Work: advanced", on_click=WIP)

        # Select AQL criteria
        st.button("Make AI Work: standard", on_click=nextPage)




#--------------------------------------------------

def page3():

    with page.container():

        st.title("Selected AQL General I, Lot size 250-500 Batchsize 32")

        st.button("Accept", on_click=nextPage)
        
        st.button("Reject", on_click=restart)

#--------------------------------------------------

def page4():

    with page.container():
            
            st.title("Select imput type")
    
            st.button("upload 32 imagages.", on_click=nextPage)

            st.button("Take 32 live the images.", on_click=WIP)

#--------------------------------------------------
def page5():

    with page.container():

        st.title("Upload an image batch")

        # Store images
        st.session_state.batch = st.file_uploader("", type=None, accept_multiple_files=True)

        if st.session_state.batch is not None:

            st.image(st.session_state.batch)

        st.button("Process images", on_click=nextPage)

#--------------------------------------------------

def page99():
    with page.container():

        st.title("WIP: Work in progress")
        st.button("Press button to restart.", on_click=restart())

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

elif st.session_state.page == 4:

    page4()

elif st.session_state.page == 5:

    page5()


elif st.session_state.page >= 99:

    page99()
#--------------------------------------------------