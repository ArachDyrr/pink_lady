#--------------------------------------------------

import streamlit as st

from PIL import Image
from modules.streamlit_functions import set_device
import torch
import torchvision.transforms as T
import io as io
from modules.MyAQLclass import MyAQLclass
from modules.pdf_modules.pdf_generator import generate_pdf

#--------------------------------------------------
imported_model_state_path = "./streamlit_20230622-111305_resnet18_more_pinky_loss.pt"
device = set_device()
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

        # st.button("Select AQL criteria", on_click=nextPage)
        st.button("Select AQL criteria", on_click=nextPage)

#--------------------------------------------------

def page2():

    with page.container():

        st.title("select protocol")

        # Select AQL criteria
        # 
        # st.button("Make AI Work: advanced", on_click=WIP)

        # Select AQL criteria
        st.button("Make AI Work: standard", on_click=nextPage)




#--------------------------------------------------

def page3():

    with page.container():

        st.title("Selected AQL General I, Lot size 250-500 Batchsize 32")
        st.divider()
        st.lotname = st.text_input("Lot name", value="EX20230630AQ2981")
        st.inspectorsname = st.text_input("Inspector's name", value="Fatima")
        st.divider()
        st.origin = st.text_input("Country code of origin", value="NED")
        st.destination = st.text_input("Country code of destination", value="GBR")
        st.incoterms = st.text_input("Incoterms", value="FCA")
        st.incotermslocation = st.text_input("Incoterms location", value="Rotterdam")
        st.countryoftesting = st.text_input("Country of testing", value="NED")
        st.contractor = st.text_input("Contractor", value="Pink Lady")
        st.divider()
        st.text(f'{st.inspectorsname} proceed with {st.lotname}')
        st.text(f'Origin: {st.origin} | Destination: {st.destination} | Contractor: {st.contractor}')
        st.divider()
        st.button(f"Accept AQL protocol", on_click=nextPage)
        
        st.button("Reject AQL protocol", on_click=restart)

#--------------------------------------------------

def page4():

    with page.container():
            
            st.title("Select input type")
    
            st.button("upload 32 imagages.", on_click=nextPage)

            # st.button("Take 32 live the images.", on_click=WIP)

#--------------------------------------------------
def page5():

    with page.container():

        st.title("Upload an image batch")

        # Store images
        st.session_state.batch = st.file_uploader("", type=None, accept_multiple_files=True)
        print("batch size: ", len(st.session_state.batch))

    
    if st.session_state.batch is not None:

        print("batch size: ", len(st.session_state.batch))

        st.button("Process images", on_click=nextPage)

#--------------------------------------------------
def page6():

    with page.container():

        st.title("AQL AI test results")
        st.divider()


    dataset_x = st.session_state.batch
   
    pil_images = []
   
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224,224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # T.ToTensor()
         ])

    for item in dataset_x:
        pil_image = Image.open(item)
        pil_images.append(pil_image)




    # # # Load the model
    model = torch.load(imported_model_state_path)
    model=model.to(device)
    model.eval()

    goodapplescore = 0
    badapplescore = 0
    for item in pil_images:
        with torch.no_grad():
            with st.spinner("working magic"):
                tensor_image = transform(item)
                tensor_image = tensor_image.unsqueeze(0)
                tensor_image = tensor_image.to(device)
                result = model(tensor_image)
                probabilities = torch.nn.functional.softmax(result[0], dim=0)
                if probabilities[0] > probabilities[1]:
                    goodapplescore += 1
                if probabilities[0] < probabilities[1]:
                    badapplescore += 1
    
    st.batchsize = len(pil_images)
    if goodapplescore + badapplescore != st.batchsize:
        st.text("Automated AQL test is unreliable. n\
                Please have the batch manually inspected.")


    st.goodapplescore = goodapplescore
    st.badapplescore = badapplescore


    # initiate the AQL class
    testingcase = MyAQLclass()
    testingcase.test_input = badapplescore
    st.testclass = testingcase.output()

    if st.session_state.batch is not None:
        st.text(f"batch size: {len(pil_images)}")
        st.text(f"accepted apples: {goodapplescore}")
        st.text(f"rejected apples: {badapplescore}")
        st.divider()
        st.subheader(f"The AQL quality class label for this lot is: class_{st.testclass}")
        st.divider()
        st.text("The AQL test is based on the following parameters:")
        st.text(f"Lot size: {testingcase.get_lotsize()} | Product class: {testingcase.get_product_class()} | Test inspection level: {testingcase.get_test_inspection_lvl()}")
        st.button(f"Accept", on_click=nextPage)
        st.button(f"Reject", on_click=restart)

#--------------------------------------------------
def page7():

    with page.container():

        st.title('The documents')
        st.text('The documents are they should appear in the designated locale shortly.')

        report_info = {
            "Lot name": st.lotname,
            "Taric": "0808 10 00 00",
            "Country of origin": st.origin,
            "Country of destination": st.destination,
            "Incoterms": st.incoterms,
            "Incoterms location": st.incotermslocation,
            "Country of testing": st.countryoftesting,
            "Contractor": st.contractor,
            "Inspector": st.inspectorsname,
            "Lot size": "500",
            "Test batch size" : st.batchsize,
            "Test inspection level": "G-I",
            "Accepted apples": st.goodapplescore,
            "Rejected apples": st.badapplescore,
            "Product class": st.testclass,
        }

        generate_pdf(st.lotname, report_info)


        st.button("Return to start", on_click=restart)

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

elif st.session_state.page == 6:
    
    page6()

elif st.session_state.page == 7:
    
    page7()

elif st.session_state.page >= 99:

    page99()
#--------------------------------------------------