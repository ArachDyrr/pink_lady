#--------------------------------------------------

import streamlit as st

from PIL import Image
# from modules.streamlit_functions import set_device, MyAQLclass, generate_pdf, generate_lotname, show_pdf, download_pdf
import torch
import torchvision.transforms as T
import io as io
#--------------------------------------------------

# imports
import numpy as np
# import torch
import torchvision.transforms as T
import base64
import streamlit as st

from collections import Counter
from datetime import datetime
from os import listdir, makedirs
from os.path import exists
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register the custom fonts
pdfmetrics.registerFont(TTFont("Monospace", "Monospace.ttf"))
pdfmetrics.registerFont(TTFont("MonospaceBold", "MonospaceBold.ttf"))
pdfmetrics.registerFont(TTFont("MonospaceOblique", "MonospaceOblique.ttf"))


# function to set device to GPU/mps if available
def set_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # print(f"Device is '{device}'")
    return device


class MyAQLclass:
    def __init__(
        self,
        test_input=None,
        lotsize=500,
        product_class=1,
        test_inspection_lvl="I",
        first_max_percentage=0.4,
        secund_max_percentage=6.5,
        third_max_percentage=15,
    ):
        self.test_input = test_input
        self.lotsize = lotsize
        self.product_class = product_class
        self.test_inspection_lvl = test_inspection_lvl
        self.first_max_percentage = first_max_percentage
        self.secund_max_percentage = secund_max_percentage
        self.third_max_percentage = third_max_percentage

    def get_test_input(self):
        return self.test_input

    def get_lotsize(self):
        return self.lotsize

    def get_product_class(self):
        return self.product_class

    def get_test_inspection_lvl(self):
        return self.test_inspection_lvl

    def get_first_percentage(self):
        return self.first_max_percentage

    def get_secund_percentage(self):
        return self.secund_max_percentage

    def get_third_percentage(self):
        return self.third_max_percentage

    def get_reject_percentage(self):
        return self.third_max_percentage

    def table_B(codeLeter="F", inspection_lvl="I"):
        if codeLeter == "F" and inspection_lvl == "I":
            return 32
        else:
            return None

    def table_A(self, lotsize, test_inspection_lvl):
        if lotsize >= 281 and lotsize <= 500 and test_inspection_lvl == "I":
            code_letter = "F"
            return code_letter
        else:
            return None

    def table_sample_size(self, codeLeter, acceptablePercent):
        sample_size = 0
        if codeLeter == "F" and acceptablePercent <= 0.4:
            sample_size = 32
        elif codeLeter == "F" and acceptablePercent <= 6.5:
            sample_size = 20
        elif codeLeter == "F" and acceptablePercent <= 15:
            sample_size = 20
        else:
            sample_size = None

        return sample_size

    def rejectionValue(self, codeLeter, acceptablePercent):
        reject_nr = 0
        if codeLeter == "F" and acceptablePercent <= 0.4:
            reject_nr = 1
        elif codeLeter == "F" and acceptablePercent <= 6.5:
            reject_nr = 2
        elif codeLeter == "F" and acceptablePercent <= 15:
            reject_nr = 6
        else:
            reject_nr = None

        return reject_nr

    def first_rejectionValue(codeLeter, acceptablePercent):
        pass

    def batch_size(self):
        letter_code = self.table_A(self.lotsize, self.test_inspection_lvl)
        batch_size_first = self.table_sample_size(
            letter_code, self.first_max_percentage
        )
        batch_size_secund = self.table_sample_size(
            letter_code, self.secund_max_percentage
        )
        batch_size_third = self.table_sample_size(
            letter_code, self.third_max_percentage
        )
        batch_size = max([batch_size_first, batch_size_secund, batch_size_third])

        return batch_size

    def cut_off_per_class(self):
        letter_code = self.table_A(self.lotsize, self.test_inspection_lvl)
        reject_value_first = self.rejectionValue(letter_code, self.first_max_percentage)
        reject_value_secund = self.rejectionValue(
            letter_code, self.secund_max_percentage
        )
        reject_value_third = self.rejectionValue(letter_code, self.third_max_percentage)
        cut_off_per_class = [
            reject_value_first,
            reject_value_secund,
            reject_value_third,
        ]
        return cut_off_per_class

    def AQL_test_parameters(self):
        AQL_test_parameters = [self.batch_size()] + list(self.cut_off_per_class())
        return AQL_test_parameters

    def output(self):
        input = self.test_input
        class_cutoff = self.cut_off_per_class()
        if input == None:
            raise ValueError("Input is empty")
        elif input < class_cutoff[0]:
            return "I"
        elif input < class_cutoff[1]:
            return "II"
        elif input < class_cutoff[2]:
            return "III"
        else:
            return "Rejected"
        


def generate_pdf(file_name, data_dict, file_folder="../streamlit/data/"):
    file_path = file_folder + file_name + ".pdf"
    if not exists(file_folder):
        makedirs(file_folder)
    pdf = canvas.Canvas(file_path)

    # Set the font and font size
    pdf.setFont("Monospace", 12)

    # Set the dimensions and position of the background image
    image_path = "./header.png"
    x = 0
    y = 0

    # Draw the background image
    pdf.drawImage(image_path, x, y, width=pdf._pagesize[0], height=pdf._pagesize[1] / 2)

    # Set the dimensions and position of the logo
    logo_path = "./miw.png"
    logo_width = 80
    logo_height = 50
    logo_x = pdf._pagesize[0] - logo_width - 20  # Adjust the values as needed
    logo_y = pdf._pagesize[1] - logo_height - 20  # Adjust the values as needed

    # Draw the logo on the PDF
    pdf.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height)

    # Set the dimensions and position of the logo
    banner_path = "./banner.png"
    banner_width = 200
    banner_height = 35
    banner_x = pdf._pagesize[0] - banner_width - 350  # Adjust the values as needed
    banner_y = pdf._pagesize[1] - banner_height - 20  # Adjust the values as needed

    # Draw the logo on the PDF
    pdf.drawImage(
        banner_path, banner_x, banner_y, width=banner_width, height=banner_height
    )

    # Set the dimensions and position of the boxes
    box_width = 400  # Updated width
    box_height = 15
    box_x = 100  # x-coordinate for the boxes
    box_y = logo_y - 20  # y-coordinate below the logo

    # Draw the boxes and text on the PDF
    for key, value in data_dict.items():
        # Draw the box
        pdf.rect(box_x, box_y, box_width, box_height)

        # Draw the key on the left side of the box
        pdf.drawString(box_x + 2, box_y + 2, key)

        # Draw the value on the right side of the box
        value_x = box_x + box_width - pdf.stringWidth(str(value)) - 2
        pdf.drawString(value_x, box_y + 2, str(value))

        # Update the y-coordinate for the next box
        box_y -= box_height + 5

    pdf.save()
    return file_path

def count_files_with_datestamp(datestamp, folder_path):
    file_names = listdir(folder_path)
    count = Counter(file for file in file_names if datestamp in file)
    capped_count = 1+ min(sum(count.values()), 9998)
    capped_count = str(capped_count).zfill(4)
    return capped_count

def generate_lotname (country="NED",type="EXA" ,file_folder="../streamlit/data/"):
    create_folder(file_folder)
    datestamp=datetime.now().strftime("%Y%m%d")
    dailycount=count_files_with_datestamp(datestamp, file_folder)
    lotname = type+country + datestamp + str(dailycount)
    return lotname

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)  

def download_pdf(file_path, lotname):
    with open(file_path, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(
        label=f"Download report {lotname} as .pdf",
        data=PDFbyte,
        file_name=st.lotname + ".pdf",
        mime="application/octet-stream",
    )

def create_folder(folder_path):
    if not exists(folder_path):
        makedirs(folder_path)
    return folder_path
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
        st.lotname = st.text_input("Lot name", value=generate_lotname())
        st.inspectorsname = st.text_input("Inspector's name", value="Fatima")
        st.divider()
        st.taric = st.text_input("Taric", value="080810 00 00")
        st.tradename = st.text_input("Trade name", value="Apples")
        st.origin = st.text_input("Country code of origin", value="NED")
        st.destination = st.text_input("Country code of destination", value="GBR")
        st.incoterms = st.text_input("Incoterms", value="FCA")
        st.incotermslocation = st.text_input("Incoterms location", value="Rotterdam")
        st.countryoftesting = st.text_input("Country of testing", value="NED")
        st.contractor = st.text_input("Contractor", value="Pink Lady")
        st.divider()
        st.text(f'{st.inspectorsname} accept to proceed with {st.lotname}')
        st.text(f'taric: {st.taric} | incoterms: {st.incoterms} | incoterms location: {st.incotermslocation}')
        st.text(f'Origin: {st.origin} | Destination: {st.destination} | Contractor: {st.contractor}')
        st.divider()
        st.button(f"Accept AQL protocol", on_click=nextPage)
        
        st.button("Reject AQL protocol", on_click=restart)

#--------------------------------------------------

def page4():

    with page.container():
            
            st.title("Select input type")
    
            st.button("upload 32 apple images.", on_click=nextPage)

            # st.button("Take 32 live the images.", on_click=WIP)

#--------------------------------------------------
def page5():

    with page.container():

        st.title("Upload an image batch")

        # Store images
        st.session_state.batch = st.file_uploader("", type='jpg', accept_multiple_files=True)

    
    if st.session_state.batch is not None:


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

        st.title('The document')

        report_info = {
            "Lot name": st.lotname,
            "Taric": st.taric,
            "Trade name": st.tradename,
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
            "AQL classification": f"class_{st.testclass}",
        }

        generated_pdf = generate_pdf(st.lotname, report_info)
  

        show_pdf(generated_pdf)

        download_pdf(generated_pdf, st.lotname)

        st.button("Return to start", on_click=restart)

#--------------------------------------------------
def page8():
    pass # WIP in case of more time a chat functionality as seen in factory/Apples_chat.ipynb would be implemented here.

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

elif st.session_state.page == 6:
    
    page6()

elif st.session_state.page == 7:
    
    page7()

elif st.session_state.page >= 99:

    page99()
#--------------------------------------------------