from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register the custom fonts
pdfmetrics.registerFont(TTFont('Monospace', 'Monospace.ttf'))
pdfmetrics.registerFont(TTFont('MonospaceBold', 'MonospaceBold.ttf'))
pdfmetrics.registerFont(TTFont('MonospaceOblique', 'MonospaceOblique.ttf'))



def generate_pdf(file_name, data_dict, file_folder='./storage/data/pdf/'):
    file_path = file_folder + file_name + '.pdf'
    pdf = canvas.Canvas(file_path)

    # Set the font and font size
    pdf.setFont('Monospace', 12)

    # Set the dimensions and position of the background image
    image_path = './header.png'
    x = 0
    y = 0

    # Draw the background image
    pdf.drawImage(image_path, x, y, width=pdf._pagesize[0], height=pdf._pagesize[1]/2)

    # Set the dimensions and position of the logo
    logo_path = './miw.png'
    logo_width = 80
    logo_height = 50
    logo_x = pdf._pagesize[0] - logo_width - 20  # Adjust the values as needed
    logo_y = pdf._pagesize[1] - logo_height - 20  # Adjust the values as needed

    # Draw the logo on the PDF
    pdf.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height)

    # Set the dimensions and position of the logo
    banner_path = './banner.png'
    banner_width = 200
    banner_height = 35
    banner_x = pdf._pagesize[0] - banner_width - 350  # Adjust the values as needed
    banner_y = pdf._pagesize[1] - banner_height - 20  # Adjust the values as needed

    # Draw the logo on the PDF
    pdf.drawImage(banner_path, banner_x, banner_y, width=banner_width, height=banner_height)

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


if __name__ == '__main__':
    test_file_name = 'Fa_dict_test'
    test_text = {"Hello, Fa" : "I'm a function-generated PDF!"}

    generate_pdf(test_file_name, test_text)

    # Get a list of available fonts
    font_list = pdfmetrics.getRegisteredFontNames()
    # Print the list of fonts
    for font in font_list:
        print(font)

