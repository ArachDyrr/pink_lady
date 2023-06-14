from reportlab.pdfgen import canvas

# def generate_pdf(file_name, text, file_folder='./storage/data/pdf/'):
#     file_path = file_folder + file_name + '.pdf'
#     pdf = canvas.Canvas(file_path)

#     # Set the font and font size
#     pdf.setFont('Helvetica', 12)

#     # Set the dimensions and position of the background image
#     image_path = './header.png'
#     x = 0
#     y = 0

#     # Draw the background image
#     pdf.drawImage(image_path, x, y, width=pdf._pagesize[0], height=pdf._pagesize[1]/2)

#     # Set the dimensions and position of the logo
#     logo_path = './miw.png'
#     logo_width = 100
#     logo_height = 50
#     logo_x = pdf._pagesize[0] - logo_width - 20  # Adjust the values as needed
#     logo_y = pdf._pagesize[1] - logo_height - 20  # Adjust the values as needed

#     # Draw the logo on the PDF
#     pdf.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height)

#     # Draw text on the PDF in multiple lines
#     x_text = 100  # x-coordinate for the text
#     y_text = logo_y - 20  # y-coordinate below the logo
#     line_height = 15  # vertical distance between lines

#     for line in text.split('\n'):
#         pdf.drawString(x_text, y_text, line)
#         y_text -= line_height

#     pdf.save()


def generate_pdf(file_name, data_dict, file_folder='./storage/data/pdf/'):
    file_path = file_folder + file_name + '.pdf'
    pdf = canvas.Canvas(file_path)

    # Set the font and font size
    pdf.setFont('Helvetica', 12)

    # Set the dimensions and position of the background image
    image_path = './header.png'
    x = 0
    y = 0

    # Draw the background image
    pdf.drawImage(image_path, x, y, width=pdf._pagesize[0], height=pdf._pagesize[1]/2)

    # Set the dimensions and position of the logo
    logo_path = './miw.png'
    logo_width = 100
    logo_height = 50
    logo_x = pdf._pagesize[0] - logo_width - 20  # Adjust the values as needed
    logo_y = pdf._pagesize[1] - logo_height - 20  # Adjust the values as needed

    # Draw the logo on the PDF
    pdf.drawImage(logo_path, logo_x, logo_y, width=logo_width, height=logo_height)

    # Draw text on the PDF in multiple lines
    x_text = 100  # x-coordinate for the text
    y_text = logo_y - 20  # y-coordinate below the logo
    line_height = 15  # vertical distance between lines

    for key, value in data_dict.items():
        if isinstance(value, str) and '\n' in value:
            lines = value.split('\n')
            for line in lines:
                pdf.drawString(x_text, y_text, f'{key}: {line}')
                y_text -= line_height
        else:
            pdf.drawString(x_text, y_text, f'{key}: {value}')
            y_text -= line_height

    pdf.save()

if __name__ == '__main__':
    test_file_name = 'function_test_hellow_world_multiline_background_logo'
    test_text = "Hello, World \n I'm a function-generated PDF!"

    generate_pdf(test_file_name, test_text)
