import PyPDF2
from PyPDF2 import PdfReader

def inspect_pdf_header_footer(pdf_path):
    # Open the PDF file
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        
        # Access and print metadata
        metadata = reader.metadata
        print("Metadata in the PDF Header:")
        for key, value in metadata.items():
            print(f"{key}: {value}")

        # Access and print the trailer dictionary
        trailer = reader.trailer
        print("\nPDF Trailer Information:")
        for key in trailer.keys():
            print(f"{key}: {trailer[key]}")

        # Print the starting bytes to see the PDF version
        file.seek(0)
        print("\nPDF File Header (First Line):")
        print(file.readline().decode().strip())

# Example usage
inspect_pdf_header_footer("/Users/moizzia/Downloads/mypdf/Apple Pages/000618.pdf")
#####################################################################################################################################
import os
import fitz  # PyMuPDF

def remove_metadata(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each PDF in the directory
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open the PDF
            doc = fitz.open(input_path)

            # Clear metadata
            doc.set_metadata({})

            # Save the PDF without metadata
            doc.save(output_path)
            doc.close()
            print(f"Metadata removed from {filename}")

# Example usage
input_folder_path = "/Users/moizzia/Downloads/nohead/PScript5.dll"
output_folder_path = "/Users/moizzia/Downloads/head11/PScript5.dll"
remove_metadata(input_folder_path, output_folder_path)