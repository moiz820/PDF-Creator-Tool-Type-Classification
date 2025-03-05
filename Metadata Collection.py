import os
import shutil
import fitz  # PyMuPDF

# Define paths
pdf_directory = r"C:\Users\moizz\Downloads\Tool"  # Change this path
output_directory = r"C:\Users\moizz\Downloads\Safedocs"  # Where filtered PDFs will be moved

# List of Creator Tools to filter
allowed_creators = {
    "Microsoft® Word 2016",
    "Microsoft® Word 2010",
    "Microsoft® Word 2013",
    "Microsoft® Word for Microsoft 365",
    "Word",
    "Microsoft® Office Word 2007",
    "Microsoft® Word 2019",
    "Writer",
    "Pages",
    "Adobe InDesign 16.1 (Macintosh)",
    "Adobe InDesign CS6 (Windows)"
}

def extract_creator(pdf_path):
    """Extract the Creator Tool from PDF metadata."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata or {}
        return metadata.get("creator", "Unknown")
    except Exception as e:
        return "Error"

def filter_and_move_pdfs(pdf_directory, output_directory):
    """Filter PDFs based on Creator Tool and move them to respective directories."""
    if not os.path.exists(pdf_directory):
        print(f"Error: Directory '{pdf_directory}' does not exist.")
        return

    os.makedirs(output_directory, exist_ok=True)  # Ensure output directory exists

    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            creator = extract_creator(pdf_path)

            if creator in allowed_creators:
                creator_folder = os.path.join(output_directory, creator.replace(" ", "_"))  # Replace spaces with underscores
                os.makedirs(creator_folder, exist_ok=True)  # Create directory if not exists
                
                destination_path = os.path.join(creator_folder, pdf_file)
                shutil.move(pdf_path, destination_path)
                print(f"Moved: {pdf_file} → {creator_folder}")

    print("\n PDF Filtering and Transfer Completed!")

# Run the filtering and moving function
filter_and_move_pdfs(pdf_directory, output_directory)
##############################################################################################################
import os
import csv
import fitz  # PyMuPDF

# Define directory containing PDFs
pdf_directory = r"C:\Users\moizz\Downloads\Tools  # Change this path
output_csv = r"C:\Users\moizz\Downloads\safedoc.csv"

def extract_metadata(pdf_path):
    """Extract metadata from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata or {}

        creator = metadata.get("creator", "Unknown")
        producer = metadata.get("producer", "Unknown")

        return {"creator": creator, "producer": producer, "status": "Success"}
    except Exception as e:
        return {"creator": "N/A", "producer": "N/A", "status": f"Error: {str(e)}"}

def process_pdfs(directory, output_csv):
    """Scan all PDFs in a directory and create a CSV of metadata."""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Creator Tool", "Producer", "Status"])  # CSV header

        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            print(f"Processing: {pdf_file}")
            metadata = extract_metadata(pdf_path)

            writer.writerow([pdf_file, metadata["creator"], metadata["producer"], metadata["status"]])
            print(f"✅ Metadata extracted: {pdf_file}")

    print(f"\n📄 Metadata extraction completed! CSV saved at: {output_csv}")

# Run the script
process_pdfs(pdf_directory, output_csv)
