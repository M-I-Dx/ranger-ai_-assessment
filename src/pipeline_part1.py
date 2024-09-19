import os
import shutil
#from information_extraction.lamma import RFXFeatureExtractor
from data_processing.markdown_conversion import pdf_to_markdown


def process_pdf_directories(pdf_dirs, markdown_base_dir, info_base_dir):
    """
    Process PDF directories, convert PDFs to markdown, and extract relevant features.
    Args:
    - pdf_dirs (list): List of directories containing PDF files for different RFX types.
    - markdown_base_dir (str): Base directory where markdown files will be saved.
    - info_base_dir (str): Base directory where extracted information will be saved.
    """
    for pdf_dir in pdf_dirs:

        rfx_type = os.path.basename(pdf_dir)
        markdown_dir = os.path.join(markdown_base_dir, rfx_type)
        info_dir = os.path.join(info_base_dir, rfx_type)
        os.makedirs(markdown_dir, exist_ok=True)
        os.makedirs(info_dir, exist_ok=True)
        
        print(f"Processing PDFs in directory: {pdf_dir}")
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, pdf_file)
                markdown_file_name = os.path.splitext(pdf_file)[0] + ".md"
                markdown_file_path = os.path.join(markdown_dir, markdown_file_name)
                print(f"Converting {pdf_file} to markdown...")
                pdf_to_markdown(pdf_path, markdown_file_path)
        
        print(f"Extracting features from markdown files in {markdown_dir}...")
        markdown_files = [os.path.join(markdown_dir, file) for file in os.listdir(markdown_dir) if file.endswith(".md")]
        extractor = RFXFeatureExtractor(markdown_files)
        extractor.process_documents()
        for markdown_file in markdown_files:
            base_name = os.path.splitext(os.path.basename(markdown_file))[0]
            extracted_file = os.path.join(markdown_dir, f"{base_name}.txt")
            if os.path.exists(extracted_file):
                shutil.move(extracted_file, os.path.join(info_dir, f"{base_name}.txt"))

        print(f"Feature extraction completed for {pdf_dir}. Results saved in {info_dir}.\n")
    print("Pipeline finished.")

if __name__ == "__main__":
    main_pdf_dir = input("Enter the main PDF directory: ").strip()
    markdown_base_dir = input("Enter the markdown directory: ").strip()
    info_base_dir = input("Enter the info directory: ").strip()
    pdf_dirs = [os.path.join(main_pdf_dir, rfx_type) for rfx_type in os.listdir(main_pdf_dir)]
    process_pdf_directories(pdf_dirs, markdown_base_dir, info_base_dir)