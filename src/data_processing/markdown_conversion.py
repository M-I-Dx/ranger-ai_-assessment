import pdfplumber
import markdownify


def pdf_to_markdown(pdf_path, markdown_path):
    """
    Convert a PDF file to a markdown file.
    Args:
    - pdf_path (str): Path to the input PDF file.
    - markdown_path (str): Path to the output markdown file.
    """
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"
    markdown_text = markdownify.markdownify(all_text, heading_style="ATX")

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    
    print(f"Markdown saved to {markdown_path}")
