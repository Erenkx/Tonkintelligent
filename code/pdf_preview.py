"""
Displays a preview of a PDF file on a specified page using Streamlit.
"""

import base64

import fitz
import streamlit as st


def display_pdf_preview(
    pdf_path: str, 
    page_number: int = 0,
    width: int = 600
) -> None:
    """
    Displays a preview of a PDF file on a specified page.

    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number to display (1-indexed). Default
            is 0.
        width (int): Width of the displayed image in pixels. Default
            is 600.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes('png')
        base64_img = base64.b64encode(img_bytes).decode('utf-8')

        st.markdown(
            f'<img src="data:image/png;base64,{base64_img}" '
            f'width="{width}px">',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f'Unable to load page {page_number}: {e}')
