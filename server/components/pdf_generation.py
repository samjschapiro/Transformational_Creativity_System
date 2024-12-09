import os
import uuid
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch

def generate_output_pdf(logic_text, english_text, output_path):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch

    # Ensure all arrows are displayed correctly
    logic_text = logic_text.replace('■ightarrow', '→')
    logic_text = logic_text.replace('\\rightarrow', '→')
    english_text = english_text.replace('■ightarrow', '→')
    english_text = english_text.replace('\\rightarrow', '→')

    c = canvas.Canvas(output_path, pagesize=LETTER)
    width, height = LETTER
    c.setFont("Helvetica", 12)
    margin = 1 * inch
    y = height - margin

    # If neither logic nor English sections have content, we can title the whole PDF differently
    has_logic = bool(logic_text.strip())
    has_english = bool(english_text.strip())

    if not has_logic and not has_english:
        # No content at all
        c.drawString(margin, y, "No formalized content found.")
        c.save()
        return

    # Title for the PDF
    c.drawString(margin, y, "Formalized Argument Reconstructions")
    y -= 0.5 * inch

    # Draw logic section if available
    if has_logic:
        logic_lines = logic_text.split('\n')
        for line in logic_lines:
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - margin
            c.drawString(margin, y, line)
            y -= 0.3 * inch

        # Add spacing before English section
        y -= 0.5 * inch
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - margin

    # Draw English section if available
    if has_english:
        english_lines = english_text.split('\n')
        for line in english_lines:
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - margin
            c.drawString(margin, y, line)
            y -= 0.3 * inch

    c.save()
