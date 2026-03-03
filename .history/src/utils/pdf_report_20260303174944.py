from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO


def generate_pdf(name, age, stress_level, ai_report):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    normal_style = styles["Normal"]

    elements.append(Paragraph("AI Stress Intelligence Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Name: {name}", normal_style))
    elements.append(Paragraph(f"Age: {age}", normal_style))
    elements.append(Paragraph(f"Stress Level: {stress_level}", normal_style))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("AI Cognitive Analysis:", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(ai_report, normal_style))

    doc.build(elements)

    buffer.seek(0)
    return buffer