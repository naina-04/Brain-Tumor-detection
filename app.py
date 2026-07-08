import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import timm
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import numpy as np
import cv2

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .upload-section {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    .confidence-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 15px 40px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        font-size: 14px;
        color: #6b7280;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Brain Tumor Detection System")
st.markdown("<p style='text-align: center; color: #6b7280; font-size: 18px;'>AI-Powered Medical Imaging Analysis using Swin Transformer</p>", unsafe_allow_html=True)

class SwinTransformerBrainTumor(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

# Disease information database
DISEASE_INFO = {
    "glioma": {
        "description": "Gliomas are tumors that originate from glial cells in the brain and spinal cord. They are the most common type of primary brain tumor.",
        "symptoms": ["Headaches", "Seizures", "Memory loss", "Personality changes", "Nausea and vomiting"],
        "severity": "High",
        "recommendations": [
            "Immediate consultation with a neurosurgeon",
            "MRI with contrast for detailed imaging",
            "Biopsy to determine tumor grade",
            "Consider multidisciplinary tumor board review",
            "Discuss treatment options: surgery, radiation, chemotherapy"
        ],
        "medications": [
            {
                "name": "Temozolomide (Temodar)",
                "purpose": "Chemotherapy agent - First-line treatment for glioblastoma",
                "dosage": "150-200 mg/m² daily for 5 days every 28 days",
                "side_effects": "Nausea, fatigue, low blood counts, constipation",
                "precautions": "Monitor blood counts regularly, take on empty stomach"
            },
            {
                "name": "Bevacizumab (Avastin)",
                "purpose": "Targeted therapy - Blocks blood vessel growth to tumor",
                "dosage": "10 mg/kg IV every 2 weeks",
                "side_effects": "High blood pressure, bleeding, wound healing problems",
                "precautions": "Monitor blood pressure, avoid surgery for 28 days after last dose"
            },
            {
                "name": "Levetiracetam (Keppra)",
                "purpose": "Anti-seizure medication - Prevents seizures",
                "dosage": "500-1500 mg twice daily",
                "side_effects": "Drowsiness, dizziness, mood changes",
                "precautions": "Do not stop suddenly, may cause behavioral changes"
            },
            {
                "name": "Dexamethasone",
                "purpose": "Corticosteroid - Reduces brain swelling and inflammation",
                "dosage": "2-4 mg every 6-12 hours",
                "side_effects": "Weight gain, mood changes, increased blood sugar, insomnia",
                "precautions": "Take with food, taper dose gradually, monitor blood sugar"
            }
        ]
    },
    "meningioma": {
        "description": "Meningiomas are tumors that arise from the meninges, the membranes surrounding the brain and spinal cord. Most are benign and slow-growing.",
        "symptoms": ["Headaches", "Vision problems", "Hearing loss", "Memory loss", "Seizures"],
        "severity": "Moderate",
        "recommendations": [
            "Consultation with neurosurgeon for evaluation",
            "Regular monitoring with MRI scans",
            "Surgical removal if tumor is growing or causing symptoms",
            "Radiation therapy for inoperable cases",
            "Follow-up imaging every 6-12 months"
        ],
        "medications": [
            {
                "name": "Hydroxyurea",
                "purpose": "Chemotherapy - May slow tumor growth in inoperable cases",
                "dosage": "20-30 mg/kg/day orally",
                "side_effects": "Low blood counts, nausea, skin changes",
                "precautions": "Regular blood tests required, use contraception"
            },
            {
                "name": "Phenytoin (Dilantin)",
                "purpose": "Anti-seizure medication - Controls seizures",
                "dosage": "300-400 mg daily in divided doses",
                "side_effects": "Dizziness, drowsiness, gum overgrowth",
                "precautions": "Monitor drug levels, maintain good oral hygiene"
            },
            {
                "name": "Dexamethasone",
                "purpose": "Corticosteroid - Reduces swelling around tumor",
                "dosage": "2-4 mg daily as needed",
                "side_effects": "Weight gain, mood changes, increased appetite",
                "precautions": "Take with food, do not stop abruptly"
            },
            {
                "name": "Mifepristone (Korlym)",
                "purpose": "Progesterone receptor blocker - May slow tumor growth",
                "dosage": "300 mg daily (under clinical trial)",
                "side_effects": "Fatigue, nausea, low potassium",
                "precautions": "Experimental use, requires close monitoring"
            }
        ]
    },
    "pituitary": {
        "description": "Pituitary tumors develop in the pituitary gland and can affect hormone production. Most are benign adenomas.",
        "symptoms": ["Vision problems", "Hormonal imbalances", "Headaches", "Fatigue", "Unexplained weight changes"],
        "severity": "Moderate",
        "recommendations": [
            "Endocrinologist consultation for hormone evaluation",
            "Comprehensive hormone panel testing",
            "MRI of pituitary gland with dedicated sequences",
            "Consider transsphenoidal surgery if indicated",
            "Medication to control hormone levels"
        ],
        "medications": [
            {
                "name": "Cabergoline (Dostinex)",
                "purpose": "Dopamine agonist - Shrinks prolactin-secreting tumors",
                "dosage": "0.25-1 mg twice weekly",
                "side_effects": "Nausea, dizziness, nasal congestion",
                "precautions": "Take with food, monitor prolactin levels regularly"
            },
            {
                "name": "Bromocriptine (Parlodel)",
                "purpose": "Dopamine agonist - Reduces prolactin levels",
                "dosage": "2.5-15 mg daily in divided doses",
                "side_effects": "Nausea, headache, dizziness",
                "precautions": "Start with low dose, take with food"
            },
            {
                "name": "Octreotide (Sandostatin)",
                "purpose": "Somatostatin analog - Controls growth hormone excess",
                "dosage": "100-500 mcg subcutaneously 3 times daily",
                "side_effects": "Diarrhea, gallstones, injection site pain",
                "precautions": "Rotate injection sites, monitor gallbladder"
            },
            {
                "name": "Pasireotide (Signifor)",
                "purpose": "Somatostatin analog - For Cushing's disease",
                "dosage": "0.6-0.9 mg twice daily subcutaneously",
                "side_effects": "High blood sugar, diarrhea, nausea",
                "precautions": "Monitor blood glucose closely, may need diabetes medication"
            },
            {
                "name": "Levothyroxine (Synthroid)",
                "purpose": "Thyroid hormone replacement - For hypothyroidism",
                "dosage": "25-200 mcg daily (individualized)",
                "side_effects": "Heart palpitations if dose too high, weight changes",
                "precautions": "Take on empty stomach, regular thyroid function tests"
            }
        ]
    },
    "notumor": {
        "description": "No tumor detected. The brain tissue appears normal with no signs of abnormal growth or masses.",
        "symptoms": ["None related to tumor"],
        "severity": "None",
        "recommendations": [
            "Continue regular health check-ups",
            "Maintain healthy lifestyle habits",
            "If symptoms persist, consult with neurologist",
            "Follow up if new symptoms develop",
            "Keep records for future reference"
        ],
        "medications": [
            {
                "name": "No medication required",
                "purpose": "No tumor detected - No specific treatment needed",
                "dosage": "N/A",
                "side_effects": "N/A",
                "precautions": "Maintain healthy lifestyle, regular check-ups"
            }
        ]
    }
}

def estimate_tumor_size(image, prediction):
    """
    Estimate tumor size using image processing techniques
    Returns tumor area percentage and size category
    """
    if prediction.lower() == 'notumor' or 'no tumor' in prediction.lower():
        return 0.0, "N/A - No Tumor Detected", None
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to detect abnormal regions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate total image area
    total_area = gray.shape[0] * gray.shape[1]
    
    # Find the largest contour (likely the tumor)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        tumor_area = cv2.contourArea(largest_contour)
        
        # Calculate percentage
        area_percentage = (tumor_area / total_area) * 100
        
        # Create visualization
        vis_image = img_array.copy()
        cv2.drawContours(vis_image, [largest_contour], -1, (255, 0, 0), 2)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text with size
        cv2.putText(vis_image, f"Tumor Region", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Determine size category
        if area_percentage < 5:
            size_category = "Small"
        elif area_percentage < 15:
            size_category = "Medium"
        elif area_percentage < 30:
            size_category = "Large"
        else:
            size_category = "Very Large"
        
        # Convert back to PIL
        vis_pil = Image.fromarray(vis_image)
        
        return area_percentage, size_category, vis_pil
    else:
        return 0.0, "Unable to detect", None

def generate_pdf_report(patient_name, patient_id, age, gender, prediction, confidence, all_probs, class_names, image, tumor_size_percent, tumor_size_category):
    """Generate a professional PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    # Title
    story.append(Paragraph("🧠 BRAIN TUMOR DETECTION REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Header line
    story.append(Table([['']], colWidths=[7*inch], style=[
        ('LINEABOVE', (0,0), (-1,0), 2, colors.HexColor('#667eea')),
    ]))
    story.append(Spacer(1, 0.2*inch))
    
    # Report Information
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    report_data = [
        ['Report Generated:', report_date],
        ['Analysis Method:', 'Swin Transformer Deep Learning Model'],
        ['Model Accuracy:', '99.08%']
    ]
    
    report_table = Table(report_data, colWidths=[2*inch, 4.5*inch])
    report_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(report_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information
    story.append(Paragraph("PATIENT INFORMATION", heading_style))
    patient_data = [
        ['Patient Name:', patient_name],
        ['Patient ID:', patient_id],
        ['Age:', age],
        ['Gender:', gender]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # MRI Image Section
    if image is not None:
        story.append(Paragraph("MRI SCAN IMAGE", heading_style))
        
        # Convert PIL image to BytesIO for ReportLab
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Add image to PDF (centered, reasonable size)
        img_width = 3.5 * inch
        img_height = 3.5 * inch
        
        # Create a table to center the image
        mri_image = RLImage(img_buffer, width=img_width, height=img_height)
        image_table = Table([[mri_image]], colWidths=[7*inch])
        image_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(image_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Diagnosis Result
    story.append(Paragraph("DIAGNOSIS RESULT", heading_style))
    
    result_data = [
        ['Prediction:', prediction.upper()],
        ['Confidence Level:', f'{confidence:.2f}%'],
        ['Severity:', DISEASE_INFO[prediction.lower().replace(' ', '').replace('_', '')]['severity']],
        ['Tumor Size:', tumor_size_category],
        ['Affected Area:', f'{tumor_size_percent:.2f}%' if tumor_size_percent > 0 else 'N/A']
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 4.5*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#dbeafe')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Probability Distribution
    story.append(Paragraph("PROBABILITY DISTRIBUTION", heading_style))
    prob_data = [['Class', 'Probability']]
    for i, cls in enumerate(class_names):
        prob_data.append([cls.upper(), f'{all_probs[i]:.2f}%'])
    
    prob_table = Table(prob_data, colWidths=[3*inch, 3.5*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')])
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Disease Information
    disease_key = prediction.lower().replace(' ', '').replace('_', '')
    disease_data = DISEASE_INFO[disease_key]
    
    story.append(Paragraph("DISEASE INFORMATION", heading_style))
    story.append(Paragraph(disease_data['description'], normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Symptoms
    story.append(Paragraph("<b>Common Symptoms:</b>", normal_style))
    for symptom in disease_data['symptoms']:
        story.append(Paragraph(f"• {symptom}", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    story.append(Paragraph("MEDICAL RECOMMENDATIONS", heading_style))
    for i, rec in enumerate(disease_data['recommendations'], 1):
        story.append(Paragraph(f"{i}. {rec}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Medication Information
    story.append(Paragraph("RECOMMENDED MEDICATIONS", heading_style))
    
    if disease_key != 'notumor':
        for idx, med in enumerate(disease_data['medications'], 1):
            # Medication name and purpose
            med_header = f"<b>{idx}. {med['name']}</b>"
            story.append(Paragraph(med_header, normal_style))
            
            # Create medication details table
            med_data = [
                ['Purpose:', med['purpose']],
                ['Typical Dosage:', med['dosage']],
                ['Common Side Effects:', med['side_effects']],
                ['Precautions:', med['precautions']]
            ]
            
            med_table = Table(med_data, colWidths=[1.5*inch, 5*inch])
            med_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eff6ff')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(med_table)
            story.append(Spacer(1, 0.15*inch))
        
        # Important medication note
        med_note_style = ParagraphStyle(
            'MedNote',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#dc2626'),
            alignment=TA_JUSTIFY,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=5
        )
        story.append(Paragraph(
            "<b>⚠️ IMPORTANT:</b> These medications are commonly used but must be prescribed by a qualified physician. "
            "Dosages are individualized based on patient factors. Never self-medicate. Always consult your healthcare provider "
            "before starting, stopping, or changing any medication.",
            med_note_style
        ))
    else:
        story.append(Paragraph("No medication required. The scan shows no signs of tumor.", normal_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(Spacer(1, 0.2*inch))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#ef4444'),
        alignment=TA_JUSTIFY,
        leftIndent=20,
        rightIndent=20
    )
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an AI-based system for educational and research purposes only. "
        "It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare "
        "professionals for accurate diagnosis and treatment decisions. Tumor size estimation is approximate and based on image processing. "
        "Medication information is for reference only and must be prescribed by a licensed physician. Dosages and treatment plans "
        "should be individualized based on patient-specific factors.",
        disclaimer_style
    ))
    
    # Footer
    story.append(Spacer(1, 0.2*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph(f"Brain Tumor Detection System | Generated on {report_date}", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('swin_brain_tumor_complete.pth', map_location=device, weights_only=False)
    
    # CRITICAL: Use exact class names from checkpoint to match training order
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    
    model = SwinTransformerBrainTumor(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device, class_names, checkpoint

model, device, class_names, checkpoint = load_model()

# Sidebar - Model Information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown(f"""
    <div class='info-card'>
        <p><b>Status:</b> ✅ Active</p>
        <p><b>Architecture:</b> Swin Transformer</p>
        <p><b>Accuracy:</b> {checkpoint.get('accuracy', 0)*100:.2f}%</p>
        <p><b>Classes:</b> {len(class_names)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Detected Classes")
    for cls in class_names:
        st.markdown(f"• {cls.upper()}")
    
    st.markdown("### ℹ️ About")
    st.info(
        "This AI system uses state-of-the-art Swin Transformer architecture "
        "to analyze brain MRI scans and detect potential tumors with 99.08% accuracy."
    )
    
    st.markdown("### ⚠️ Important Note")
    st.warning(
        "This tool is for educational purposes only. "
        "Always consult healthcare professionals for medical diagnosis."
    )

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear MRI scan image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    else:
        st.info("👆 Please upload an MRI scan image to begin analysis")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Patient Information Form
    if uploaded_file:
        st.markdown("### 👤 Patient Information")
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
        
        col_a, col_b = st.columns(2)
        with col_a:
            patient_id = st.text_input("Patient ID", placeholder="e.g., P12345")
            age = st.text_input("Age", placeholder="e.g., 45")
        with col_b:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file:
        st.markdown("### 🔬 Analysis Results")
        
        if st.button("🚀 Analyze MRI Scan", use_container_width=True):
            with st.spinner("🔄 Analyzing MRI scan... Please wait..."):
                # Image preprocessing
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # Prediction
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                
                result = class_names[predicted.item()]
                conf = confidence.item() * 100
                
                # Estimate tumor size
                tumor_size_percent, tumor_size_category, tumor_vis = estimate_tumor_size(image, result)
                
                # Store results in session state
                st.session_state.result = result
                st.session_state.confidence = conf
                st.session_state.probs = probs[0].cpu().numpy()
                st.session_state.image = image
                st.session_state.tumor_size_percent = tumor_size_percent
                st.session_state.tumor_size_category = tumor_size_category
                st.session_state.tumor_vis = tumor_vis
                st.session_state.analyzed = True
        
        # Display results if analysis is done
        if hasattr(st.session_state, 'analyzed') and st.session_state.analyzed:
            result = st.session_state.result
            conf = st.session_state.confidence
            probs = st.session_state.probs
            tumor_size_percent = st.session_state.tumor_size_percent
            tumor_size_category = st.session_state.tumor_size_category
            tumor_vis = st.session_state.tumor_vis
            
            # Main prediction result
            st.markdown(f"""
            <div class='result-box'>
                🎯 DIAGNOSIS: {result.upper()}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence score
            st.markdown(f"""
            <div class='confidence-box'>
                <h3 style='margin:0; color: #059669;'>Confidence Score</h3>
                <p style='font-size: 32px; font-weight: bold; margin: 10px 0; color: #1e3a8a;'>{conf:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Tumor Size Information
            if tumor_size_category != "N/A - No Tumor Detected":
                st.markdown(f"""
                <div class='info-card' style='border-left: 5px solid #f59e0b;'>
                    <h3 style='margin:0; color: #f59e0b;'>📏 Tumor Size Analysis</h3>
                    <p style='font-size: 20px; font-weight: bold; margin: 10px 0; color: #1e3a8a;'>{tumor_size_category}</p>
                    <p style='margin: 5px 0;'>Affected Area: <b>{tumor_size_percent:.2f}%</b> of scan region</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show tumor visualization if available
                if tumor_vis is not None:
                    with st.expander("🔍 View Tumor Region Detection"):
                        st.image(tumor_vis, caption="Detected Tumor Region (Green box indicates approximate location)", use_container_width=True)
                        st.info("Note: This is an approximate visualization based on image processing. Clinical imaging tools provide more accurate measurements.")
            else:
                st.markdown(f"""
                <div class='info-card' style='border-left: 5px solid #10b981;'>
                    <h3 style='margin:0; color: #10b981;'>✅ No Tumor Detected</h3>
                    <p style='margin: 10px 0;'>The scan appears normal with no signs of tumor.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability distribution
            st.markdown("#### 📊 Probability Distribution")
            for i, cls in enumerate(class_names):
                prob_value = float(probs[i]) * 100
                st.progress(float(probs[i]), text=f"{cls.upper()}: {prob_value:.2f}%")
            
            # Disease Information
            disease_key = result.lower().replace(' ', '').replace('_', '')
            if disease_key in DISEASE_INFO:
                disease_data = DISEASE_INFO[disease_key]
                
                st.markdown("#### 📋 Disease Information")
                st.markdown(f"<div class='info-card'>", unsafe_allow_html=True)
                st.markdown(f"**Description:** {disease_data['description']}")
                st.markdown(f"**Severity Level:** {disease_data['severity']}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                with st.expander("🔍 View Symptoms & Recommendations"):
                    st.markdown("**Common Symptoms:**")
                    for symptom in disease_data['symptoms']:
                        st.markdown(f"• {symptom}")
                    
                    st.markdown("**Medical Recommendations:**")
                    for i, rec in enumerate(disease_data['recommendations'], 1):
                        st.markdown(f"{i}. {rec}")
                
                # Medication Information
                st.markdown("#### 💊 Recommended Medications")
                
                if disease_key != 'notumor':
                    st.info("⚠️ These medications must be prescribed by a qualified physician. Dosages are individualized. Never self-medicate.")
                    
                    for idx, med in enumerate(disease_data['medications'], 1):
                        with st.expander(f"💊 {idx}. {med['name']}"):
                            st.markdown(f"**Purpose:** {med['purpose']}")
                            st.markdown(f"**Typical Dosage:** {med['dosage']}")
                            st.markdown(f"**Common Side Effects:** {med['side_effects']}")
                            st.markdown(f"**Precautions:** {med['precautions']}")
                else:
                    st.success("✅ No medication required. The scan shows no signs of tumor.")
            
            # Download PDF Report
            st.markdown("#### 📄 Generate Report")
            
            if patient_name and patient_id:
                # Generate PDF
                all_probs_percent = [p * 100 for p in probs]
                pdf_buffer = generate_pdf_report(
                    patient_name=patient_name,
                    patient_id=patient_id,
                    age=age if age else "N/A",
                    gender=gender,
                    prediction=result,
                    confidence=conf,
                    all_probs=all_probs_percent,
                    class_names=class_names,
                    image=st.session_state.image,
                    tumor_size_percent=tumor_size_percent,
                    tumor_size_category=tumor_size_category
                )
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"Brain_Tumor_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ Please fill in Patient Name and Patient ID to generate PDF report")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p><b>Brain Tumor Detection System</b> | Powered by Swin Transformer AI</p>
    <p style='font-size: 12px;'>⚠️ For Educational and Research Purposes Only | Not for Clinical Use</p>
    <p style='font-size: 12px;'>© 2024 | Developed as Final Year Project</p>
</div>
""", unsafe_allow_html=True)
