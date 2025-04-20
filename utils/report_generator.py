"""
Report generator for the Symptom Checker application.
This module creates printable and shareable reports from symptom analysis results.
"""

import json
import os
import datetime
import tempfile
import base64
import io
import uuid
from jinja2 import Template
from weasyprint import HTML
import qrcode
from PIL import Image

def format_date(date_obj=None):
    """Format date for reports"""
    if date_obj is None:
        date_obj = datetime.datetime.now()
    return date_obj.strftime("%B %d, %Y at %I:%M %p")

def generate_qr_code(data, size=200, save_path=None):
    """
    Generate a QR code for the given data.
    
    Args:
        data (str): Data to encode in the QR code
        size (int): Size of the QR code image in pixels
        save_path (str, optional): Path to save the QR code image
        
    Returns:
        str: Base64 encoded QR code image for embedding in HTML
    """
    from pathlib import Path
    import datetime
    
    # Ensure QR codes directory exists
    qr_codes_dir = Path("static/qr_codes")
    qr_codes_dir.mkdir(parents=True, exist_ok=True)
    
    # Create QR code instance
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    # Add data to the QR code
    qr.add_data(data)
    qr.make(fit=True)
    
    # Create an image from the QR code
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Resize if needed
    img = img.resize((size, size))
    
    # Save to specified path or generate a default one
    if save_path:
        img.save(save_path)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"static/qr_codes/qr_code_{timestamp}.png"
        img.save(default_path)
    
    # Convert to base64 for embedding in HTML
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def generate_html_report(analysis, symptoms, user_data):
    """
    Generate an HTML report from the symptom analysis results.
    
    Args:
        analysis (dict): The symptom analysis results
        symptoms (list): List of reported symptoms
        user_data (dict): User information (age, gender, etc.)
        
    Returns:
        str: HTML report content
    """
    # HTML report template
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Symptom Analysis Report for {{ user_data.patient_name }}</title>
        <style>
            body {
                font-family: 'Helvetica', 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #ddd;
            }
            .header h1 {
                color: #1E88E5;
                margin-bottom: 5px;
            }
            .header h2 {
                color: #1976D2;
                margin-top: 0;
                margin-bottom: 15px;
            }
            .report-date {
                color: #666;
                font-style: italic;
            }
            .app-name {
                font-weight: bold;
                color: #1976D2;
                display: block;
                margin-bottom: 10px;
                font-size: 1.2em;
            }
            .section {
                margin-bottom: 30px;
            }
            .section h2 {
                color: #1E88E5;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .user-info {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .user-info table {
                width: 100%;
            }
            .user-info table td {
                padding: 8px;
            }
            .user-info table td:first-child {
                font-weight: bold;
                width: 150px;
            }
            .symptoms-list {
                columns: 2;
                margin-bottom: 20px;
            }
            .risk-level {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 15px;
                font-weight: bold;
                text-transform: uppercase;
                margin-bottom: 10px;
            }
            .risk-high {
                background-color: #ffebee;
                color: #c62828;
                border: 1px solid #ef9a9a;
            }
            .risk-moderate {
                background-color: #fff8e1;
                color: #ff8f00;
                border: 1px solid #ffe082;
            }
            .risk-low {
                background-color: #e8f5e9;
                color: #2e7d32;
                border: 1px solid #a5d6a7;
            }
            .risk-unknown {
                background-color: #E0E0E0;
                color: #757575;
                border: 1px solid #BDBDBD;
            }
            .condition-card {
                border: 1px solid #eee;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 5px;
            }
            .condition-card h3 {
                margin-top: 0;
                color: #1E88E5;
            }
            .medical-advice {
                background-color: #E3F2FD;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .disclaimer {
                background-color: #FFF8E1;
                padding: 15px;
                border-radius: 5px;
                margin-top: 30px;
                font-size: 0.9em;
            }
            .sources {
                font-size: 0.9em;
                color: #666;
            }
            .print-button {
                background-color: #1E88E5;
                color: white;
                border: none;
                padding: 10px 15px;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;
                margin-bottom: 20px;
            }
            .print-button:hover {
                background-color: #1976D2;
            }
            @media print {
                .print-button {
                    display: none;
                }
                body {
                    font-size: 12pt;
                    color: #000;
                }
                .header h1 {
                    color: #000;
                }
                .section h2, .condition-card h3 {
                    color: #000;
                }
                .disclaimer {
                    border: 1px solid #ccc;
                }
                .doctor-reference {
                    page-break-inside: avoid;
                    border: 1px solid #000 !important;
                    background-color: #f9f9f9 !important;
                }
                .doctor-reference h3 {
                    color: #000 !important; 
                }
                /* Add a footer with page numbers */
                @page {
                    margin: 1cm;
                    @bottom-center {
                        content: "Page " counter(page) " of " counter(pages);
                        font-size: 10pt;
                    }
                }
            }
        </style>
    </head>
    <body>
        <button class="print-button" onclick="window.print()">Print Report</button>
        
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="app-name"> Birla AI Health Advisor - Medical Report</span>
                    <h1>Symptom Analysis Report</h1>
                    <h2>Patient: {{ user_data.patient_name }}</h2>
                    <p class="report-date">Generated on {{ report_date }}</p>
                    {% if user_data.patient_image %}
                    <div style="margin-top: 15px;">
                        <img src="data:image/jpeg;base64,{{ user_data.patient_image }}" 
                             alt="Patient Photo" 
                             style="max-width: 200px; border-radius: 8px; border: 1px solid #ddd;">
                    </div>
                    {% endif %}
                </div>
                <div style="text-align: center;">
                    {% if qr_code_image %}
                    <img src="{{ qr_code_image }}" alt="QR Code" style="width: 120px; height: 120px;">
                    <p style="font-size: 0.8em; margin-top: 5px; color: #666;">Scan for digital report</p>
                    {% endif %}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Patient Information</h2>
            <div class="user-info">
                <table>
                    <tr>
                        <td>Age:</td>
                        <td>{{ user_data.age }}</td>
                    </tr>
                    <tr>
                        <td>Gender:</td>
                        <td>{{ user_data.gender }}</td>
                    </tr>
                    <tr>
                        <td>Symptom Duration:</td>
                        <td>{{ user_data.duration }}</td>
                    </tr>
                    <tr>
                        <td>Symptom Severity:</td>
                        <td>{{ user_data.severity }}</td>
                    </tr>
                    {% if user_data.additional_info %}
                    <tr>
                        <td>Additional Info:</td>
                        <td>{{ user_data.additional_info }}</td>
                    </tr>
                    {% endif %}
                </table>
            </div>
        </div>
        
        <div class="section">
            <h2>Reported Symptoms</h2>
            <div class="symptoms-list">
                <ul>
                    {% for symptom in symptoms %}
                    <li>{{ symptom }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>Analysis Results</h2>
            
            <h3>Risk Assessment</h3>
            {% if analysis.risk_level.lower() == "high" %}
                <div class="risk-level risk-high">High Risk</div>
                {% if analysis.seek_medical_attention %}
                <p><strong>⚠️ Recommendation: Please seek medical attention as soon as possible.</strong></p>
                {% endif %}
            {% elif analysis.risk_level.lower() == "moderate" %}
                <div class="risk-level risk-moderate">Moderate Risk</div>
                {% if analysis.seek_medical_attention %}
                <p><strong>⚠️ Recommendation: Please consult with a healthcare provider.</strong></p>
                {% endif %}
            {% elif analysis.risk_level.lower() == "low" %}
                <div class="risk-level risk-low">Low Risk</div>
                <p>Based on your symptoms, immediate medical attention may not be necessary, but consult a healthcare provider if symptoms persist or worsen.</p>
            {% else %}
                <div class="risk-level risk-unknown">Unknown Risk</div>
                <p>Risk level could not be determined. Please consult a healthcare provider.</p>
            {% endif %}
            
            <h3>Possible Conditions</h3>
            {% if analysis.possible_conditions %}
                {% for condition in analysis.possible_conditions %}
                <div class="condition-card">
                    <h3>{{ condition.name }}</h3>
                    <p><strong>Description:</strong> {{ condition.description }}</p>
                    
                    <p><strong>Common Symptoms:</strong></p>
                    <ul>
                        {% for symptom in condition.common_symptoms %}
                        <li>{{ symptom }}</li>
                        {% endfor %}
                    </ul>
                    
                    {% if condition.diet_recommendations %}
                    <p><strong>Diet Recommendations:</strong></p>
                    <ul>
                        {% for recommendation in condition.diet_recommendations %}
                        <li>{{ recommendation }}</li>
                        {% endfor %}
                    </ul>
                    {% endif %}
                </div>
                {% endfor %}
            {% else %}
                <p>No specific conditions identified based on the provided symptoms.</p>
            {% endif %}
            
            <h3>General Health Advice</h3>
            <div class="medical-advice">
                {{ analysis.general_advice }}
            </div>
            
            {% if analysis.medical_sources %}
            <h3>Medical Sources</h3>
            <div class="sources">
                <ul>
                    {% for source in analysis.medical_sources %}
                    <li><a href="{{ source }}">{{ source }}</a></li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        
        <div class="doctor-reference" style="margin: 2rem 0; padding: 1.5rem; border: 1px dashed #1E88E5; border-radius: 8px;">
            <h3 style="color: #1E88E5; margin-top: 0;">For Healthcare Provider Reference</h3>
            <p>This section is designed to help healthcare providers quickly review this patient's reported symptoms.</p>
            
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 1rem;">
                <div style="flex: 2;">
                    <h4>Summary:</h4>
                    <ul>
                        <li><strong>Patient:</strong> {{ user_data.patient_name }}</li>
                        <li><strong>Age/Gender:</strong> {{ user_data.age }}/{{ user_data.gender }}</li>
                        <li><strong>Risk Assessment:</strong> {{ analysis.risk_level }}</li>
                        <li><strong>Primary Symptoms:</strong> {{ symptoms|join(', ') }}</li>
                        <li><strong>Duration:</strong> {{ user_data.duration }}</li>
                        <li><strong>Report ID:</strong> {{ report_id }}</li>
                    </ul>
                </div>
                <div style="flex: 1; text-align: center;">
                    {% if qr_code_image %}
                    <img src="{{ qr_code_image }}" alt="QR Code" style="width: 120px; height: 120px; margin-bottom: 8px;">
                    <p style="font-size: 0.8em; margin: 0; color: #666;">Scan for digital reference</p>
                    {% endif %}
                </div>
            </div>
            
            <div style="margin-top: 1rem; border-top: 1px dotted #ccc; padding-top: 1rem;">
                <p><strong>Healthcare Provider Notes:</strong></p>
                <div style="height: 80px; border: 1px solid #ddd; border-radius: 4px;"></div>
            </div>
        </div>
        
        <div class="disclaimer">
            <h3>Medical Disclaimer</h3>
            <p><strong>Important:</strong> This report is generated based on the symptoms you provided and is NOT a medical diagnosis.
            The information provided should not replace professional medical advice, diagnosis, or treatment.
            Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
            <p>If you are experiencing a medical emergency, please call your local emergency number or go to the nearest emergency room immediately.</p>
        </div>
        
        <div style="margin-top: 3rem; text-align: center; color: #777; font-size: 0.8em; border-top: 1px solid #eee; padding-top: 1rem;">
            <p>Report ID: {{ report_id }} | Generated on {{ report_date }} | AI Health Advisor v1.0</p>
        </div>
        
        <script>
            // Add automatic print dialog when the button is clicked
            document.querySelector('.print-button').addEventListener('click', function() {
                window.print();
            });
        </script>
    </body>
    </html>
    """
    
    # Create template and render with data
    template = Template(template_str)
    
    # Generate a unique report ID
    report_id = str(uuid.uuid4())
    
    # Create QR code data - we'll include a comprehensive summary for medical professionals
    qr_data = {
        "report_id": report_id,
        "patient": user_data.get('patient_name', 'Unknown'),
        "age": user_data.get('age', 'Unknown'),
        "gender": user_data.get('gender', 'Unknown'),
        "timestamp": format_date(),
        "risk_level": analysis.get('risk_level', 'Unknown'),
        "seek_medical_attention": analysis.get('seek_medical_attention', False),
        "symptoms": symptoms,
        "duration": user_data.get('duration', 'Unknown'),
        "severity": user_data.get('severity', 'Unknown'),
        "app": "AI Health Advisor - Medical Report",
        "conditions": [c.get('name') for c in analysis.get('possible_conditions', [])][:3]
    }
    
    # Convert to JSON string for QR code
    qr_json = json.dumps(qr_data)
    
    # Generate QR code image
    qr_code_image = generate_qr_code(qr_json, size=150)
    
    # Render the template with all data including QR code
    rendered_html = template.render(
        report_date=format_date(),
        user_data=user_data,
        symptoms=symptoms,
        analysis=analysis,
        qr_code_image=qr_code_image,
        report_id=report_id
    )
    
    return rendered_html

def save_report_to_file(html_content, filename=None):
    """
    Save a generated HTML report to a file.
    
    Args:
        html_content (str): HTML report content
        filename (str, optional): Filename for the report
        
    Returns:
        str: Path to the saved file
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"symptom_report_{timestamp}.html"
    
    # Save the file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return filename

def generate_pdf_report(html_content, patient_name="Patient"):
    """
    Generate a PDF report from HTML content.
    
    Args:
        html_content (str): HTML content to convert to PDF
        patient_name (str): Name of the patient for the filename
        
    Returns:
        tuple: (pdf_bytes, filename) - The PDF as bytes and suggested filename
    """
    # Create sanitized patient name for filename
    safe_name = patient_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if safe_name == "Not_specified":
        safe_name = "Patient"
    
    # Create a date string for the filename
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"AI_Health_Advisor_{safe_name}_{date_str}.pdf"
    
    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_html:
        temp_html.write(html_content.encode('utf-8'))
        temp_html_path = temp_html.name
    
    try:
        # Convert HTML to PDF
        html = HTML(filename=temp_html_path)
        pdf_bytes = html.write_pdf()
        return pdf_bytes, filename
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_html_path):
            os.unlink(temp_html_path)