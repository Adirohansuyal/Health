import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import base64
import datetime
import json
from pathlib import Path

# Import UI helpers
from utils.ui_helpers import (local_css, display_header,
                              display_medical_disclaimer, display_risk_badge,
                              display_medical_attention_alert,
                              display_symptom_list, display_condition_card,
                              display_chat_message, display_footer)

# Import report generator
from utils.report_generator import generate_html_report, generate_pdf_report, generate_qr_code

# Try to use Gemini if API key is available, otherwise fall back to local analyzer
if os.environ.get("GOOGLE_API_KEY"):
    from utils.gemini_helper import analyze_symptoms, get_symptom_conversation
    USING_AI = True
else:
    from utils.symptom_analyzer import analyze_symptoms
    USING_AI = False

from data.symptoms import COMMON_SYMPTOMS, BODY_SYSTEMS, SYMPTOM_SEVERITY, SYMPTOM_DURATION


def main():
    # Set page configuration
    st.set_page_config(page_title="Birla AI Symptoms Checker",
                       page_icon="🏥",
                       layout="wide",
                       initial_sidebar_state="expanded")

    # Load custom CSS
    if Path("static/style.css").exists():
        local_css("static/style.css")

    # Main container for app content
    with st.container():
        # Display header
        display_header()

        # Medical disclaimer
        display_medical_disclaimer(expanded=True)

        # Sidebar for user information input with improved layout
        st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h2 style="color: #1E88E5;">Patient Information</h2>
            <p style="color: #666;">Please provide your details below</p>
        </div>
        """,
                            unsafe_allow_html=True)

        # Two-column layout in sidebar
        with st.sidebar:
            # User information form with improved design
            with st.form("user_info_form"):
                # User details section
                st.markdown(
                    '<div style="background-color: #F5F5F5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">',
                    unsafe_allow_html=True)

                # Patient name
                patient_name = st.text_input(
                    "Patient Name", placeholder="Enter your full name")
                # Store in session state for report
                st.session_state.patient_name = patient_name

                # Basic info in columns
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age",
                                          min_value=1,
                                          max_value=120,
                                          value=30)
                    # Store in session state for report
                    st.session_state.age = age
                with col2:
                    gender = st.selectbox("Gender",
                                          ["Male", "Female", "Other"])
                    # Store in session state for report
                    st.session_state.gender = gender

                # Image upload
                uploaded_image = st.file_uploader("Upload Patient Photo",
                                                  type=['png', 'jpg', 'jpeg'])
                if uploaded_image is not None:
                    # Display the uploaded image
                    st.image(uploaded_image, width=200)
                    # Convert image to base64 for storage
                    image_bytes = uploaded_image.getvalue()
                    image_b64 = base64.b64encode(image_bytes).decode()
                    st.session_state.patient_image = image_b64

                st.markdown('</div>', unsafe_allow_html=True)

                # Symptom selection
                st.markdown("""
                <div style="margin: 1rem 0;">
                    <h3 style="color: #1E88E5; margin-bottom: 0.5rem;">Symptoms</h3>
                    <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">Select all that apply</p>
                </div>
                """,
                            unsafe_allow_html=True)

                # Organize symptoms by body system
                system_tabs = st.tabs(list(BODY_SYSTEMS.keys()))

                # Create a master list to store all selected symptoms
                if 'all_selected_symptoms' not in st.session_state:
                    st.session_state.all_selected_symptoms = []

                # For each body system, display relevant symptoms
                for i, (system, symptoms) in enumerate(BODY_SYSTEMS.items()):
                    with system_tabs[i]:
                        # Select symptoms for this system
                        system_symptoms = st.multiselect(
                            f"Select {system.lower()} symptoms",
                            options=symptoms,
                            default=[])

                        # Update master list
                        for symptom in system_symptoms:
                            if symptom not in st.session_state.all_selected_symptoms:
                                st.session_state.all_selected_symptoms.append(
                                    symptom)

                # Display currently selected symptoms
                if st.session_state.all_selected_symptoms:
                    st.markdown("""
                    <div style="margin: 1rem 0;">
                        <h4 style="color: #1E88E5;">Currently Selected:</h4>
                    </div>
                    """,
                                unsafe_allow_html=True)

                    for symptom in st.session_state.all_selected_symptoms:
                        st.markdown(
                            f"<div style='padding: 0.3rem 0.8rem; background-color: #E3F2FD; border-radius: 20px; margin-bottom: 0.5rem; display: inline-block; margin-right: 0.5rem;'>{symptom}</div>",
                            unsafe_allow_html=True)

                # Custom symptoms input
                st.markdown("<div style='margin: 1rem 0;'>",
                            unsafe_allow_html=True)
                custom_symptoms = st.text_input(
                    "Add other symptoms (comma-separated)")
                st.markdown("</div>", unsafe_allow_html=True)

                # Additional details
                st.markdown("""
                <div style="margin: 1rem 0;">
                    <h3 style="color: #1E88E5; margin-bottom: 0.5rem;">Additional Details</h3>
                    <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">Help us understand your condition better</p>
                </div>
                """,
                            unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    duration = st.selectbox("Duration", SYMPTOM_DURATION)
                    # Store in session state for report
                    st.session_state.duration = duration
                with col2:
                    severity = st.selectbox("Severity", SYMPTOM_SEVERITY)
                    # Store in session state for report
                    st.session_state.severity = severity

                # Additional information
                additional_info = st.text_area(
                    "Medical History, Allergies, etc.", height=100)
                # Store in session state for report
                st.session_state.additional_info = additional_info

                # Submit button with improved styling
                st.markdown("<div style='margin-top: 2rem;'>",
                            unsafe_allow_html=True)
                submit_button = st.form_submit_button("Analyze Symptoms")
                st.markdown("</div>", unsafe_allow_html=True)

        # Process user input when the form is submitted
        if submit_button:
            # Validate input
            if not st.session_state.all_selected_symptoms and not custom_symptoms:
                st.error("Please select at least one symptom.")
                return

            # Process custom symptoms if provided
            all_symptoms = st.session_state.all_selected_symptoms.copy()
            if custom_symptoms:
                custom_symptoms_list = [
                    s.strip() for s in custom_symptoms.split(',') if s.strip()
                ]
                all_symptoms.extend(custom_symptoms_list)

            # Show processing message with progress
            progress_container = st.empty()
            progress_bar = st.progress(0)

            progress_container.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h3>Analyzing your symptoms...</h3>
                <p>Please wait while our AI processes your information</p>
            </div>
            """,
                                        unsafe_allow_html=True)

            # Simulate progress
            for i in range(101):
                progress_bar.progress(i)
                time.sleep(0.01)  # Small delay for visual effect

            # Call symptom analyzer
            analysis_result = analyze_symptoms(all_symptoms, age, gender,
                                               duration, severity,
                                               additional_info)

            # Clear the progress indicators
            progress_container.empty()
            progress_bar.empty()

            # Check for error
            if analysis_result.get("error", False):
                st.error(
                    analysis_result.get("message",
                                        "An error occurred during analysis."))
                return

            # Display results
            display_results(analysis_result, all_symptoms)

        # Information section about the app
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h3 style="color: #1E88E5;">About this App</h3>
        </div>
        """,
                            unsafe_allow_html=True)

        if USING_AI:
            app_description = """
            <div style="padding: 1rem; background-color: #E3F2FD; border-radius: 8px;">
                <p>This Symptom Checker uses <strong>Google's Gemini AI or a Backend Model</strong> to analyze your symptoms and 
                provide potential health conditions, advice, and diet recommendations.
                
                Remember that this tool does not replace professional medical advice.</p>
            </div>
            """
        else:
            app_description = """
            <div style="padding: 1rem; background-color: #F5F5F5; border-radius: 8px;">
                <p>This Symptom Checker uses a comprehensive symptom database to analyze your symptoms and 
                provide potential health conditions, advice, and diet recommendations.</p>
                
                <p style="margin-top: 0.5rem;"><em>Remember that this tool does not replace professional medical advice.</em></p>
            </div>
            """

        st.sidebar.markdown(app_description, unsafe_allow_html=True)
        
        # Meet the Developers section
        st.sidebar.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <h3 style="color: #1E88E5;">Meet the Developer</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("""
            <a href="https://www.linkedin.com/in/aditya-suyal/" target="_blank" 
               style="display: inline-block; padding: 0.5rem 1rem; 
                      background-color: #0077B5; color: white; 
                      text-decoration: none; border-radius: 4px;
                      text-align: center; width: 100%;
                      font-weight: bold; margin: 0.5rem 0;">
                Connect with Developer 🔗
            </a>
            """, 
            unsafe_allow_html=True)

        # Footer
        display_footer()


def display_results(analysis, symptoms):
    """Display the analysis results in a structured format."""
    # Create a container with animation for the results
    with st.container():
        st.markdown("""
        <div class="fade-in" style="animation-duration: 0.8s;">
            <h2 style="color: #1E88E5; margin-bottom: 1.5rem;">Analysis Results</h2>
        </div>
        """,
                    unsafe_allow_html=True)

        # Summary section with improved layout
        st.markdown("""
        <div style="background-color: #F8F9FA; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 5px solid #1E88E5;">
            <h3 style="margin-top: 0; color: #1E88E5;">Summary</h3>
            <p>Based on the symptoms you've reported, we've analyzed potential health conditions and provided recommendations.</p>
        </div>
        """,
                    unsafe_allow_html=True)

        # Main info in a card-like container
        col1, col2 = st.columns([2, 1])

        with col1:
            # Display risk level with better styling
            risk_level = analysis.get("risk_level", "unknown").lower()
            st.markdown("<h4>Risk Assessment:</h4>", unsafe_allow_html=True)
            display_risk_badge(risk_level)

            # Medical attention recommendation with improved styling
            seek_medical_attention = analysis.get("seek_medical_attention",
                                                  False)
            display_medical_attention_alert(seek_medical_attention)

        with col2:
            # Display reported symptoms with improved styling
            st.markdown("<h4>Your Reported Symptoms:</h4>",
                        unsafe_allow_html=True)

            # Create a styled card for symptoms
            st.markdown("""
            <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 8px; height: 100%;">
            """,
                        unsafe_allow_html=True)

            display_symptom_list(symptoms)

            st.markdown("</div>", unsafe_allow_html=True)

        # Possible conditions section with tabs and cards
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 style="color: #1E88E5; margin-bottom: 1rem;">Possible Conditions</h3>
        </div>
        """,
                    unsafe_allow_html=True)

        possible_conditions = analysis.get("possible_conditions", [])

        if not possible_conditions:
            st.info(
                "No specific conditions identified based on your symptoms. Please consult a healthcare professional for a proper diagnosis."
            )
        else:
            # Create tabs for each condition for better organization
            condition_tabs = st.tabs(
                [condition["name"] for condition in possible_conditions])

            for i, tab in enumerate(condition_tabs):
                condition = possible_conditions[i]
                with tab:
                    # Use our UI helper for a styled condition card
                    display_condition_card(condition)

        # General advice section with improved styling
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 style="color: #1E88E5; margin-bottom: 1rem;">General Health Advice</h3>
        </div>
        """,
                    unsafe_allow_html=True)

        # Styled advice box
        st.markdown(f"""
        <div style="background-color: #E3F2FD; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <p style="margin: 0;">{analysis.get("general_advice", "No specific advice available.")}</p>
        </div>
        """,
                    unsafe_allow_html=True)

        # Medical sources with improved styling
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <h3 style="color: #1E88E5; margin-bottom: 1rem;">Medical Sources</h3>
        </div>
        """,
                    unsafe_allow_html=True)

        sources = analysis.get("medical_sources", [])
        if sources:
            st.markdown(
                '<div style="background-color: #F8F9FA; padding: 1rem; border-radius: 8px;">',
                unsafe_allow_html=True)
            for source in sources:
                st.markdown(
                    f'<p style="margin-bottom: 0.5rem;"><a href="{source}" target="_blank">{source}</a></p>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No specific medical sources provided.")

        # AI Symptom Assistant with chat-like interface
        if USING_AI:
            st.markdown("""
            <div style="margin-top: 2rem;">
                <h3 style="color: #1E88E5; margin-bottom: 1rem;">💬 Symptom Assistant</h3>
                <p>Ask our AI assistant about your symptoms for more personalized advice:</p>
            </div>
            """,
                        unsafe_allow_html=True)

            # Initialize conversation state if it doesn't exist
            if 'conversation' not in st.session_state:
                st.session_state.conversation = []

            # Create a styled chat container
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            # Get a conversational response about the symptoms
            if len(symptoms) > 0:
                ai_response = get_symptom_conversation(
                    symptoms, st.session_state.conversation)

                # Display the messages using our helper functions
                display_chat_message(
                    f"I have these symptoms: {', '.join(symptoms)}",
                    is_user=True)
                display_chat_message(ai_response, is_user=False)

                # Store this interaction
                st.session_state.conversation.append({
                    "role":
                    "user",
                    "parts": [f"I have these symptoms: {', '.join(symptoms)}"]
                })
                st.session_state.conversation.append({
                    "role": "model",
                    "parts": [ai_response]
                })

                # Allow follow-up questions with improved styling
                st.markdown('<div style="margin-top: 1rem;">',
                            unsafe_allow_html=True)
                follow_up = st.text_input(
                    "Ask a follow-up question:",
                    placeholder="Type your health question here...")
                st.markdown('</div>', unsafe_allow_html=True)

                if follow_up:
                    with st.spinner("Getting response..."):
                        # Display the user message
                        display_chat_message(follow_up, is_user=True)

                        # Add this question to the conversation
                        st.session_state.conversation.append({
                            "role":
                            "user",
                            "parts": [follow_up]
                        })

                        # Get AI response
                        response = get_symptom_conversation(
                            symptoms, st.session_state.conversation)

                        # Display AI response
                        display_chat_message(response, is_user=False)

                        # Store response
                        st.session_state.conversation.append({
                            "role":
                            "model",
                            "parts": [response]
                        })

            st.markdown('</div>', unsafe_allow_html=True)

        # Store analysis in session state for report generation
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = {}

        # Store current analysis, symptoms, and user data for report generation
        st.session_state.current_analysis = {
            'analysis': analysis,
            'symptoms': symptoms,
            'user_data': {
                'patient_name':
                st.session_state.get('patient_name', 'Not specified'),
                'age':
                st.session_state.get('age', 30),
                'gender':
                st.session_state.get('gender', 'Not specified'),
                'duration':
                st.session_state.get('duration', 'Not specified'),
                'severity':
                st.session_state.get('severity', 'Not specified'),
                'additional_info':
                st.session_state.get('additional_info', ''),
                'patient_image':
                st.session_state.get('patient_image', None)
            }
        }

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            # Create tabs for different report formats
            report_tabs = st.tabs(["PDF Report", "HTML Report", "QR Code"])

            with report_tabs[0]:  # PDF Tab
                # Generate PDF data first
                pdf_data = generate_pdf_report(
                    generate_html_report(
                        st.session_state.current_analysis['analysis'],
                        st.session_state.current_analysis['symptoms'],
                        st.session_state.current_analysis['user_data']),
                    st.session_state.current_analysis['user_data'].get(
                        'patient_name', 'Patient'))[0]

                if st.download_button(
                        label="📄 Generate PDF Report",
                        key="generate_pdf_report",
                        data=pdf_data,
                        file_name=
                        f"AI_Health_Advisor_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"):
                    st.success("PDF Report downloaded successfully!")

            with report_tabs[1]:  # HTML Tab
                if st.button("🌐 Generate HTML Report",
                             key="generate_html_report"):
                    # Create HTML report
                    user_data = st.session_state.current_analysis['user_data']
                    html_report = generate_html_report(analysis, symptoms,
                                                       user_data)

                    # Create a download link for the report
                    b64_html = base64.b64encode(html_report.encode()).decode()
                    href = f'data:text/html;base64,{b64_html}'

                    # Get patient name for the filename (default if not provided)
                    patient_name = user_data.get('patient_name',
                                                 'Patient').replace(" ", "_")
                    if patient_name == 'Not_specified':
                        patient_name = 'Patient'

                    # Create a date string for the filename
                    date_str = datetime.datetime.now().strftime("%Y%m%d")
                    filename = f"AI_Health_Advisor_{patient_name}_{date_str}.html"

                    st.success("HTML Report generated successfully!")
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0; padding: 1.5rem; background-color: #E3F2FD; border-radius: 8px;">
                        <h3 style="margin-top: 0;">HTML Report Ready</h3>
                        <p>You can now download and print your interactive symptom analysis report.</p>
                        <a href="{href}" download="{filename}" 
                           style="display: inline-block; background-color: #1E88E5; color: white; padding: 0.8rem 1.5rem; 
                                  text-decoration: none; border-radius: 4px; font-weight: bold; margin-top: 1rem;">
                            Download HTML Report
                        </a>
                        <p style="margin-top: 1rem; font-size: 0.9rem;">
                            <i>The report will open in a new tab where you can review and print it.</i>
                        </p>
                    </div>
                    """,
                                unsafe_allow_html=True)

            with report_tabs[2]:  # QR Code Tab
                st.markdown("""
                <div style="margin-bottom: 1rem;">
                    <h4 style="color: #1E88E5;">Scan for Doctor Reference</h4>
                    <p>The QR code contains your symptom information for quick reference by healthcare providers.</p>
                </div>
                """,
                            unsafe_allow_html=True)

                # Create data for QR code
                user_data = st.session_state.current_analysis['user_data']

                # Create a comprehensive data structure for the QR code
                qr_data = {
                    "patient":
                    user_data.get('patient_name', 'Unknown'),
                    "age":
                    user_data.get('age', 'Unknown'),
                    "gender":
                    user_data.get('gender', 'Unknown'),
                    "symptoms":
                    symptoms,
                    "timestamp":
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "risk_level":
                    analysis.get('risk_level', 'Unknown'),
                    "seek_medical_attention":
                    analysis.get('seek_medical_attention', False),
                    "duration":
                    user_data.get('duration', 'Unknown'),
                    "severity":
                    user_data.get('severity', 'Unknown'),
                    "app":
                    "AI Health Advisor"
                }

                # Convert to JSON string
                qr_json = json.dumps(qr_data)

                # Generate QR code
                qr_code_image = generate_qr_code(qr_json, size=300)

                # Display the QR code with instructions
                st.markdown("""
                <div style="text-align: center; margin: 1rem 0;">
                    <h3 style="color: #1E88E5;">QR Code Ready</h3>
                    <p>Show this to your healthcare provider for quick access to your symptom details.</p>
                </div>
                """,
                            unsafe_allow_html=True)

                # Display the QR code
                st.markdown(
                    f'<div style="display: flex; justify-content: center;"><img src="{qr_code_image}" style="width: 300px; height: 300px;"></div>',
                    unsafe_allow_html=True)

                # Display a caption
                st.markdown("""
                <div style="text-align: center; margin-top: 1rem;">
                    <p style="color: #666; font-size: 0.9rem;">
                        <i>Your healthcare provider can scan this code to quickly access your symptom details.</i>
                    </p>
                </div>
                """,
                            unsafe_allow_html=True)

                # Option to save the QR code
                st.markdown(f"""
                <div style="text-align: center; margin-top: 1rem;">
                    <a href="{qr_code_image}" download="symptom_qr_code.png" 
                       style="display: inline-block; background-color: #1E88E5; color: white; padding: 0.5rem 1rem; 
                              text-decoration: none; border-radius: 4px; font-weight: bold;">
                        Save QR Code
                    </a>
                </div>
                """,
                            unsafe_allow_html=True)

        with col2:
            if st.button("🔍 Health Resources", key="health_resources"):
                st.markdown("""
                <div style="padding: 1rem; background-color: #F5F5F5; border-radius: 8px; margin-top: 1rem;">
                    <h4 style="color: #1E88E5;">Helpful Health Resources</h4>
                    <ul>
                        <li><a href="https://www.who.int" target="_blank">World Health Organization</a></li>
                        <li><a href="https://www.mayoclinic.org" target="_blank">Mayo Clinic</a></li>
                        <li><a href="https://www.cdc.gov" target="_blank">Centers for Disease Control and Prevention</a></li>
                        <li><a href="https://medlineplus.gov" target="_blank">MedlinePlus</a></li>
                    </ul>
                </div>
                """,
                            unsafe_allow_html=True)

        # Medical disclaimer reminder with improved styling
        st.markdown("""
        <div style="margin-top: 2rem; background-color: #FFF8E1; padding: 1rem; border-radius: 8px; border-left: 5px solid #FFA000;">
            <h4 style="margin-top: 0; color: #FFA000;">⚠️ Medical Disclaimer</h4>
            <p><strong>Remember:</strong> This analysis is based on the symptoms you provided and is not a medical diagnosis.
            The information provided should not replace professional medical advice.
            If you're concerned about your health, please consult a healthcare professional.</p>
        </div>
        """,
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
