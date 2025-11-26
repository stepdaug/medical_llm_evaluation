# Google API service account email: "streamlit-validator-writer@llm-validator.iam.gserviceaccount.com"
# pip install streamlit pandas gspread gspread-dataframe oauth2client
# activate streamlit && cd /d D:/projects/medical_llm_validator/human_validation_of_evaluator && python -m streamlit run "human_assessment_app.py"

import streamlit as st
from pathlib import Path
import json
import os
import pandas as pd
import gspread
from datetime import datetime
import ast
import uuid

# --- CONFIGURATION ---
# BASE_PATH = Path(r"D:\projects\medical_llm_validator\cases") # full data from harddrive
# PATH TO CAES FOR GITHUB UPLOAD
BASE_PATH = Path(__file__).parent / "cases_github"

REVIEWERS = ["", "JV", "MH","test1","test2", "SA"] 
SECTIONS = {
    "Localisation": "localisation",
    "Differential diagnosis": "differential_diagnosis",
    "Investigations": "investigations",
    "Management": "management"
}

MODEL_OPTIONS = {
    "GPT-5.1 (OpenAI)": "openai_gpt-5-1",
    "Gemini 3 Pro (Google)": "google_gemini-3-pro-preview",
    "GPT-5 (OpenAI)": "openai"
}

# --- STATE MANAGEMENT ---
# We use a unique ID for the form. Every time we switch cases, we change this ID.
# This forces Streamlit to generate BRAND NEW widgets with empty states.
if "form_uuid" not in st.session_state:
    st.session_state.form_uuid = str(uuid.uuid4())

def trigger_new_form_state():
    """Generates a new UUID to force all widgets to reset completely."""
    st.session_state.form_uuid = str(uuid.uuid4())

# --- NAVIGATION HANDLER ---
# This runs at the top. If a pending update exists, we apply it.
if "pending_case_update" in st.session_state:
    # 1. Update the selector widget
    st.session_state.sb_case_selector = st.session_state.pending_case_update
    
    # 2. Force a new form UUID so the next page load is blank
    trigger_new_form_state()
    
    # 3. Clean up
    del st.session_state.pending_case_update

# --- GOOGLE SHEETS CONNECTION ---
def init_gspread_connection():
    try:
        creds = st.secrets["gcp_service_account"]
        sa = gspread.service_account_from_dict(creds)
        return sa
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets. Check your secrets.toml file. Error: {e}")
        return None

@st.cache_data(ttl=300)
def get_completion_status():
    sa = init_gspread_connection()
    if not sa:
        return pd.DataFrame()
    
    try:
        sheet_url = st.secrets["gcp_service_account"]["g_sheet_url"]
        spreadsheet = sa.open_by_url(sheet_url)
        worksheet = spreadsheet.sheet1
        records = worksheet.get_all_records()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
    except Exception as e:
        st.warning(f"Could not read completion status from Google Sheet. Error: {e}")
        return pd.DataFrame()

def write_to_gsheet(data: dict):
    sa = init_gspread_connection()
    if not sa:
        return False
    
    try:
        sheet_url = st.secrets["gcp_service_account"]["g_sheet_url"]
        spreadsheet = sa.open_by_url(sheet_url)
        worksheet = spreadsheet.sheet1
        
        header = [
            "timestamp", "reviewer_initials", "case_number", "provider",
            "localisation_feedback", "localisation_comment",
            "ddx_feedback", "ddx_comment",
            "investigations_feedback", "investigations_comment",
            "management_feedback", "management_comment",
            "case_feasibility", "case_feasibility_comment",
            "other_inaccuracies", "other_inaccuracies_comment"
        ]
        if not worksheet.get_all_values():
            worksheet.append_row(header)
        row_to_append = [data.get(h, "") for h in header]
        worksheet.append_row(row_to_append)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Failed to write to Google Sheet. Error: {e}")
        return False

# --- DATA LOADING FUNCTIONS ---
def get_available_cases(base_path: Path) -> list:
    cases = [d.name.split('_')[1] for d in base_path.iterdir() if d.is_dir() and d.name.startswith("case_")]
    return sorted(cases)

@st.cache_data(show_spinner="Loading case data...")
def load_case_data(case_number: str, model_id: str) -> dict:
    case_path = BASE_PATH / f"case_{case_number}"
    data = {"valid": True, "errors": []}

    # 1. Load Image
    image_path = case_path / "combined_examination_summary.png"
    if image_path.exists():
        with open(image_path, 'rb') as f:
            data["image"] = f.read()
    else:
        data["errors"].append("Image file not found.")
        data["valid"] = False

    # 2. Determine Filenames based on Model ID
    if model_id == "openai":
        file_prefix = "openai"
        report_filename = "validation_report_openai.json"
    else:
        file_prefix = model_id 
        report_filename = f"validation_report_{model_id}.json"

    # 3. Load LLM Answers
    data["llm_answers"] = {}
    for display_name, section_key in SECTIONS.items():
        answer_file = case_path / f"q_{file_prefix}_{section_key}.json"
        if answer_file.exists():
            try:
                with open(answer_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    data["llm_answers"][display_name] = content.get("response", "Error: 'response' key not found.")
            except json.JSONDecodeError:
                data["llm_answers"][display_name] = f"Error: JSON Decode Error in {answer_file.name}."
        else:
            data["llm_answers"][display_name] = f"File not found: {answer_file.name}"
    
    # 4. Load Validation Report
    report_file = case_path / report_filename
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            data["report"] = json.load(f)
    else:
        data["errors"].append(f"Report file not found: {report_filename}")
        data["valid"] = False

    return data

# --- MAIN APP LAYOUT ---
st.set_page_config(layout="wide", page_title="Clinical Case Evaluator Review")
available_cases_display = [] 

def get_reviewer_assignment(reviewer, completion_df):
    all_cases = get_available_cases(BASE_PATH)
    id_map = {int(c): c for c in all_cases if c.isdigit()}
    valid_cases = []
    case_model_map = {}

    m_gpt = "openai_gpt-5-1"
    m_gem = "google_gemini-3-pro-preview"

    def add_case(i, force_model=None):
        if i in id_map:
            c_str = id_map[i]
            model = force_model if force_model else (m_gpt if i % 2 != 0 else m_gem)
            if c_str not in valid_cases: 
                valid_cases.append(c_str)
            case_model_map[c_str] = model

    if reviewer in ["JV", "MH","test1","test2"]:
        for i in range(1, 11): add_case(i)
        if reviewer == "JV" or reviewer == "test1":
            for i in range(11, 41): add_case(i)
        elif reviewer == "MH" or reviewer == "test2":
            for i in range(41, 71): add_case(i)
    
    return sorted(valid_cases, key=lambda x: int(x)), case_model_map

# --- SIDEBAR FOR SELECTIONS ---
with st.sidebar:
    st.header("Reviewer Details")
    reviewer_initials = st.selectbox("Select Your Initials", options=REVIEWERS, index=0)
    
    case_number = None
    provider = None 
    
    if reviewer_initials:
        completion_df = get_completion_status()
        completed_cases = set()

        if not completion_df.empty:
            completion_df['case_number'] = completion_df['case_number'].astype(str).str.strip().str.zfill(4)
            if reviewer_initials == "SA":
                user_completed = completion_df[completion_df['reviewer_initials'] == "SA"]
            else:
                user_completed = completion_df[completion_df['reviewer_initials'] == reviewer_initials]
            
            completed_cases = set(user_completed['case_number'])

        if reviewer_initials == "SA":
            available_cases_display = get_available_cases(BASE_PATH)
        else:
            assigned_cases, case_model_map = get_reviewer_assignment(reviewer_initials, completion_df)
            available_cases_display = assigned_cases

        def format_case_number(case_num):
            if case_num in completed_cases:
                return f"{case_num}  ✅"
            else:
                return f"{case_num}  🔴"
        
        # When manual selection changes, trigger a new form UUID
        case_number_formatted = st.selectbox(
            "Select Case Number", 
            options=available_cases_display, 
            format_func=format_case_number,
            on_change=trigger_new_form_state, # THIS ENSURES BLANK SLATE ON MANUAL CHANGE
            key="sb_case_selector"
        )
        case_number = case_number_formatted.split(" ")[0] if case_number_formatted else None

        if reviewer_initials == "SA":
            model_display_name = st.selectbox(
                "Select Model (Admin View)",
                options=list(MODEL_OPTIONS.keys()),
                on_change=trigger_new_form_state
            )
            provider = MODEL_OPTIONS[model_display_name]
        elif case_number:
            provider = case_model_map.get(case_number)

    else:
        st.info("Select initials to begin.")
        
    st.header("Display Options")
    image_size_multiplier = st.slider("Adjust Image Size", 0.2, 1.0, 0.6, 0.05)

# --- MAIN CONTENT AREA ---
if case_number and reviewer_initials and provider:
    case_data = load_case_data(case_number, provider)

    if not case_data["valid"]:
        st.error("Could not load all necessary files.")
        for error in case_data["errors"]:
            st.warning(f"- {error}")
    else:
        st.image(case_data["image"], width=int(1000 * image_size_multiplier))

        st.subheader("Evaluation Sections:")
        tabs = st.tabs(list(SECTIONS.keys()))
        
        # We retrieve the current unique ID for this case view
        uid = st.session_state.form_uuid

        for i, (display_name, file_key) in enumerate(SECTIONS.items()):
            with tabs[i]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"LLM's Answer for {display_name}")
                    raw_llm_answer = case_data["llm_answers"].get(display_name, "")

                    LLM_VAR_NAMES = {
                        "Localisation": "lesion_locations",
                        "Differential diagnosis": "differential_diagnosis",
                        "Investigations": "investigations",
                        "Management": "treatments"
                    }

                    prose_part = raw_llm_answer
                    parsed_list = None
                    var_name = LLM_VAR_NAMES.get(display_name)

                    if var_name and f"{var_name} = " in raw_llm_answer:
                        parts = raw_llm_answer.split(f"{var_name} = ", 1)
                        prose_part = parts[0].rstrip()
                        try:
                            parsed_data = ast.literal_eval(parts[1].strip())
                            if isinstance(parsed_data, list):
                                parsed_list = parsed_data
                        except:
                            prose_part = raw_llm_answer

                    if display_name == "Management":
                        prose_part = prose_part.replace("*   **Treatment:**", "**Treatment:**")

                    st.markdown(prose_part, unsafe_allow_html=True)

                    if parsed_list:
                         if display_name != "Management":
                            st.markdown("##### Final answer:")
                            st.markdown("\n".join([f"- {item}" for item in parsed_list]))
                
                with col2:
                    st.subheader(f"Evaluation of LLM's Answer")
                    report_string = case_data["report"]["marking_output"].get(display_name, {}).get("report_string", "Evaluation not found.")
                    st.markdown(report_string.replace('---', '####'), unsafe_allow_html=True)
                    
                    st.divider()
                    st.subheader(f"Your Feedback")
                    
                    # DYNAMIC KEYS: We append the UID to ensure these are unique per case load
                    key_feedback = f"{uid}_{file_key}_feedback"
                    key_comment = f"{uid}_{file_key}_comment"
                    
                    st.radio(
                        f"Do you agree with the statements in the Evaluation of Answer for {display_name}?",
                        ("Statements are accurate", "Some statements are inaccurate"),
                        key=key_feedback, index=None, horizontal=True
                    )
                    
                    # Read current state directly using the dynamic key
                    if st.session_state.get(key_feedback) == "Some statements are inaccurate":
                        st.text_area(
                            "For each inaccurate statement, provide a concise explanation of the error:",
                            key=key_comment
                        )

        st.subheader("General Questions on the Case")
        
        st.radio(
            "Is this case feasibly reflective of a patient with deficits related to MS (either recent relapse or chronic)?",
            ("Yes", "No"), key=f"{uid}_case_feasibility", index=None, horizontal=True
        )
        
        if st.session_state.get(f"{uid}_case_feasibility") == "No":
            st.text_area("Briefly explain how the case is not feasible:", key=f"{uid}_case_feasibility_comment")
        
        st.radio(
            "Did you identify any other inaccuracies in the LLM's answers not highlighted by the automated evaluation?",
            ("Yes", "No"), key=f"{uid}_other_inaccuracies", index=None, horizontal=True
        )
        if st.session_state.get(f"{uid}_other_inaccuracies") == "Yes":
            st.text_area("Please describe the other inaccuracies you identified:", key=f"{uid}_other_inaccuracies_comment")

        # --- SUBMISSION LOGIC ---
        if st.button("Submit All Completed Feedback for this Case", type="primary"):
            validation_errors = []
            
            # Helper to get value for current form ID
            def get_val(suffix):
                return st.session_state.get(f"{uid}_{suffix}")

            # 1. Validation Checks
            required_suffixes = [f"{fk}_feedback" for fk in SECTIONS.values()] + ["case_feasibility", "other_inaccuracies"]
            if not all(get_val(s) is not None for s in required_suffixes):
                validation_errors.append("Please answer all Yes/No or Agree/Disagree questions.")

            for display_name, file_key in SECTIONS.items():
                if get_val(f"{file_key}_feedback") == "Some statements are inaccurate": 
                    if not str(get_val(f"{file_key}_comment")).strip():
                        validation_errors.append(f"Please provide a comment for '{display_name}'.")

            if get_val("case_feasibility") == "No":
                if not str(get_val("case_feasibility_comment")).strip():
                    validation_errors.append("Please explain why the case is not feasible.")
            
            if get_val("other_inaccuracies") == "Yes":
                if not str(get_val("other_inaccuracies_comment")).strip():
                    validation_errors.append("Please describe the other inaccuracies.")

            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                with st.spinner("Submitting your feedback..."):
                    submission_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "reviewer_initials": reviewer_initials,
                        "case_number": case_number,
                        "provider": provider,
                        "localisation_feedback": get_val("localisation_feedback"),
                        "localisation_comment": get_val("localisation_comment"),
                        "ddx_feedback": get_val("differential_diagnosis_feedback"),
                        "ddx_comment": get_val("differential_diagnosis_comment"),
                        "investigations_feedback": get_val("investigations_feedback"),
                        "investigations_comment": get_val("investigations_comment"),
                        "management_feedback": get_val("management_feedback"),
                        "management_comment": get_val("management_comment"),
                        "case_feasibility": get_val("case_feasibility"),
                        "case_feasibility_comment": get_val("case_feasibility_comment"),
                        "other_inaccuracies": get_val("other_inaccuracies"),
                        "other_inaccuracies_comment": get_val("other_inaccuracies_comment")
                    }
                    if write_to_gsheet(submission_data):
                        st.success("Feedback submitted successfully! Thank you.")
                        
                        # --- AUTO-ADVANCE LOGIC ---
                        try:
                            current_idx = available_cases_display.index(case_number)
                            next_idx = current_idx + 1
                            
                            if next_idx < len(available_cases_display):
                                next_case_val = available_cases_display[next_idx]
                                if next_case_val in available_cases_display:
                                    # Set Pending Update for Top of Script
                                    st.session_state.pending_case_update = next_case_val
                                    st.rerun()
                                else:
                                    st.warning(f"Could not auto-advance: '{next_case_val}' not found.")
                            else:
                                st.balloons()
                                st.info("You have reached the end of your assigned cases!")
                        except ValueError:
                            st.warning("Please select the next case manually.")
                        except Exception as e:
                            st.warning(f"Auto-advance error: {e}")
else:
    st.info("Please select your initials and a case from the sidebar to begin.")
