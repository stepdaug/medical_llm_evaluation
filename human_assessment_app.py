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

# --- CONFIGURATION ---
# BASE_PATH = Path(r"D:\projects\medical_llm_validator\cases") # full data from harddrive
# PATH TO CAES FOR GITHUB UPLOAD
script_dir = Path(__file__).parent 
BASE_PATH = script_dir / "cases_github"

REVIEWERS = ["", "JV", "MH", "SA"] # Add initials of your reviewers. "" is for the placeholder.
SECTIONS = {
    "Localisation": "localisation",
    "Differential diagnosis": "differential_diagnosis",
    "Investigations": "investigations",
    "Management": "management"
}

# --- GOOGLE SHEETS CONNECTION ---
def init_gspread_connection():
    """Initialises a connection to Google Sheets using Streamlit's secrets."""
    try:
        creds = st.secrets["gcp_service_account"]
        sa = gspread.service_account_from_dict(creds)
        return sa
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets. Check your secrets.toml file. Error: {e}")
        return None

@st.cache_data(ttl=300) # Cache for 5 minutes to avoid hitting the API too often
def get_completion_status():
    """Reads the Google Sheet to determine which cases have been completed by reviewers."""
    sa = init_gspread_connection()
    if not sa:
        return pd.DataFrame() # Return empty DataFrame on connection failure
    
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
    """Appends a dictionary of data as a new row to the configured Google Sheet."""
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
        st.cache_data.clear() # clear the cache after writing new data
        return True
    except Exception as e:
        st.error(f"Failed to write to Google Sheet. Error: {e}")
        return False

def reset_feedback_state():
    """Clears all feedback-related keys from session_state."""
    # This is the primary dictionary holding feedback
    if 'feedback' in st.session_state:
        st.session_state.feedback = {}
    
    # Also, explicitly delete widget keys to be certain they reset
    keys_to_delete = []
    for key in st.session_state.keys():
        if key.endswith('_feedback') or key.endswith('_comment') or key.startswith('case_'):
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        del st.session_state[key]

# --- DATA LOADING FUNCTIONS ---
def get_available_cases(base_path: Path) -> list:
    """Scans the base path for available case folders."""
    cases = [d.name.split('_')[1] for d in base_path.iterdir() if d.is_dir() and d.name.startswith("case_")]
    return sorted(cases)

@st.cache_data(show_spinner="Loading case data...")
def load_case_data(case_number: str, provider: str) -> dict:
    """Loads all necessary data for a given case and provider."""
    case_path = BASE_PATH / f"case_{case_number}"
    data = {"valid": True, "errors": []}

    # Load image (FIXED: Load as bytes, not a path string)
    image_path = case_path / "combined_examination_summary.png"
    if image_path.exists():
        with open(image_path, 'rb') as f:
            data["image"] = f.read()
    else:
        data["errors"].append("Image file not found.")
        data["valid"] = False

    # Load LLM answers
    data["llm_answers"] = {}
    for display_name, file_key in SECTIONS.items():
        answer_file = case_path / f"q_{provider}_{file_key}.json"
        if answer_file.exists():
            try:
                with open(answer_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    data["llm_answers"][display_name] = content.get("response", "Error: 'response' key not found.")
            except json.JSONDecodeError:
                data["llm_answers"][display_name] = f"Error: Could not decode JSON from {answer_file.name}."
        else:
            data["llm_answers"][display_name] = "Answer file not found."
    
    # Load validation report
    report_file = case_path / f"validation_report_{provider}.json"
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            data["report"] = json.load(f)
    else:
        data["errors"].append(f"validation_report_{provider}.json not found.")
        data["valid"] = False

    return data

# --- MAIN APP LAYOUT ---
st.set_page_config(layout="wide", page_title="Clinical Case Evaluator Review")
# st.title("Human Validation of Automated Evaluator")

# --- SIDEBAR FOR SELECTIONS ---
with st.sidebar:
    st.header("Reviewer Details")
    reviewer_initials = st.selectbox("Select Your Initials", options=REVIEWERS, index=0)
    
    st.header("Case Selection")
    available_cases = get_available_cases(BASE_PATH)
    case_number = None
    if reviewer_initials:
        completion_df = get_completion_status()
        completed_cases = set()

        if not completion_df.empty:
            # Ensure case_number is treated as a string for consistent comparison
            completion_df['case_number'] = completion_df['case_number'].astype(str)
            user_completed = completion_df[completion_df['reviewer_initials'] == reviewer_initials]
            completed_cases = set(user_completed['case_number'])

        def format_case_number(case_num):
            """Formats the case number with a color emoji based on completion status."""
            if case_num in completed_cases:
                return f"{case_num}  ✅"  # Green for completed
            else:
                return f"{case_num}  🔴"  # Red for incomplete
        
        case_number_formatted = st.selectbox(
            "Select Case Number", 
            options=available_cases,
            format_func=format_case_number,
            on_change=reset_feedback_state
        )
        # We still need the raw case number for file loading
        case_number = case_number_formatted.split(" ")[0] if case_number_formatted else None

    else:
        # Default display if no reviewer is selected
        case_number = st.selectbox("Select Case Number", options=available_cases)


    provider = st.selectbox("Select Model",
                            options=["openai", "goog"],
                            on_change=reset_feedback_state
                            )

    st.header("Display Options")
    image_size_multiplier = st.slider(
        "Adjust Image Size", 
        min_value=0.2,  # 30% of max size
        max_value=1.0,  # 100% of max size
        value=0.6,      # Default value
        step=0.05
    )

# --- MAIN CONTENT AREA ---
if case_number and reviewer_initials:
    case_data = load_case_data(case_number, provider)

    if not case_data["valid"]:
        st.error("Could not load all necessary files for this case. Please check the folder structure and file names.")
        for error in case_data["errors"]:
            st.warning(f"- {error}")
    else:
        demographics = case_data["report"].get("patient_demographics", {})
        # st.header(f"Case {case_number} | {demographics.get('age', 'N/A')}y {demographics.get('sex', 'N/A')}")
        # We set a reasonable base width (e.g., 1000px) and apply the multiplier to it.
        # This gives direct control over the size.
        st.image(
            case_data["image"], 
            # caption="Clinical Examination Summary", 
            width=int(1000 * image_size_multiplier) # Remove use_column_width and set width
        )

        # st.divider()
        st.subheader("Evaluation Sections:")

        tab_titles = list(SECTIONS.keys())
        tabs = st.tabs(tab_titles)
        
        if 'feedback' not in st.session_state:
            st.session_state.feedback = {}

        for i, (display_name, file_key) in enumerate(SECTIONS.items()):
            with tabs[i]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"LLM's Answer for {display_name}")
                    raw_llm_answer = case_data["llm_answers"].get(display_name, "")

                    # 1. Map section names to the variable names in the LLM's output string
                    LLM_VAR_NAMES = {
                        "Localisation": "lesion_locations",
                        "Differential diagnosis": "differential_diagnosis",
                        "Investigations": "investigations",
                        "Management": "treatments"
                    }

                    prose_part = raw_llm_answer
                    parsed_list = None
                    var_name = LLM_VAR_NAMES.get(display_name)

                    # 2. Try to split the prose from the list and parse the list
                    if var_name:
                        search_string = f"{var_name} = "
                        if search_string in raw_llm_answer:
                            parts = raw_llm_answer.split(search_string, 1)
                            prose_part = parts[0].rstrip()
                            list_string = parts[1].strip()
                            
                            try:
                                # Use ast.literal_eval for safely parsing Python literals
                                parsed_data = ast.literal_eval(list_string)
                                if isinstance(parsed_data, list):
                                    parsed_list = parsed_data
                            except (ValueError, SyntaxError):
                                # If parsing fails, we'll just show the raw text.
                                # The prose_part remains the full original answer.
                                st.warning(f"Could not parse the data list for '{display_name}'. It might be malformed.")
                                prose_part = raw_llm_answer

                    # 3. Apply the specific formatting tweak for Management prose
                    if display_name == "Management":
                        prose_part = prose_part.replace("*   **Treatment:**", "**Treatment:**")

                    # 4. Display the prose part of the answer
                    st.markdown(prose_part, unsafe_allow_html=True)

                    # 5. If we successfully parsed a list, display it below
                    if parsed_list:
                        # st.divider()
                        
                        
                        # Handle the complex structure for Management
                        if display_name == "Management":
                            pass
                            # for item in parsed_list:
                                # st.markdown(f"**{item.get('treatment', 'N/A')}**")
                                # with st.container(border=True):
                                #     st.markdown(f"**Timing:** {item.get('timing', 'N/A')}")
                                #     st.markdown("**Reasons:**")
                                #     for reason in item.get('reasons', []):
                                #         st.markdown(f"- {reason}")
                        
                        # Handle the simple list of strings for other sections
                        else:
                            st.markdown("##### Final answer:")
                            md_list = "\n".join([f"- {item}" for item in parsed_list])
                            st.markdown(md_list)
                
                with col2:
                    st.subheader(f"Evaluation of LLM's Answer")
                    report_string = case_data["report"]["marking_output"].get(display_name, {}).get("report_string", "Evaluation not found.")
                    st.markdown(report_string.replace('---', '####'), unsafe_allow_html=True)
                    
                    st.divider()
                    st.subheader(f"Your Feedback")
                    
                    key_feedback = f"{file_key}_feedback"
                    key_comment = f"{file_key}_comment"
                    
                    st.session_state.feedback[key_feedback] = st.radio(
                        f"Do you agree with the statements in the Evaluation of Answer for {display_name}?",
                        ("Statements are accurate", "Some statements are inaccurate"),
                        key=key_feedback, index=None, horizontal=True
                    )
                    
                    if st.session_state.feedback[key_feedback] == "Some statements are inaccurate":
                        st.session_state.feedback[key_comment] = st.text_area(
                            "For each inaccurate statement, provide a concise explanation of the error:",
                            key=key_comment
                        )
                    else:
                        st.session_state.feedback[key_comment] = ""

        # General Questions
        # st.divider()
        st.subheader("General Questions on the Case")
        
        st.session_state.feedback["case_feasibility"] = st.radio(
            "Is this case feasibly reflective of a patient with MS (either recent relapse or chronic)?",
            ("Yes", "No"), key="case_feasibility", index=None, horizontal=True
        )
        # Conditional text area based on the radio button's state
        if st.session_state.feedback["case_feasibility"] == "No":
            st.session_state.feedback["case_feasibility_comment"] = st.text_area(
                "Briefly explain how the case is not feasible:",
                key="case_feasibility_comment"
            )
        else:
            # Ensure the comment is cleared if the user switches back to "Yes"
            st.session_state.feedback["case_feasibility_comment"] = ""
        
        st.session_state.feedback["other_inaccuracies"] = st.radio(
            "Did you identify any other inaccuracies in the LLM's answers not highlighted by the automated evaluation?",
            ("Yes", "No"), key="other_inaccuracies", index=None, horizontal=True
        )
        if st.session_state.feedback["other_inaccuracies"] == "Yes":
            st.session_state.feedback["other_inaccuracies_comment"] = st.text_area(
                "Please describe the other inaccuracies you identified:",
                key="other_inaccuracies_comment"
            )
        else:
            st.session_state.feedback["other_inaccuracies_comment"] = ""

        # st.divider()
        
        # --- SUBMISSION LOGIC ---
        if st.button("Submit All Completed Feedback for this Case", type="primary"):
            validation_errors = []
            
            # If the user marks the case as not feasible, we only require the comment for that field.
            # This allows them to submit a "bad case" without filling everything else out.
            # is_case_infeasible = st.session_state.feedback.get("case_feasibility") == "No"
            
            # if is_case_infeasible: # In this special case, the ONLY requirement is the feasibility comment.
            #     if not st.session_state.feedback.get("case_feasibility_comment", "").strip():
            #         validation_errors.append("Please explain why the case is not feasible since you selected 'No'.")
            # else:
            # Run all the checks to ensure all questions have been answered.
            # 1. Check that all radio buttons have a selection
            required_radio_fields = [f"{fk}_feedback" for fk in SECTIONS.values()] + ["case_feasibility", "other_inaccuracies"]
            if not all(st.session_state.feedback.get(field) is not None for field in required_radio_fields):
                validation_errors.append("Please answer all Yes/No or Agree/Disagree questions.")

            # 2. Check that conditional text boxes are filled if they are visible
            for display_name, file_key in SECTIONS.items():
                feedback_key = f"{file_key}_feedback"
                comment_key = f"{file_key}_comment"
                # NOTE: I updated the string to match the options in your radio button
                if st.session_state.feedback.get(feedback_key) == "Some statements are inaccurate": 
                    if not st.session_state.feedback.get(comment_key, "").strip():
                        validation_errors.append(f"Please provide a comment for '{display_name}' since you selected 'Some statements are inaccurate'.")

            # This check is redundant if it's handled by the `is_case_infeasible` block above, but leaving it here for the `else` case doesn't hurt.
            if st.session_state.feedback.get("case_feasibility") == "No":
                if not st.session_state.feedback.get("case_feasibility_comment", "").strip():
                    validation_errors.append("Please explain why the case is not feasible since you selected 'No' for 'Is this case feasibly reflective of a patient with MS (either recent relapse or chronic)?'.")
            
            if st.session_state.feedback.get("other_inaccuracies") == "Yes":
                if not st.session_state.feedback.get("other_inaccuracies_comment", "").strip():
                    validation_errors.append("Please describe the other inaccuracies you identified since you selected 'Yes' for 'Did you identify any other inaccuracies in the LLM's answers not highlighted by the automated evaluation?'.")

            # 3. If there are any errors, display them. Otherwise, submit.
            # This part remains UNCHANGED because the submission dictionary already uses .get()
            # which gracefully handles missing values by substituting None (or "" in the gsheet function).
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                with st.spinner("Submitting your feedback..."):
                    # Prepare data for submission
                    submission_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "reviewer_initials": reviewer_initials,
                        "case_number": case_number,
                        "provider": provider,
                        "localisation_feedback": st.session_state.feedback.get("localisation_feedback"),
                        "localisation_comment": st.session_state.feedback.get("localisation_comment"),
                        "ddx_feedback": st.session_state.feedback.get("differential_diagnosis_feedback"),
                        "ddx_comment": st.session_state.feedback.get("differential_diagnosis_comment"),
                        "investigations_feedback": st.session_state.feedback.get("investigations_feedback"),
                        "investigations_comment": st.session_state.feedback.get("investigations_comment"),
                        "management_feedback": st.session_state.feedback.get("management_feedback"),
                        "management_comment": st.session_state.feedback.get("management_comment"),
                        "case_feasibility": st.session_state.feedback.get("case_feasibility"),
                        "case_feasibility_comment": st.session_state.feedback.get("case_feasibility_comment"),
                        "other_inaccuracies": st.session_state.feedback.get("other_inaccuracies"),
                        "other_inaccuracies_comment": st.session_state.feedback.get("other_inaccuracies_comment")
                    }
                    if write_to_gsheet(submission_data):
                        st.success("Feedback submitted successfully! Thank you.")
                        st.balloons()
                    else:
                        st.error("Submission failed. Please check the logs or contact Stephen.")
else:
    st.info("Please select your initials and a case from the sidebar to begin.")