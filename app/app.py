import streamlit as st
import joblib
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="🔍",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("model/fake_job_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")


# ==========================================
# TITLE
# ==========================================

st.title("🔍 Fake Job Detection System")

st.write(
    "Detect potentially fraudulent job postings using "
    "Machine Learning and suspicious-indicator analysis."
)


# ==========================================
# INPUT METHOD
# ==========================================

st.subheader("Choose Job Input Method")

input_method = st.radio(
    "Select an option:",
    [
        "Enter Job Details",
        "Upload Job Advertisement Image"
    ]
)


# ==========================================
# VARIABLES
# ==========================================

job_text = ""


# ==========================================
# TEXT INPUT
# ==========================================

if input_method == "Enter Job Details":

    st.subheader("📝 Job Details")

    title = st.text_input("Job Title")

    company_profile = st.text_area("Company Profile")

    description = st.text_area("Job Description")

    requirements = st.text_area("Requirements")

    benefits = st.text_area("Benefits")

    job_text = (
        title + " "
        + company_profile + " "
        + description + " "
        + requirements + " "
        + benefits
    )


# ==========================================
# IMAGE INPUT
# ==========================================

else:

    st.subheader("📷 Upload Job Advertisement")

    uploaded_file = st.file_uploader(
        "Upload a job advertisement image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        # Open image
        image = Image.open(uploaded_file)

        # Display image
        st.image(
            image,
            caption="Uploaded Job Advertisement",
            use_container_width=True
        )

        # OCR
        with st.spinner("🔎 Extracting text from image..."):

            extracted_text = pytesseract.image_to_string(image)

        # Display extracted text
        st.subheader("📄 Extracted Job Text")

        if extracted_text.strip():

            st.text_area(
                "OCR Result",
                extracted_text,
                height=250
            )

            job_text = extracted_text

        else:

            st.warning(
                "⚠️ No readable text was detected in the image."
            )


# ==========================================
# CHECK JOB BUTTON
# ==========================================

if st.button("🔍 Check Job"):

    # ==========================================
    # VALIDATE INPUT
    # ==========================================

    if not job_text.strip():

        st.warning(
            "⚠️ Please enter job details or upload a job advertisement."
        )

    else:

        # ==========================================
        # TF-IDF
        # ==========================================

        job_vector = tfidf.transform([job_text])


        # ==========================================
        # ML PREDICTION
        # ==========================================

        prediction = model.predict(job_vector)

        probability = model.predict_proba(job_vector)[0][1]

        ml_score = probability * 100


        # ==========================================
        # SUSPICIOUS INDICATORS
        # ==========================================

        suspicious_words = {

            "registration fee": 25,
            "processing fee": 25,
            "security deposit": 25,
            "pay money": 25,
            "payment required": 25,
            "guaranteed job": 25,
            "urgent hiring": 20,
            "no experience required": 15,
            "high salary": 15,
            "easy money": 20,
            "work from home": 10,
            "training fee": 25,
            "application fee": 25

        }


        found_indicators = []

        job_text_lower = job_text.lower()


        for word, score in suspicious_words.items():

            if word in job_text_lower:

                found_indicators.append(
                    (word, score)
                )


        # ==========================================
        # HYBRID RISK SCORE
        # ==========================================

        indicator_score = sum(
            score
            for word, score in found_indicators
        )

        final_risk_score = min(
            ml_score + indicator_score,
            100
        )


        # ==========================================
        # FINAL RESULT
        # ==========================================

        if final_risk_score >= 70:

            final_result = "FAKE JOB"
            result_icon = "🔴"

        elif final_risk_score >= 30:

            final_result = "SUSPICIOUS JOB"
            result_icon = "🟡"

        else:

            final_result = "REAL JOB"
            result_icon = "🟢"


        # ==========================================
        # DISPLAY RESULT
        # ==========================================

        st.divider()

        st.subheader(
            f"{result_icon} {final_result}"
        )


        # ==========================================
        # RISK SCORE
        # ==========================================

        st.metric(
            "Scam Risk Score",
            f"{final_risk_score:.2f}%"
        )


        # ==========================================
        # RISK LEVEL
        # ==========================================

        if final_risk_score < 30:

            st.success(
                "🟢 LOW RISK"
            )

        elif final_risk_score < 70:

            st.warning(
                "🟡 MEDIUM RISK"
            )

        else:

            st.error(
                "🔴 HIGH RISK"
            )


        # ==========================================
        # WHY THIS RESULT?
        # ==========================================

        st.subheader(
            "🔎 Why was this result given?"
        )

        st.write(
            f"Machine Learning Risk Score: "
            f"**{ml_score:.2f}%**"
        )

        st.write(
            f"Suspicious Indicator Score: "
            f"**+{indicator_score}%**"
        )


        # ==========================================
        # SUSPICIOUS INDICATORS
        # ==========================================

        if found_indicators:

            st.warning(
                "⚠️ Suspicious Indicators Detected"
            )

            for word, score in found_indicators:

                st.write(
                    f"• **{word}** (+{score} risk points)"
                )

        else:

            st.info(
                "✅ No obvious suspicious indicators detected."
            )


        # ==========================================
        # RECOMMENDATION
        # ==========================================

        st.subheader(
            "💡 Recommendation"
        )

        if final_risk_score >= 70:

            st.error(
                "🚨 High scam risk. Do not pay registration, "
                "processing or security fees. Verify the employer "
                "through reliable sources before proceeding."
            )

        elif final_risk_score >= 30:

            st.warning(
                "⚠️ This job contains suspicious signals. "
                "Verify the employer and job details carefully."
            )

        else:

            st.success(
                "✅ Low scam risk detected. Still verify the "
                "employer before sharing personal information."
            )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.divider()

st.subheader(
    "📊 Model Performance"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "98.01%")

with col2:
    st.metric("Precision", "76.44%")

with col3:
    st.metric("Recall", "87.85%")

with col4:
    st.metric("F1 Score", "81.75%")


# ==========================================
# CONFUSION MATRIX
# ==========================================

st.subheader(
    "📈 Confusion Matrix"
)

confusion_data = {
    "Actual": [
        "Real Job",
        "Fake Job"
    ],

    "Predicted Real": [
        3346,
        22
    ],

    "Predicted Fake": [
        49,
        159
    ]
}

st.table(confusion_data)


# ==========================================
# ABOUT
# ==========================================

st.divider()

st.subheader(
    "ℹ️ About the System"
)

st.write(
    """
This system uses TF-IDF for text feature extraction and
Logistic Regression for fake-job classification.

Users can either enter job details manually or upload
a job advertisement image. OCR extracts text from the
uploaded advertisement, after which the extracted text
is analyzed using the same machine-learning pipeline.

The system provides a scam risk score, suspicious indicators,
risk level and recommendation.
"""
)