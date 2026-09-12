import streamlit as st
import joblib
import pytesseract
import re
from PIL import Image, ImageEnhance, ImageFilter


# ==========================================
# TESSERACT OCR CONFIGURATION
# ==========================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


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
# VARIABLE
# ==========================================

job_text = ""


# ==========================================
# TEXT INPUT
# ==========================================

if input_method == "Enter Job Details":

    st.subheader("📝 Job Details")

    title = st.text_input("Job Title")

    company_profile = st.text_area(
        "Company Profile"
    )

    description = st.text_area(
        "Job Description"
    )

    requirements = st.text_area(
        "Requirements"
    )

    benefits = st.text_area(
        "Benefits"
    )

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

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Job Advertisement",
            use_container_width=True
        )

        # ==========================================
        # OCR
        # ==========================================

        with st.spinner(
            "🔎 Extracting text from image..."
        ):

            # Improve OCR for small / stylized job posters
            ocr_image = image.convert("RGB")

            # Upscale small posters before OCR
            if ocr_image.width < 1600:
                scale = 1600 / ocr_image.width
                new_size = (
                    int(ocr_image.width * scale),
                    int(ocr_image.height * scale)
                )
                ocr_image = ocr_image.resize(new_size)

            # Moderate enhancement: too much contrast can damage
            # coloured poster text, so keep the values controlled.
            ocr_image = ImageEnhance.Contrast(ocr_image).enhance(1.4)
            ocr_image = ImageEnhance.Sharpness(ocr_image).enhance(1.8)
            ocr_image = ocr_image.filter(ImageFilter.SHARPEN)

            # OCR ensemble for posters. A single full-image OCR pass can
            # miss small lines such as "Fee - Rs. 198". We therefore run
            # OCR on the full poster and several overlapping regions, then
            # combine the useful text. This is more robust for stylized
            # advertisements and does not depend on a fixed poster layout.
            ocr_results = []

            # Full poster: scattered text
            ocr_results.append(
                pytesseract.image_to_string(
                    ocr_image,
                    config="--psm 11"
                )
            )

            # Full poster fallback: block-style text
            ocr_results.append(
                pytesseract.image_to_string(
                    ocr_image,
                    config="--psm 6"
                )
            )

            # Overlapping horizontal regions help OCR recover small text
            # that gets ignored in the full image.
            w, h = ocr_image.size
            regions = [
                (0, 0, w, int(h * 0.40)),
                (0, int(h * 0.25), w, int(h * 0.70)),
                (0, int(h * 0.50), w, h),
            ]

            for left, top, right, bottom in regions:
                crop = ocr_image.crop((left, top, right, bottom))
                ocr_results.append(
                    pytesseract.image_to_string(
                        crop,
                        config="--psm 11"
                    )
                )

            # Specifically re-check likely money/contact lines. This is
            # still generic: it does not assume the amount or currency.
            money_crop = ocr_image.crop(
                (0, int(h * 0.45), int(w * 0.75), int(h * 0.75))
            )
            ocr_results.append(
                pytesseract.image_to_string(
                    money_crop,
                    config="--psm 6"
                )
            )

            # Merge non-empty OCR outputs. Keeping all passes gives the
            # downstream keyword/context rules the best chance to recover
            # words missed by one OCR pass.
            extracted_text = "\n".join(
                text.strip()
                for text in ocr_results
                if text and text.strip()
            )

        # ==========================================
        # DISPLAY OCR TEXT
        # ==========================================

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
            "⚠️ Please enter job details or upload "
            "a job advertisement."
        )

    else:

        # ==========================================
        # TF-IDF FEATURE EXTRACTION
        # ==========================================

        job_vector = tfidf.transform(
            [job_text]
        )


        # ==========================================
        # MACHINE LEARNING PREDICTION
        # ==========================================

        prediction = model.predict(
            job_vector
        )

        probability = model.predict_proba(
            job_vector
        )[0][1]

        ml_score = probability * 100


        # ==========================================
        # TEXT NORMALIZATION
        # ==========================================

        # Keep an untouched lowercase copy for regex checks such as email.
        # The normalized version below changes @ into "at", so email
        # detection must use the raw copy.
        raw_text_lower = job_text.lower()
        job_text_lower = raw_text_lower

        # Normalize rupee symbol
        job_text_lower = job_text_lower.replace(
            "₹",
            " rs "
        )

        # Normalize @ symbol
        job_text_lower = job_text_lower.replace(
            "@",
            " at "
        )

        # IMPORTANT:
        # Convert slash to space.
        # Example:
        # registration/joining fee
        # becomes:
        # registration joining fee

        job_text_lower = job_text_lower.replace(
            "/",
            " "
        )

        # Normalize hyphens
        job_text_lower = job_text_lower.replace(
            "-",
            " "
        )

        # Normalize multiple spaces
        job_text_lower = re.sub(
            r"\s+",
            " ",
            raw_text_lower
        ).strip()


        # ==========================================
        # SAFE TEXT FOR SCAM DETECTION
        # ==========================================

        # IMPORTANT:
        # Negative phrases such as:
        #
        # "no joining fee"
        # "no registration fee"
        # "no payment required"
        #
        # must NOT trigger scam indicators.

        safe_text = job_text_lower


        # ==========================================
        # NEGATIVE / SAFE PHRASES
        # ==========================================

        negative_phrases = [

            # Registration
            "no registration fee",
            "no registration fees",
            "no registration charge",
            "no registration charges",
            "no registration amount",

            # Registration + Joining
            "no registration joining fee",
            "no registration joining fees",
            "no registration joining charge",
            "no registration joining charges",
            "no registration joining amount",

            # Joining
            "no joining fee",
            "no joining fees",
            "no joining charge",
            "no joining charges",
            "no joining amount",
            "no joining payment",

            # Application
            "no application fee",
            "no application fees",
            "no application charge",
            "no application charges",

            # Processing
            "no processing fee",
            "no processing fees",
            "no processing charge",
            "no processing charges",

            # Security
            "no security deposit",
            "no deposit required",
            "no security amount",

            # Training
            "no training fee",
            "no training fees",
            "no training charge",
            "no training charges",

            # Payment
            "no payment required",
            "no payment is required",
            "no payment needed",
            "no payment necessary",

            # General
            "no fees required",
            "no fee required",
            "no fees",
            "no fee",
            "free to apply",
            "apply for free"
        ]


        # ==========================================
        # REMOVE NEGATIVE PHRASES
        # ==========================================

        for phrase in negative_phrases:

            safe_text = safe_text.replace(
                phrase,
                " "
            )


        # Normalize again

        safe_text = re.sub(
            r"\s+",
            " ",
            safe_text
        ).strip()


        # ==========================================
        # SUSPICIOUS INDICATORS
        # ==========================================

        suspicious_patterns = {

            "registration fee": (
                [
                    "registration fee",
                    "registration fees",
                    "registration charge",
                    "registration charges",
                    "registration amount",
                    "registration payment",
                    "pay registration"
                ],
                30
            ),

            "joining fee": (
                [
                    "joining fee",
                    "joining fees",
                    "joining charge",
                    "joining charges",
                    "package joining fee",
                    "package joining fees",
                    "joining amount",
                    "joining payment"
                ],
                35
            ),

            "payment required before joining": (
                [
                    "pay while joining",
                    "pay before joining",
                    "payment while joining",
                    "payment before joining",
                    "pay just rs",
                    "pay rs",
                    "payment required",
                    "payment is required",
                    "pay to join",
                    "pay for joining"
                ],
                40
            ),

            "processing fee": (
                [
                    "processing fee",
                    "processing fees",
                    "processing charge",
                    "processing charges",
                    "processing amount"
                ],
                30
            ),

            "security deposit": (
                [
                    "security deposit",
                    "deposit required",
                    "pay deposit",
                    "refundable deposit",
                    "security amount"
                ],
                30
            ),

            "training fee": (
                [
                    "training fee",
                    "training fees",
                    "training charge",
                    "training charges",
                    "pay for training",
                    "paid training",
                    "training package"
                ],
                30
            ),

            "guaranteed job": (
                [
                    "guaranteed job",
                    "job guaranteed",
                    "100% job guarantee",
                    "job guarantee",
                    "guarantee job",
                    "guaranteed employment"
                ],
                25
            ),

            "urgent hiring": (
                [
                    "urgent hiring",
                    "urgent requirement",
                    "immediate hiring",
                    "immediate joining",
                    "join immediately",
                    "hiring immediately"
                ],
                20
            ),

            "no experience required": (
                [
                    "no experience required",
                    "no experience needed",
                    "no experience"
                ],
                8
            ),

            "high earning claim": (
                [
                    "maximum earning",
                    "maximum earnings",
                    "earn rs",
                    "daily income",
                    "high salary",
                    "high earning",
                    "earn up to",
                    "earn upto",
                    "huge income",
                    "attractive salary",
                    "unlimited earning",
                    "unlimited earnings",
                    "unlimited earning potential",
                    "earning potential"
                ],
                20
            ),

            "work from home": (
                [
                    "work from home",
                    "working at home",
                    "work from mobile",
                    "work from laptop",
                    "work from mobile laptop",
                    "work from mobile or laptop",
                    "working from mobile",
                    "working from laptop"
                ],
                5
            ),

            "limited seats": (
                [
                    "limited seats",
                    "limited vacancies",
                    "limited openings",
                    "few seats left",
                    "limited positions"
                ],
                15
            ),

            "apply immediately": (
                [
                    "apply now",
                    "apply immediately",
                    "message now",
                    "send message now",
                    "dm now",
                    "contact now",
                    "apply today"
                ],
                10
            ),

            "whatsapp recruitment": (
                [
                    "whatsapp",
                    "whatsapp me",
                    "contact on whatsapp",
                    "message on whatsapp"
                ],
                5
            )
        }


        # ==========================================
        # FIND SUSPICIOUS INDICATORS
        # ==========================================

        found_indicators = []

        for indicator, (patterns, score) in suspicious_patterns.items():

            matched = False

            for pattern in patterns:

                if pattern in safe_text:

                    matched = True
                    break

            if matched:

                found_indicators.append(
                    (indicator, score)
                )


        # ==========================================
        # CONTEXTUAL OCR MONEY / REGISTRATION DETECTION
        # ==========================================
        # OCR may separate the words "Registration" and "Fee" or
        # misread the rupee amount. We therefore do NOT depend on
        # the exact amount being recognised.

        registration_present = "registration" in safe_text
        fee_present = "fee" in safe_text or "fees" in safe_text

        strong_scam_context = (
            "no experience needed" in safe_text
            or "no experience required" in safe_text
            or "limited seats" in safe_text
            or "unlimited earning" in safe_text
            or "earning potential" in safe_text
            or "online business" in safe_text
        )

        if registration_present and fee_present:
            found_indicators.append(
                ("registration fee detected", 40)
            )

        # OCR can miss the word "Registration" even when it clearly reads
        # the actual fee line. A fee/payment request combined with common
        # recruitment-poster context is therefore treated as a strong risk
        # signal. This does not depend on the exact currency or amount.
        elif fee_present and strong_scam_context:
            found_indicators.append(
                ("job fee/payment request detected", 40)
            )

        elif registration_present and strong_scam_context:
            found_indicators.append(
                ("suspicious registration request", 30)
            )


        # ==========================================
        # LEGITIMATE / VERIFICATION SIGNALS
        # ==========================================

        legitimate_patterns = {

            "company information": (
                [
                    "pvt ltd",
                    "private limited",
                    "ltd.",
                    "inc.",
                    "corporation",
                    "technologies",
                    "solutions",
                    "software company"
                ],
                4
            ),

            "specific job position": (
                [
                    "position:",
                    "positions:",
                    "role:",
                    "job title:",
                    "designation:",
                    "manager",
                    "engineer",
                    "developer",
                    "executive",
                    "analyst",
                    "specialist",
                    "intern",
                    "internship"
                ],
                4
            ),

            "experience requirement": (
                [
                    "experience",
                    "years experience",
                    "years of experience",
                    "fresher",
                    "entry level"
                ],
                4
            ),

            "specific location": (
                [
                    "location:",
                    "delhi",
                    "chennai",
                    "bangalore",
                    "bengaluru",
                    "hyderabad",
                    "mumbai",
                    "pune",
                    "noida",
                    "gurgaon",
                    "gurugram",
                    "kolkata"
                ],
                3
            ),

            "professional application method": (
                [
                    "mail your resume",
                    "send your resume",
                    "submit your resume",
                    "careers@",
                    "hr@",
                    "apply through our website",
                    "official website",
                    "company website"
                ],
                5
            ),

            "interview process": (
                [
                    "interview",
                    "technical interview",
                    "hr interview",
                    "selection process",
                    "screening process",
                    "assessment"
                ],
                4
            ),

            "no payment mentioned": (
                [
                    "no registration fee",
                    "no registration fees",
                    "no joining fee",
                    "no joining fees",
                    "no application fee",
                    "no application fees",
                    "no payment required",
                    "no payment is required",
                    "no fees required",
                    "no fee required",
                    "free to apply",
                    "apply for free"
                ],
                6
            )
        }


        # ==========================================
        # FIND LEGITIMATE SIGNALS
        # ==========================================

        found_legitimate = []

        for indicator, (patterns, score) in legitimate_patterns.items():

            matched = False

            for pattern in patterns:

                if pattern in job_text_lower:

                    matched = True
                    break

            if matched:

                found_legitimate.append(
                    (indicator, score)
                )


        # ==========================================
        # CORPORATE EMAIL DETECTION
        # ==========================================

        free_email_domains = [
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "rediffmail.com",
            "aol.com",
            "icloud.com",
            "protonmail.com"
        ]

        email_matches = re.findall(
            r'[\w\.-]+@[\w\.-]+\.\w+',
            job_text_lower
        )

        corporate_email_found = False

        for email in email_matches:

            domain = email.split("@")[-1].strip()

            if domain not in free_email_domains:

                corporate_email_found = True
                break

        if corporate_email_found:

            found_legitimate.append(
                (
                    "corporate/company-domain email",
                    6
                )
            )


        # ==========================================
        # OFFICIAL WEBSITE DETECTION
        # ==========================================

        website_patterns = [
            "www.",
            "https://",
            "http://"
        ]

        website_found = any(
            pattern in job_text_lower
            for pattern in website_patterns
        )

        if website_found:

            found_legitimate.append(
                (
                    "official website",
                    4
                )
            )


        # ==========================================
        # CALCULATE SUSPICIOUS SCORE
        # ==========================================

        indicator_score = sum(
            score
            for indicator, score in found_indicators
        )

        indicator_score = min(
            indicator_score,
            60
        )


        # ==========================================
        # CALCULATE LEGITIMATE SCORE
        # ==========================================

        legitimate_score = sum(
            score
            for indicator, score in found_legitimate
        )

        legitimate_score = min(
            legitimate_score,
            25
        )


        # ==========================================
        # IMPROVED HYBRID RISK SCORING
        # ==========================================

        # ML contributes 30%.

        adjusted_ml_score = ml_score * 0.30

        suspicious_contribution = indicator_score

        legitimate_reduction = legitimate_score

        final_risk_score = (
            adjusted_ml_score
            + suspicious_contribution
            - legitimate_reduction
        )


        # ==========================================
        # CONTEXTUAL ML CONTROL
        # ==========================================

        # If there are no suspicious indicators,
        # ML alone cannot make the job HIGH RISK.

        if indicator_score == 0:

            final_risk_score = min(
                final_risk_score,
                35
            )

        # Mild suspicious indicators

        elif indicator_score < 20:

            final_risk_score = min(
                final_risk_score,
                55
            )


        # ==========================================
        # PAYMENT / MONEY REQUEST RULE
        # ==========================================

        payment_risk_found = any(
            indicator in [
                "registration fee",
                "joining fee",
                "payment required before joining",
                "processing fee",
                "security deposit",
                "training fee",
                "registration fee detected",
                "suspicious registration request",
                "job fee/payment request detected"
            ]
            for indicator, score in found_indicators
        )

        if payment_risk_found:

            final_risk_score = max(
                final_risk_score,
                70
            )


        # Strong contextual rule for OCR posters. If a poster contains
        # registration + fee, the payment request is a major risk signal
        # even when the ML model probability is low.
        if any(
            indicator == "registration fee detected"
            for indicator, score in found_indicators
        ):
            final_risk_score = max(
                final_risk_score,
                75
            )


        # ==========================================
        # STRONG SCAM COMBINATION
        # ==========================================

        strong_scam_indicators = 0

        strong_categories = [
            "guaranteed job",
            "urgent hiring",
            "no experience required",
            "high earning claim",
            "limited seats"
        ]

        for indicator, score in found_indicators:

            if indicator in strong_categories:

                strong_scam_indicators += 1


        if strong_scam_indicators >= 3:

            final_risk_score = max(
                final_risk_score,
                65
            )


        # ==========================================
        # WFH + WHATSAPP + HIGH EARNING
        # ==========================================

        has_wfh = any(
            indicator == "work from home"
            for indicator, score in found_indicators
        )

        has_whatsapp = any(
            indicator == "whatsapp recruitment"
            for indicator, score in found_indicators
        )

        has_high_earning = any(
            indicator == "high earning claim"
            for indicator, score in found_indicators
        )

        if (
            has_wfh
            and has_whatsapp
            and has_high_earning
        ):

            final_risk_score = max(
                final_risk_score,
                60
            )


        # ==========================================
        # SPECIAL PROTECTION FOR EXPLICITLY
        # NO-FEE JOBS
        # ==========================================

        explicit_no_fee = any(
            phrase in job_text_lower
            for phrase in [
                "no registration fee",
                "no registration fees",
                "no joining fee",
                "no joining fees",
                "no registration joining fee",
                "no registration joining fees",
                "no application fee",
                "no application fees",
                "no payment required",
                "no payment is required",
                "no fees required",
                "no fee required",
                "free to apply",
                "apply for free"
            ]
        )

        if explicit_no_fee and not payment_risk_found:

            # An explicit no-fee statement should not
            # be treated as a payment scam.

            final_risk_score = min(
                final_risk_score,
                35
            )


        # ==========================================
        # FINAL SCORE LIMIT
        # ==========================================

        final_risk_score = max(
            0,
            min(
                final_risk_score,
                100
            )
        )


        # ==========================================
        # FINAL CLASSIFICATION
        # ==========================================

        if final_risk_score >= 70:

            final_result = (
                "HIGH RISK / POSSIBLE FAKE JOB"
            )

            result_icon = "🔴"

        elif final_risk_score >= 40:

            final_result = "SUSPICIOUS JOB"

            result_icon = "🟡"

        else:

            final_result = "LOW RISK JOB"

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

        if final_risk_score < 40:

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
        # RISK ANALYSIS
        # ==========================================

        st.subheader(
            "🔎 Risk Analysis"
        )

        st.write(
            f"Machine Learning Score: "
            f"**{ml_score:.2f}%**"
        )

        st.write(
            f"Adjusted ML Contribution: "
            f"**{adjusted_ml_score:.2f}%**"
        )

        st.write(
            f"Suspicious Indicator Score: "
            f"**+{indicator_score}%**"
        )

        st.write(
            f"Legitimate Signal Reduction: "
            f"**-{legitimate_score}%**"
        )


        # ==========================================
        # SUSPICIOUS INDICATORS
        # ==========================================

        st.subheader(
            "⚠️ Suspicious Indicators"
        )

        if found_indicators:

            for indicator, score in found_indicators:

                st.write(
                    f"🔴 **{indicator}** "
                    f"(+{score} risk points)"
                )

        else:

            st.success(
                "✅ No major suspicious indicators detected."
            )


        # ==========================================
        # POSITIVE VERIFICATION SIGNALS
        # ==========================================

        st.subheader(
            "✅ Positive / Verification Signals"
        )

        if found_legitimate:

            for indicator, score in found_legitimate:

                st.write(
                    f"🟢 **{indicator}** "
                    f"(-{score} risk points)"
                )

        else:

            st.info(
                "ℹ️ No strong verification signals detected."
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
                "joining, processing or security fees. Verify "
                "the employer through reliable sources before "
                "proceeding."
            )

        elif final_risk_score >= 40:

            st.warning(
                "⚠️ This job contains some suspicious signals. "
                "Verify the employer, recruiter and job details "
                "carefully before proceeding."
            )

        else:

            st.success(
                "✅ Low scam risk detected. The advertisement "
                "contains some positive verification signals. "
                "Still verify the employer before sharing "
                "personal information."
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

    st.metric(
        "Accuracy",
        "98.01%"
    )

with col2:

    st.metric(
        "Precision",
        "76.44%"
    )

with col3:

    st.metric(
        "Recall",
        "87.85%"
    )

with col4:

    st.metric(
        "F1 Score",
        "81.75%"
    )


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

st.table(
    confusion_data
)


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

The system combines machine-learning prediction,
suspicious-indicator analysis and legitimate verification
signals to generate an informative scam risk score.

The system provides a risk level, suspicious indicators,
positive verification signals and a recommendation to
help users identify potentially fraudulent job postings.
"""
)