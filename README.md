# -Fraud-transaction-receipt-and-loan-complaint-detection.-
This project detects fraud in both transaction receipts using image-based OCR and loan complaints using text classification techniques.


# 🕵️‍♂️ Fraud Detection in Loan Applications, Transaction Receipts & Loan Complaints

## 🔍 Project Overview

This project uses machine learning and OCR to detect fraudulent activities across three different data sources:

1. **Loan Applications (Tabular Data)**
2. **Transaction Receipts (Image + Text OCR)**
3. **Loan Complaints (Text-based messages or images)**

It integrates multiple ML workflows into a single unified framework to flag suspicious activities based on data patterns or extracted content.

---

## 📁 Project Structure

```
├── loan_applications.csv         # Tabular loan dataset
├── transactions.csv              # (Optional) Additional transactions
├── models/
│   ├── loan_model.pkl            # Fraud detection model for tabular data
│   ├── receipt_model.pkl         # Receipt fraud classifier
│   └── complaint_model.pkl       # Text-based complaint classifier
├── images/                       # Receipts / Complaints uploaded
├── ocr_utils.py                  # OCR functions using Tesseract
├── app.ipynb / main.py           # Main notebook/script for execution
└── README.md                     # You’re reading it :)
```

---

## 📊 1. Loan Application Fraud Detection

* **Input**: `loan_applications.csv`
* **Features**: Includes employment status, income, loan type, amount, EMI, credit score, etc.
* **Target**: `fraud_flag` (1 for fraud, 0 for genuine)

### ✅ Model Pipeline

* Preprocessing: One-hot encoding, scaling
* Model: `RandomForestClassifier`
* Accuracy: \~100% on clean sample data

---

## 📸 2. Transaction Receipt Fraud Detection (OCR + ML)

* **Input**: Scanned images or screenshots of receipts
* **Tech**: `pytesseract` used to extract text from images
* **Features Extracted**:

  * `contains_gibberish`
  * `mentions_balance_due`
  * `pending/unknown status`
  * `invoice/abuse`

### ✅ Model Pipeline

* Simple rule-based feature extraction
* Model: `RandomForestClassifier`
* Custom trained using sample receipts

---

## 🗣️ 3. Loan Complaint Text Message Fraud Detection

* **Input**: Text messages, complaints, legal notices, or screenshot images
* **Model**: NLP-based pipeline using `CountVectorizer + RandomForest`
* Trained on a small sample dataset of real and fraud complaints

### ✅ Fraud Indicators

* Requests for UPI payments
* Threatening legal messages
* Fake links or app references
* Suspicious tone or fake signatures

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn tensorflow keras opencv-python pytesseract pillow
```

Also, make sure [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) is installed and configured on your system or Colab.

---

## 🚀 How to Run

1. Upload the dataset: `loan_applications.csv`
2. Train the loan fraud model (preprocessing + model training included)
3. Upload a **receipt image** or **complaint screenshot**
4. The OCR + text classifier will:

   * Extract text
   * Detect possible fraud
   * Output: ✅ NOT FRAUD or 🚨 FRAUD

---

## 📌 Example Outputs

* **Receipt Fraud Detection**:

  ```
  Extracted Text:
  INTERBANK GIRO ABUSE -RMXXX
  STATUS: PENDING
  🔍 Prediction: 🚨 FRAUD
  ```

* **Complaint Text Detection**:

  ```
  "The lawyer has been appointed... pay via RupeeFund link..."
  🔍 Prediction: 🚨 FRAUD
  ```

* **Loan Application**:

  * Model predicted loan #12345 as `✅ NOT FRAUD`
  * Loan #98765 flagged as `🚨 FRAUD`

---

## 🧠 Future Improvements

* Use deep learning for text embeddings (BERT, etc.)
* OCR enhancement using better image pre-processing
* Expand training data for higher generalization
* Deploy as a Streamlit or Flask web app

---


## 🙌 Acknowledgements

* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
* Scikit-learn, Colab, Pandas, PIL, OpenCV


# DATA SET LINK
--->https://drive.google.com/file/d/1pojmO9-IGXfT0a576Avrzp5iVAvtrpfz/view?usp=sharing
