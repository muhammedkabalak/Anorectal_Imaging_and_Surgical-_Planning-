AI-Based Precision Anorectal Imaging and Surgical Planning
===========================================================

This project presents a complete deep learning pipeline for automatic segmentation of anorectal lesions in MRI images. Developed by a multidisciplinary team at Istinye University, the system aims to support radiologists and surgeons in identifying pathological regions more accurately and efficiently for pre-operative planning.

Project Objectives
------------------
- Develop an AI-powered segmentation module using medical imaging datasets.
- Build an interface for visualization and clinical usability.
- Optimize model training and inference for real-time performance.
- Enable hybrid segmentation support (AI + manual correction).
- Prepare the tool for future integration with hospital imaging systems (PACS/DICOM).

Key Features
------------
- U-Net model architecture for semantic segmentation.
- Streamlit-based interface for quick demo and testing.
- Input: Preprocessed axial mid-slices of 3D MRI scans.
- Output: Predicted binary masks + overlay visualization.
- Training Dice score: ~85%.

Dataset
-------
- Source: Medical Segmentation Decathlon – Task05: Prostate
- Format: NIfTI .nii.gz files for images and segmentation labels.
- Preprocessing: PNG conversion, normalization, center-slice extraction.

How to Run
----------

1. Environment Setup

    python -m venv venv
    source venv/bin/activate      # On Windows: venv\Scripts\activate
    pip install -r requirements.txt

2. Extract and Preprocess Dataset

    Ensure Task05_Prostate.tar is extracted to:

        ./prostate_dataset/Task05_Prostate/

    Then run:

        python convert_to_png.py

3. Train the Model

        python train_model.py

4. Launch the Demo App

        streamlit run app.py

Outputs
-------
- Sample MRI slice
- Ground truth mask
- Predicted segmentation mask
- Overlayed lesion mask on the MRI

Project Structure
-----------------
    app.py
    convert_to_png.py
    train_model.py
    model.pth
    /images
    /masks
    /prostate_dataset
    requirements.txt

Team Contributions
------------------
- Data Preprocessing – Beyzanur Yılmaz
- Model Architecture & Training – Afra Ceren
- Streamlit Frontend – Yasin Gül
- Evaluation & Reporting – Efe Çetin
- Coordination & Presentation – Muhammed Kabalak

Future Work
-----------
- Extend training data with real-world clinical MRIs.
- Support 3D volume segmentation.
- Connect interface to DICOM viewers or PACS systems.
- Add manual editing tools for radiologists.

License
-------
This project is for academic and non-commercial purposes only.
