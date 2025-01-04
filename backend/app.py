# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import cv2
import numpy as np
import requests 

from modules.ocr_processor import OCRProcessor
from modules.pii_analyzer import PIIAnalyzer
from modules.pdf_report_generator import PDFReportGenerator
from modules.aws_handler import AWSHandler

app = Flask(__name__)
CORS(app)

ocr_processor = OCRProcessor()
pii_analyzer = PIIAnalyzer()
report_generator = PDFReportGenerator()

@app.route('/api/process-local-files', methods=['POST'])
def process_local_files():
    # Ensure files are received correctly
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    
    report_data = []
    error_data = []  # To collect errors for problematic files

    for file in files:
        try:
            # Process each file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                
                # Process the file: Extract text and classify PII entities
                detected_text = ocr_processor.detect_text(temp_file.name)
                entities = pii_analyzer.classify_text(detected_text)
                
                # Filter out entities with confidence below threshold
                filtered_entities = [
                    entity for entity in entities if entity['confidence'] >= 0.10
                ]
                
                if filtered_entities:  # Only add to report if valid entities exist
                    report_data.append({
                        "image_name": file.filename,
                        "detected_text": detected_text,
                        "entities": filtered_entities
                    })
                
                os.unlink(temp_file.name)  # Delete temp file after processing
        except Exception as e:
            # Collect error for this specific file and continue
            error_data.append({
                "file": file.filename,
                "error": str(e)
            })

    # If no valid entities found and no errors, return a message
    if not report_data and not error_data:
        return jsonify({"message": "No valid PII detected in the files"}), 200

    # Generate the report if valid data is available
    report_filename = "PII_Report.pdf"
    if report_data:
        report_generator.filename = report_filename
        report_generator.generate_report(report_data)
    
    response = {
        "message": "Files processed successfully",
        "report": report_filename if report_data else None,  # Report only if valid data exists
        "errors": error_data  # Include errors in the response
    }

    return jsonify({
            "message": "local files processed successfully",
            "report": report_filename
        })



@app.route('/api/process-s3-files', methods=['POST'])
def process_s3_files():
    data = request.json
    aws_access_key = data.get('aws_access_key')
    aws_secret_access_key = data.get('aws_secret_access_key')
    region_name = data.get('region_name')

    try:
        aws_handler = AWSHandler(aws_access_key, aws_secret_access_key, region_name)
        buckets = aws_handler.list_all_buckets()
        
        report_data = []
        for bucket in buckets:
            urls = aws_handler.list_objects_in_bucket(bucket)
            for url in urls:
                response = requests.get(url)
                if response.status_code == 200:
                    image_array = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        detected_text = ocr_processor.detect_text_s3(img)
                        entities = pii_analyzer.classify_text(detected_text)
                        report_data.append({
                            "image_name": url.split('/')[-1], 
                            "detected_text": detected_text, 
                            "entities": entities
                        })

        # Generate report
        report_filename = "S3_PII_Report.pdf"
        report_generator.filename = report_filename
        report_generator.generate_report(report_data)

        return jsonify({
            "message": "S3 files processed successfully",
            "report": report_filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

REPORTS_DIRECTORY = os.path.join(os.getcwd(), '')

@app.route('/download-report/<filename>')
def download_report(filename):
    # Construct the full file path
    file_path = os.path.join(REPORTS_DIRECTORY, filename)
    print(f"Looking for file at: {file_path}")  # Debugging

    try:
        # Serve the file from the reports directory
        return send_file(file_path, as_attachment=True)
    except FileNotFoundError:
        print(f"File not found: {file_path}")  # Debugging
        return jsonify({"error": "Report not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)