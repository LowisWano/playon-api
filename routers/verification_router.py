from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import shutil
from utils.verification_utils import *

router = APIRouter(
    prefix="/verification",
    tags=["verification"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/verify-face")
async def verify(
    img1: UploadFile = File(...), 
    img2: UploadFile = File(...), 
    img3: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    """
    Endpoint to verify faces in uploaded images against a selfie image.

    This function handles the '/verify' POST request, which accepts four image files
    (three target images and one selfie image) and verifies each target image against
    the selfie image using face verification.

    Args:
        img1 (UploadFile): The first target image file.
        img2 (UploadFile): The second target image file.
        img3 (UploadFile): The third target image file.
        selfie (UploadFile): The selfie image file to compare against.

    Returns:
        dict: A dictionary containing the face verification results for each target image.
            - image (str): The path to the image that was verified.
            - verified (bool): Whether the image was verified as a match.
            - distance (float): The facial recognition distance between the images.
            - threshold (float): The threshold for verification.    

    Raises:
        HTTPException: If any error occurs during the file processing or face verification.
    """

    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # List of image files with their respective suffixes
        images = [
            (img1, "_1"),
            (img2, "_2"),
            (img3, "_3"),
            (selfie, "_selfie")
        ]

        # Save files and generate paths dynamically
        image_paths = []
        for img, suffix in images:
            filename, ext = os.path.splitext(img.filename)
            file_path = os.path.join(UPLOAD_DIR, f"{filename}{suffix}{ext}")
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(img.file, buffer)

            print(f"Saved: {file_path} - Exists: {os.path.exists(file_path)}")
            image_paths.append(file_path)

        # Run face verification asynchronously
        results = await face_verification(image_paths[-1], *image_paths[:-1])
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")
