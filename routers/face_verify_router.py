from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import shutil
from utils.face_verify_utils import face_verification

router = APIRouter(
    prefix="/verification/face",
    tags=["verification"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/verify")
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
        results = await face_verification(*image_paths)
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")
