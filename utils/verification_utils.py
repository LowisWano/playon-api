import asyncio
from deepface import DeepFace
import os

async def face_verification(selfie_image_path, *image_paths):
    """
    Asynchronously verifies a given selfie image against one or more other images using DeepFace.

    Args:
        selfie_image_path (str): The path to the selfie image
        *image_paths (str): The paths to the images to verify against the selfie image

    Returns:
        A list of dictionaries containing the results of the verification
            - image (str): The path to the image that was verified
            - verified (bool): Whether the image was verified as a match
            - distance (float): The facial recognition distance between the images
            - threshold (float): The threshold for verification

    Raises:
        FileNotFoundError: If any of the input images do not exist
        RuntimeError: If an error occurred while processing the images
    """
    results = []

    try:
        print(f"Processing Selfie: {selfie_image_path}")

        for img_path in image_paths:
            print(f"Processing Image: {img_path}")

            # Ensure the image exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"File not found: {img_path}")

            try:
                # Run DeepFace verification asynchronously
                result = await asyncio.to_thread(
                    DeepFace.verify, selfie_image_path, img_path,
                    model_name='Facenet', detector_backend='retinaface'
                )

                results.append({
                    'image': os.path.basename(img_path),
                    'verified': result['verified'],
                    'distance': result['distance'],
                    'threshold': result['threshold'],
                })

            except Exception as e:
                raise RuntimeError(f"Error processing {img_path}: {str(e)}")

            finally:
                os.remove(img_path) if os.path.exists(img_path) else None

        os.remove(selfie_image_path) if os.path.exists(selfie_image_path) else None

        return results

    except Exception as e:
        raise RuntimeError(f"Face verification failed: {str(e)}")
