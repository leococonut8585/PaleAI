from fastapi import APIRouter, UploadFile, File, Depends
from dependencies import get_current_active_user
from typing import Dict

from file_processors import stage0_process

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
    dependencies=[Depends(get_current_active_user)]
)

@router.post("/")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, str]:
    content = await file.read()
    result = await stage0_process(file.filename, content)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "stage0_result": result,
    }
