import os
import subprocess
import shlex
import logging
import signal
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Vocal Remover API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeparationRequest(BaseModel):
    input_paths: List[str]
    pretrained_model: str = 'models/baseline.pth'
    gpu: bool = False
    tta: bool = True
    postprocess: bool = False
    complex: bool = False

def execute_separation_command(cmd):
    """Executes a command and logs its output."""
    logger.info(f"Executing command: {cmd}")
    try:
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        if process.poll() != 0:
            raise subprocess.CalledProcessError(process.poll(), cmd)
    except Exception as e:
        logger.error(f"Error executing vocal separation: {e}", exc_info=True)
        raise

@app.post("/separate-audio")
async def separate_audio(request: SeparationRequest):
    try:
        # The inference script can take a comma-separated list of files.
        input_list_format = ",".join(request.input_paths)

        # Use sys.executable to ensure subprocess uses the same Python interpreter
        # that's running this server (with all conda environment packages)
        cmd = f"{sys.executable} inference.py --input '{input_list_format}' --pretrained_model '{request.pretrained_model}'"
        
        if request.tta:
            cmd += " --tta"
        if request.complex:
            cmd += " --complex"
        if request.gpu:
            cmd += " --gpu 0"
            
        execute_separation_command(cmd)

        # Build output file paths based on inference.py's naming convention
        # inference.py writes files to the current working directory
        vocals = []
        instrumentals = []

        for input_path in request.input_paths:
            basename = os.path.splitext(os.path.basename(input_path))[0]
            vocal_path = f"{basename}_Vocals.wav"
            instrumental_path = f"{basename}_Instruments.wav"

            # Verify files were created
            if not os.path.exists(vocal_path):
                logger.error(f"Expected vocal file not found: {vocal_path}")
                raise HTTPException(status_code=500, detail=f"Vocal file not created: {vocal_path}")
            if not os.path.exists(instrumental_path):
                logger.error(f"Expected instrumental file not found: {instrumental_path}")
                raise HTTPException(status_code=500, detail=f"Instrumental file not created: {instrumental_path}")

            # Convert to absolute paths
            vocals.append(os.path.abspath(vocal_path))
            instrumentals.append(os.path.abspath(instrumental_path))

        logger.info(f"Audio separation completed: {len(vocals)} vocal files, {len(instrumentals)} instrumental files")
        return {
            "vocals": vocals,
            "instrumentals": instrumentals
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during audio separation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "vocal-remover"}

@app.post("/shutdown")
async def shutdown():
    logger.info("Shutdown request received, terminating server.")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutdown", "message": "Server is shutting down."}

if __name__ == "__main__":
    import uvicorn
    # Port is configurable via environment variable
    port = int(os.environ.get("PORT", 8015))
    logger.info(f"Starting Vocal Remover server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 