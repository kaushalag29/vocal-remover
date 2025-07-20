import os
import subprocess
import shlex
import logging
import signal
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
        
        # We assume this server is run from the 'vocal-remover' directory.
        cmd = f"python inference.py --input '{input_list_format}' --pretrained_model '{request.pretrained_model}'"
        
        if request.tta:
            cmd += " --tta"
        if request.complex:
            cmd += " --complex"
        if request.gpu:
            cmd += " --gpu 0"
            
        execute_separation_command(cmd)
        
        return {"status": "success", "message": "Audio separation completed."}
    except Exception as e:
        logger.error(f"Error during audio separation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shutdown")
async def shutdown():
    logger.info("Shutdown request received, terminating server.")
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutdown", "message": "Server is shutting down."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015) 