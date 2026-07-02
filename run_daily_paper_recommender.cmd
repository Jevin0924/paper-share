@echo off
setlocal

wsl.exe -e bash -lc "cd /mnt/d/faceid/code/paper-share && export PATH=/home/wjw/.nvm/versions/node/v22.21.1/bin:/home/wjw/miniconda3/bin:$PATH && /home/wjw/miniconda3/bin/python3 tools/paper_recommender/run_daily.py %* >> logs/paper_recommender.log 2>&1"
set "TASK_EXIT=%ERRORLEVEL%"

endlocal & exit /b %TASK_EXIT%
