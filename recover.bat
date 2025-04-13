@echo off
echo Attempting to recover files for HDP Project...

REM Check if temp directory still exists
if exist "c:\HDP Project\temp" (
    echo Found temporary backup directory! Restoring files...
    xcopy "c:\HDP Project\temp\*.*" "c:\HDP Project\" /s /e /y
    echo Restored files from temporary backup.
) else (
    echo Temporary backup directory not found.
    echo Checking for other potential recovery sources...
)

REM Check Windows temporary files
if exist "%TEMP%\HDP Project*" (
    echo Found potential backup in Windows temp folder.
    echo Please check: %TEMP% for any HDP Project files
)

echo.
echo Recovery attempt completed.
echo.
echo IMPORTANT: If your files were not recovered, please:
echo 1. Check your Recycle Bin for recently deleted files
echo 2. Check if you have any other backups of your project
echo 3. If you're using Git, try "git checkout ." to restore tracked files
echo.
echo Press any key to exit...
pause > nul