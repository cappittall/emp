sudo apt-get install wine
wget https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe
wine python-3.8.10-amd64.exe
wine python -m pip install pyinstaller
wine python -m pip install your-required-packages
wine python -m PyInstaller --onefile --noconsole --add-data "data;data" --hidden-import wmi --hidden-import win32com.client gui.py
