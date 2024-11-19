pyinstaller --onefile --noconsole --add-data "data;data" --hidden-import wmi --hidden-import win32com.client gui.py
