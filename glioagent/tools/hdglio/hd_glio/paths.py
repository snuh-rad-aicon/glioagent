import os

# Parameter files are kept in-repo under tools/ckpts/hdglio (this file lives at
# tools/hdglio/hd_glio/paths.py, so tools/ = three levels up).
_TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_with_parameter_files = os.path.join(_TOOLS, 'ckpts', 'hdglio')
