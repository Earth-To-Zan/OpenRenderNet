# OpenRenderNet
*An open source blender render farm system.*
*(Works on Linux, Windows, locally, and on external networks)*

**First clone the repository to your dedicated folder:**
`git clone https://github.com/Earth-To-Zan/OpenRenderNet.git`

**Install requirements (must have `pip`/`python` installed):**
`pip install -r requirements.txt`

# Key notes & Installation:

**(Run a terminal in the same directory as the downloaded scripts)**

First off, the `coordinator_(Barebones-Example).py` will open up a server backend/webui of which you can login, upload blender jobs, manage workers, download/manage render jobs.
*(Use:* `python coordinator_(Barebones-Example).py` *in your terminal to run the coordinator script)*

Secondly, you need to run database_setup.py before running the coordinator as that will create a simple database which will store admin login info, render job info, registered worker info, and most importantly, create the API Token which the workers will need to use.

Third, you can run the coordinator both on WAN and on LAN, in order to use it from an external network (WAN):
You need you to accesss the coordinator's network's router, and then proceed to portforward for the following ports (You can config them inside the worker/coordinator scripts):
`TCP/UDP`: `5000`: Default port the coordinator uses to do most html requests.
`TCP/UDP`: `5555`: Default port the coordinator server is binded to.
`TCP/UDP`: `5556`: Default port the coordinator uses to upload and download files.

**(YOU DO NOT NEED TO PORT FORWARD FOR THE WORKER)**

With regards to the `worker.py`, you'll need to specify the `IPv4` of the coordinator device, if you are running it locally, use the LAN `IPv4`, otherwise if ran from an external network (WAN); use the global `IPvr4`.
*(Use:* `python worker.py` *in your terminal to run the worker script)*

The `worker.py` will three terminal inputs:
Coordinator IP.
Path to your blender install's `.exe` (E.g. `C:\Program Files\Blender Foundation\Blender 5.0\blender.exe`).
And your coordinator's API Token.

**(THIS HAS BEEN TESTED ONLY ON BLENDER `5.0.0`, IT IS NOT GARUANTEED THAT IT WILL WORK ON OLDER NOR FAR FUTURE BLENDER VERSIONS AS BLENDER CLI DOCUMENTATION CHANGES`)**

# Compiling to `.exe`:

**Install `pyinstaller` (must have `pip`/`python` installed):**
`pip install pyinstaller`

**(Run a terminal in the same directory as:** `worker.py`**)**

**Then, compile by running the following terminal command:**
`pyinstaller --onefile "worker.py"`

If you have any questions, feel free to add me on discord: `entromni`
This project was vibe coded in a day, I hated that there wasn't any easy to use blender render farm projects that anyone could use, especially for the latest blender versions.
I plan on eventually (maybe) hosting my own free volunteer based render farm where people can sign up to offer their GPU compute for others to use, so stay tuned for that; as next I'll need to finish and get that project working, and I should probably make a video demonstrating this current repository.

**DISCLAIMER: VIBE CODE MEANS IT WAS CREATED WITH AI, I HAD USED CHAT.DEEPSEEK.COM TO PROGRAM MOST OF THIS**

Take care, and you are welcome :3
