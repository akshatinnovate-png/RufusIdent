# RufusIdent: AI attendance for Gulmohur High School

A gate-camera attendance system for **Gulmohur High School**, Telco Town, Jamshedpur. It is being deployed in the High School section first.

It has five screens:

- **Boot screen.** Every line is a real check: camera, face engine, roster, database, Excel engine, recognition backend, and Groq.
- **AI assistant.** An agent on Groq `openai/gpt-oss-120b`. It answers questions *and acts*: it can mark students, close registers, export Excel, and open screens. Every action shows in the chat with an **Undo** button.
- **Gate camera.** A sci-fi HUD over the live webcam. Face tracking (468-point mesh), pose, light level, and FPS are all live readings. Good face crops go to the recognition backend. Unsure matches land in **Needs review**, so a wrong student is never marked silently. There is also a manual-mark search box for anyone the camera misses.
- **Class registers.** Class-wise Present / Late / Absent with one-click editing, "mark unmarked absent", and Excel export (a summary sheet plus one sheet per class).
- **Student datasheets.** Attendance %, days present and absent, streak, and the student's school rank on the attendance leaderboard.

## Run it (Windows laptop)

1. Double-click **`start.bat`**.
   - It starts a local server: Python if it is installed, otherwise the built-in PowerShell server in `serve.ps1`.
   - It then opens the app in Edge at `http://localhost:8765`.
2. Allow camera access the first time.
3. Open **Settings**:
   - **Assistant:** paste your Groq API key.
   - **Gate:** pick the camera and set the "late after" time.
   - **Data:** import the roster.

The face engine (MediaPipe) and the Excel library are bundled in `vendor/`, so the gate keeps working without internet. Only the AI assistant needs internet.

## Roster format (.xlsx or .csv)

| admission_no | name | class | section | roll | house | photo |
|---|---|---|---|---|---|---|
| 24001 | Aarav Sharma | X | A | 1 | Tagore | photos/24001.jpg |

- Header names are matched loosely ("Adm No", "Student Name" and "Std" all work).
- `photo` is optional. Put photos in a `photos/` folder next to `index.html`.

## Recognition backend contract (to build next)

The browser finds and quality-checks faces. Matching a face to a student happens on the backend.

```
GET  {backend}/health               -> 200
POST {backend}/recognize            multipart: image=face.jpg (320×320), gate=<name>
     -> {"student_id": "24001", "confidence": 0.97}
     -> {"student_id": null, "confidence": 0}   (unknown face)
```

- Scores at or above **auto-accept** (default 0.90) mark the student present, or late after the cut-off.
- Scores between **review** (0.60) and auto-accept go to the on-screen review queue.

## Where data lives today

Attendance is stored in the browser's IndexedDB on the gate laptop. Use **Settings → Data → Download backup** regularly.

`js/data.js` is the only file that reads or writes data. When the backend is ready, it can swap these functions for API calls without changing any screen.

## Privacy

This system processes face data of minors.

- Get written approval from school management, and parent consent where required (India's DPDP Act 2023 has specific rules for children's data).
- Restrict who can open the evidence folder.
- Treat the Groq API key as a password; it should move to the backend before going live.

---

Made by **GHS AI Team** · Lead developer **Akshat Sarkar** · [MIT License](LICENSE)
