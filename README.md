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
2. Allow camera access the first time. (Set up the recognition backend below first, or the gate runs in manual-mark mode.)
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

## Recognition backend (`backend/`)

The browser tracks faces at camera speed. The backend decides **who** each face is. It runs fully offline on the laptop's CPU; no graphics card is needed.

### Set up (once, on the gate laptop)

1. Install Python 3.10 or newer from python.org (tick "Add python.exe to PATH").
2. Double-click **`backend/install.bat`**.
   - It creates a virtual environment and installs the packages.
   - It downloads the models (about 300 MB) and checks their checksums.
   - It then benchmarks the laptop.
3. Put each student's photos in `photos/<admission_no>/`. Aim for about 20 per student: different days, light and angles, with and without specs.
4. Double-click **`backend/enroll.bat`**.
   - The first run on 3,200 × 20 photos takes a few hours. Run it overnight.
   - It can be stopped and resumed. Later runs only process new or changed photos.
5. `start.bat` now starts the backend automatically.

Optional: copy `backend/config.example.json` to `backend/config.json`. Point `evidence_dir` at the OneDrive folder so every decision's photo syncs automatically.

### How a student is recognised

| Stage | What it does | Why |
|---|---|---|
| Detection zone | The browser's face tracker looks at the centre square of the frame (the gate lane). | Its model is short-range; cropping makes a student 2 m away look about 1.8× bigger. |
| Quality gate (browser) | A frame is only sent if the face is big enough, frontal and well lit. Light is measured on the face, not the background. | Bad frames never reach the matcher. |
| Detector | SCRFD-10G re-detects the face and its 5 landmarks. | Same detector as enrolment, so faces line up identically. |
| Alignment | A similarity transform to the ArcFace 112×112 template. | Removes tilt and scale. |
| Embedding | ArcFace ResNet-50 trained on WebFace600K (InsightFace `buffalo_l`). | A strong open face-recognition model. |
| Gallery | About 20 photos per student. A student's score is the mean of their **best 3** photo matches. | One lucky photo can't win; one bad photo can't lose. |
| Liveness | Two anti-spoof models (MiniFASNetV2 + V1SE) look at the face and its surroundings. | Catches a printed photo or a phone screen held up to the camera. |
| Multi-frame fusion | Each person gets a track. Frames are quality-weighted and averaged, and every frame also votes. | A single odd frame can't mark anyone. |
| Decision | **Match** only if: the fused score ≥ the calibrated threshold, the best student beats the runner-up by a clear margin (doubled for look-alikes), they win most of the frame votes, and the face is live. | |
| Otherwise | **Review**, **Unknown** or **Spoof**. | A teacher confirms with one tap; nobody is marked silently. |

### What `enroll.py` checks for you

It writes `backend/data/enroll_report.csv` and flags:
- **Rejected photos:** no face, several faces, too small, blurry, or head turned away.
- **Misfiled photos:** a photo that doesn't resemble the rest of the student's folder, or looks more like a different student.
- **Thin galleries:** students with fewer than 8 usable photos.
- **Look-alikes:** siblings or twins. The gate is stricter between these students.
- **Calibration:** the accept and review thresholds are set from the school's own photos. Every photo is tested against every other student, and the report gives the genuine-pass rate and the impostor-pass rate.

### API

```
GET  /health                  enrolled students, photos, thresholds
POST /recognize               multipart: image (face crop), track, gate
     -> {status: pending|match|review|unknown|spoof|no_face|low_quality,
         student_id, score, margin, confidence, live, frames, candidates[], ms}
POST /reload                  reload after enrolling (enroll.py calls it)
GET  /photo/{admission_no}    best enrolment photo (used on datasheets)
GET  /student/{admission_no}  photo count, look-alikes
```

### Evidence photos

Every decision saves the frame:
- Matches go to `evidence/<date>/<admission_no>_<time>.jpg`.
- Unsure, unknown and spoof attempts go to the `_review`, `_unknown` and `_spoof` subfolders.

This is a few MB a day, not every frame.

## Where data lives today

Attendance is stored in the browser's IndexedDB on the gate laptop. Use **Settings → Data → Download backup** regularly.

`js/data.js` is the only file that reads or writes data. When the backend is ready, it can swap these functions for API calls without changing any screen.

## Privacy

This system processes face data of minors. Photos, face embeddings and evidence stay on the laptop (and your OneDrive), and `.gitignore` keeps them out of git.

- Get written approval from school management, and parent consent where required (India's DPDP Act 2023 has specific rules for children's data).
- Restrict who can open the evidence folder.
- Treat the Groq API key as a password; it should move to the backend before going live.

Model licences: the InsightFace `buffalo_l` models are released for **non-commercial research use**, which fits a school's own attendance system but check with InsightFace before any commercial use. MiniFASNet (anti-spoof) and MediaPipe are Apache-2.0.

---

Made by **GHS AI Team** · Lead developer **Akshat Sarkar** · [MIT License](LICENSE)
