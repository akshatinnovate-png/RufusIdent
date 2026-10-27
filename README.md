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

The browser streams camera frames to the backend. The backend finds **every** face in the frame (near or 5–6 m away), tracks each person, and decides who they are, several students at once. It runs fully offline:
- on the laptop's CPU, or
- on its built-in graphics through DirectML (Windows), when that is faster and gives the same answers.

No graphics card is needed.

### Set up (once, on the gate laptop)

1. Install Python 3.10 or newer from python.org (tick "Add python.exe to PATH").
2. Double-click **`backend/install.bat`**.
   - It creates a virtual environment and installs the packages.
   - It downloads the models (about 300 MB) and checks their checksums.
   - It then benchmarks the laptop: frames per second with 1 and 4 students in view.
3. Put each student's photos in `photos/<admission_no>/`. Aim for about 20 per student: different days, light and angles, with and without specs.
4. Double-click **`backend/enroll.bat`**.
   - The first run on 3,200 × 20 photos takes a few hours. Run it overnight.
   - It can be stopped and resumed. Later runs only process new or changed photos.
5. `start.bat` now starts the backend automatically.
6. Run **`python evaluate.py`** (in `backend/`) to see the accuracy at 1–6 m on your own students' photos, for your camera. Details below.

Optional: copy `backend/config.example.json` to `backend/config.json`. Point `evidence_dir` at the OneDrive folder so every decision's photo syncs automatically.

### Distance: what the camera has to deliver

Recognition needs pixels on the face. A face is about 15 cm wide. This is roughly how many pixels it gets:

| Camera | 1 m | 2 m | 3 m | 4 m | 5 m | 6 m |
|---|---|---|---|---|---|---|
| 1080p webcam, 78° (typical) | 178 | 89 | 59 | 44 | 36 | 30 |
| 4K webcam, 78° | 356 | 178 | 119 | 89 | 71 | 59 |
| 1080p with a varifocal/zoom lens set to 30° | 537 | 269 | 179 | 134 | 107 | 90 |

- **Detection works down to about 18–24 px.** So with a normal webcam, students are found and tracked from 5–6 m.
- **Identification works from about 24 px, but gets more certain the more pixels there are.** Far students are identified early and confirmed with more frames (5 instead of 3), and the far thresholds are stricter.
- **Liveness needs about 48 px or more.** A far student is "recognised, come closer" until the anti-spoof check has a big enough face. That happens naturally as they walk in.
- **To *mark* students reliably at 5–6 m, use a 4K camera or a zoom/varifocal camera.** Aim the zoom at the gate lane. Set its field of view in Settings → Gate so the distance read-out is right.

Run `python evaluate.py --width <camera px> --fov <degrees>` to see the real numbers for your camera on your own students.

### Several students at once

- Each frame is searched for every face (up to dozens).
- Each person gets their own track and their own multi-frame decision. Four students walking in together are all identified in parallel, in one batch.
- If two students cross paths and the tracker swaps them, the next frame won't look like that track's face anymore. The track is reset instead of mixing two people.
- The HUD draws every tracked face with its own verdict and distance:
  - VERIFYING 2/3 · 3.4 m
  - RAHUL · COME CLOSER
  - name ✓
  - TEACHER CHECK
- Once a student is marked, their face is no longer re-checked; only detection continues. So a crowd gets cheaper as it is processed.

### How a student is recognised

| Stage | What it does | Why |
|---|---|---|
| Gate lane | The browser sends the centre square of the frame at full camera resolution. | Far faces need every pixel. |
| Detector | SCRFD-10G on every frame. Every 4th frame it also runs at double resolution (the far scan). Around each small face being tracked, it runs a cheap zoomed re-detect. | Near faces every frame; far faces down to ~18 px without paying for a big scan every frame. |
| Tracker | Keeps one id per person across frames. Several people at once. | Evidence from many frames builds up per student. |
| Alignment | A similarity transform to the ArcFace 112×112 template. | Removes tilt and scale. |
| Embedding | ArcFace ResNet-50 (WebFace600K, InsightFace `buffalo_l`). All undecided faces in a frame go through in one batch. | A strong open face model; batching makes crowds cheaper. |
| Gallery: resolution-matched | Each enrolment photo is stored three ways: as is, and as the gate camera would see it at about 48 px and 28 px (downsampled, sensor noise, JPEG). A far gate face is compared like with like. | In tests this raised the right student's score at 24–30 px from 0.50–0.68 to 0.72–0.82 (some faces gain less). |
| Gallery: robust score | A student's score is the mean of their **best 3 photos**; each photo counts as the best of its versions. | One lucky photo can't win; one bad photo can't lose. |
| Gallery: learns the gate | Very confident, close, live matches are kept: the newest 5 per student, stored in `data/adaptive.npz`. | The system gets used to your camera, lighting and the students' current look (haircuts, specs). |
| Liveness | Two anti-spoof models (MiniFASNetV2 + V1SE), only on faces big enough to judge (48 px+). | A printed photo or phone screen goes to teacher review. |
| Multi-frame fusion | Frames are averaged, weighted by quality (size, sharpness, pose), so frames taken as a student comes closer count most. Every frame also votes. | A single odd frame can't mark anyone. |
| Decision | **Match** only if all of these hold. | |
| | The fused score clears the threshold for that face size (near/far calibrated separately). | |
| | The student beats the runner-up by a clear margin (doubled for look-alikes). | |
| | They win most frame votes. | |
| | At least one real enrolment photo agrees. | |
| | The face was checked live. | |
| Otherwise | **Review** (a teacher taps ✓, which includes possible photo/screen), **Unknown**, or still **pending** (keep walking). | Nobody is marked silently. |

### What `enroll.py` checks for you

It writes `backend/data/enroll_report.csv` and flags:
- **Rejected photos:** no face, several faces, too small, blurry, or head turned away.
- **Misfiled photos:** a photo that doesn't resemble the rest of the student's folder, or looks more like a different student.
- **Thin galleries:** students with fewer than 8 usable photos.
- **Look-alikes:** siblings or twins. The gate is stricter between these students.
- **Calibration:** accept and review thresholds are set separately for near and far faces, from the school's own photos. Every photo (and its far versions) is tested against every other student. The report gives the genuine-pass rate and the impostor-pass rate for each.

### Measuring accuracy: `evaluate.py`

This works on your real photos:
1. For a sample of students, one photo is held out of the gallery.
2. That photo is shrunk to the size the face would be at 1–6 m through your camera, with webcam noise, blur and JPEG added.
3. It goes through the same detect → embed → multi-frame decision as the gate.

For each distance the output shows:
- the share of faces found
- the share ranked correctly
- the share marked automatically and correctly
- the share marked **wrongly**
- the share sent to review

Any wrong mark is listed by name, so you can check those photos.

### API

```
GET  /health                  enrolled students, photos, learned samples, device, thresholds
POST /frame                   multipart: image (a gate-lane camera frame), gate
     -> {faces: [{track, box, width, status: pending|match|review|unknown|spoof|too_far|low_quality,
                  student_id, score, margin, confidence, range: near|far, live, frames, need_frames}], far_scan, ms}
POST /reload                  reload after enrolling (enroll.py calls it)
GET  /photo/{admission_no}    best enrolment photo (used on datasheets)
GET  /student/{admission_no}  photo count, look-alikes
```

### Evidence photos

Every decision saves the face (with some surroundings):
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
