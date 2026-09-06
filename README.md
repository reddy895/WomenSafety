# 🛡️ CCTV Threat & Anomaly Surveillance System (Women's Safety)

An intelligent, real-time video surveillance and threat detection system engineered specifically to enhance women's safety in public spaces, transport hubs, and campus environments. The system continuously monitors CCTV video feeds, tracks human interactions, analyzes body motion and gestures, accurately distinguishes harmless social interactions from physical assaults, and automatically dispatches immediate emergency alerts.

---

## 🔄 System Architecture & Operational Workflow

The system operates as an end-to-end automated surveillance pipeline that ingests video, analyzes human motion in real time, evaluates threat dynamics, and broadcasts emergency alerts when violence or harassment occurs.

```
       +-------------------------------------------------------+
       |               Video Input Feed Ingestion              |
       |       (Live CCTV, Webcam, RTSP Stream, Video File)    |
       +---------------------------+---------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |             Pedestrian Tracking Engine                |
       |  - Continuous multi-person detection                  |
       |  - Unique tracking ID assignment across frames        |
       +---------------------------+---------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |             Skeletal Pose & Motion Analysis           |
       |  - 17-point skeletal landmark extraction              |
       |  - Real-time limb movement and coordinate tracking    |
       +---------------------------+---------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |           Proximity & Interaction Zone Check          |
       |  - Bounding box overlap & distance thresholding       |
       |  - Filters out isolated individuals                   |
       +---------------------------+---------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |          Spatiotemporal Kinematic Evaluation          |
       |  - Upper-limb strike velocity & acceleration vectors  |
       |  - Torso tilt, orientation, and reciprocal wrap angle |
       |  - Bounding box jitter (grappling & struggle metrics) |
       |  - Inter-person distance stability & approach speed   |
       +---------------------------+---------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |             Gesture & Anomaly Classification          |
       |                                                       |
       |   [ BENIGN GESTURES ]            [ THREAT BEHAVIOR ]  |
       |   - Friendly Hugs                - Violent Lunges     |
       |   - High-Fives & Claps           - Pushing & Grabbing |
       |   - Handshakes & Normal Walks    - Physical Altercation|
       |          |                               |            |
       |          v                               v            |
       |   [ Alarm Suppressed ]           [ Temporal Filter ]  |
       +------------------------------------------+------------+
                                                  |
                                                  v
       +-------------------------------------------------------+
       |           Temporal Persistence Verification           |
       |  - Ensures aggressive behavior persists >= 1.5s       |
       |  - Prevents false triggers from brief rapid gestures  |
       +---------------------------+---------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |              Automated Emergency Response             |
       |  - Watermarked evidence snapshot captured & archived  |
       |  - Instant emergency WhatsApp alert dispatched        |
       |  - Exact location coordinates & Google Maps link sent |
       |  - Real-time visual HUD updated with threat alerts    |
       +-------------------------------------------------------+
```

---

## ⚙️ How the System Works (Detailed Working Mechanism)

The system works through an integrated, multi-stage processing pipeline designed to provide immediate alerts while eliminating false alarms:

### 1. Video Ingestion & Stream Processing
The surveillance engine continuously ingests incoming video frames from a connected source:
- **Live Camera Feeds**: Real-time USB cameras or surveillance webcams.
- **RTSP Network Streams**: IP surveillance cameras deployed across public corridors, parking facilities, or transport stops.
- **Recorded Video Files**: Prerecorded surveillance footage for forensic analysis and system evaluation.

Each frame is decoded, timestamped, and passed through the processing pipeline in real time.

---

### 2. Pedestrian Detection & Persistent Tracking
When people appear in the surveillance field of view, the system detects each individual and establishes a persistent identity:
- Each person in the frame receives a distinct numeric tracking identifier.
- As individuals move, walk, or cross paths, their IDs remain consistent across consecutive frames.
- Bounding boxes and central coordinates are continuously calculated, establishing a continuous historical trajectory for every person in the scene.

---

### 3. Skeletal Pose & Landmark Extraction
To understand what people are doing rather than simply where they are, the system extracts 17 skeletal landmark points for each tracked individual in every frame:
- **Head & Face**: Eyes, ears, nose (used for head orientation).
- **Upper Limbs**: Shoulders, elbows, and wrists (critical for strike, reach, and gesture detection).
- **Torso & Core**: Left and right hips (used to evaluate balance, stance, and body lean).
- **Lower Limbs**: Knees and ankles (used for posture and movement tracking).

These landmarks provide the geometric coordinates needed to track arm movements, body tilts, and close physical interactions.

---

### 4. Proximity & Interaction Zone Triggering
Analyzing every person against every other person across a large crowd would be inefficient. The system optimizes processing by monitoring proximity:
- When individuals are safely separated by normal social distances, they remain classified as **Normal Pedestrians**, requiring minimal computational overhead.
- When two individuals enter close physical proximity (based on centroid distance or overlapping bounding boxes), the system activates the **Interaction Zone Analyzer** for that specific pair.
- A rolling temporal window of movement history is maintained for interacting pairs to analyze behavior over time.

---

### 5. Spatiotemporal Kinematic Feature Extraction
During close interaction, the system evaluates dynamic physical features across multiple consecutive frames:

- **Upper-Limb Acceleration & Velocity**: Calculates how rapidly wrists and elbows accelerate. Sudden spikes signify punches, strikes, shoves, or violent grabbing, whereas smooth, controlled movements characterize natural gestures.
- **Torso Lean & Postural Asymmetry**: Analyzes the angle of the upper body relative to the hips. Violent scuffles cause extreme torso angles and off-balance postures.
- **Bounding Box Jitter & Distortion**: Physical altercations (wrestling, pulling, tackling) cause bounding box dimensions to rapidly deform and oscillate, creating a high jitter metric.
- **Approach Velocity & Distance Stability**: Measures how quickly people approach each other and whether their mutual distance remains stable or fluctuates erratically during contact.
- **Wrist Rendezvous Distance & Elevation**: Evaluates the relative distance between both individuals' hands and their height relative to shoulder level.

---

### 6. Intelligent Gesture & Behavior Classification
A core challenge in video surveillance is preventing false alarms caused by benign human gestures like friendly hugs, high-fives, or handshakes. The system utilizes an intelligent classification model trained to distinguish harmless interactions from violent threats:

| Interaction Type | Observed Behavioral Patterns | System Action |
| :--- | :--- | :--- |
| **Normal Walking** | Steady walking pace, consistent inter-person distance, upright posture, standard arm swing. | **Normal**: Logged as routine scene activity. |
| **High-Five / Greeting** | Hands raised above shoulder level, brief hand rendezvous in the air, stable social standing distance, no torso grappling. | **Benign**: Automatically suppressed; threat score set to 0. |
| **Friendly Hug** | Smooth decelerating approach, stable embrace distance, low limb acceleration, stable upright torsos, reciprocal arm wrap without erratic struggle. | **Benign**: Identified as harmless social contact; alert suppressed. |
| **Physical Altercation & Assault** | High limb acceleration spikes, chaotic bounding box distortion, unstable fluctuating distance, aggressive lunging, or physical struggle. | **Threat**: High anomaly score calculated; escalated to verification stage. |

---

### 7. Temporal Persistence & Alert Verification
To prevent momentary rapid movements (such as swatting an insect, adjusting a backpack, or waving goodbye) from generating erroneous alerts:
- The system requires high threat scores to persist continuously over a configurable time window (typically 1.5 seconds or longer).
- If an altercation anomaly is sustained throughout this duration, the incident is formally flagged as a verified emergency event.
- If the rapid motion ceases before the persistence threshold is reached, the anomaly timer cools down and resets.

---

### 8. Automated Evidence Capture & Watermarking
Once an altercation is verified, the system immediately captures and preserves visual evidence:
- The exact video frame displaying the physical altercation is extracted at full resolution.
- An emergency header banner is stamped onto the image, embedding:
  - Incident classification label
  - Exact timestamp
  - Surveillance camera node ID
  - Geographic coordinates
- The annotated snapshot is saved to the local incident archive for audit trails and forensic review.

---

### 9. Emergency WhatsApp Notification Dispatch
Simultaneously with evidence capture, the system triggers automated emergency communications without requiring any manual action:
- **Background Dispatch**: Alerts are transmitted in background worker threads so that the live video surveillance loop continues running at full frame rate without lag.
- **Direct Emergency Message**: Dispatches an instant WhatsApp notification to designated emergency contacts, security desks, or guardians.
- **Actionable Location Details**: Each alert includes:
  - Clear emergency warning header
  - Camera identifier and surveillance location
  - Exact timestamp
  - Direct clickable Google Maps link pin-pointing the incident site for rapid first-responder navigation
- **Evidence Snapshot Delivery**: Attaches the annotated visual snapshot of the scene for instant verification.
- **Cooldown Safeguard**: An intelligent cooldown timer prevents notification flooding by consolidating repeated detections from the same incident into a controlled communication schedule.

---

### 10. Live Visual Telemetry & Security Dashboard
For on-duty security personnel monitoring the live feed, the system renders a comprehensive visual Heads-Up Display (HUD):
- **Dynamic Bounding Boxes**:
  - 🟢 **Green**: Normal pedestrian behavior.
  - 🟡 **Yellow**: Active interaction / close proximity under evaluation.
  - 🔴 **Flashing Red**: Verified threat or physical altercation in progress.
- **Skeletal Landmark Overlays**: Real-time display of tracked body poses and connection lines between joints.
- **Telemetry Overlay**: On-screen indicators showing current monitoring status, active tracking count, processing speed, and real-time threat gauge.

---

## 📱 Emergency Alert Format

When a threat is confirmed, the system immediately transmits an actionable notification formatted as follows:

```
⚠️ RED ALERT: Physical Altercation/Harassment Anomaly Detected!
📍 Location: Surveillance Node 01 (Lat: 12.9716, Lon: 77.5946)
🕒 Timestamp: 2026-09-04 10:45:00 UTC
🗺️ Google Maps: https://maps.google.com/?q=12.9716,77.5946
```

Along with the message, the watermarked evidence snapshot captured at the exact moment of detection is archived and sent for immediate visual assessment.

---

## 🖥️ Operating the Surveillance System

The system provides flexible modes of operation tailored for different monitoring needs:

### 1. Interactive Launch
Running the surveillance application opens an intuitive selector:
- **Mode 1 (Live Camera Feed)**: Connects to local webcams or video input hardware for live continuous monitoring.
- **Mode 2 (Video File Analysis)**: Ingests recorded video footage (`.mp4`, `.avi`, `.mkv`) for validation, testing, or historical forensic analysis.

### 2. Operational Modes
- **Interactive Visual Mode**: Displays the real-time visual preview window with color-coded bounding boxes, skeletal landmark overlays, and threat status indicators.
- **Headless Edge Mode**: Runs strictly in the background without launching graphical desktop windows, suitable for edge devices, central servers, or low-power embedded surveillance installations.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
