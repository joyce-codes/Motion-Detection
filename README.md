# BodyPlay — body motion → keyboard input

Turns your webcam into an arrow-key controller. Lean/turn left → **Left**. Lean/turn
right → **Right**. Head pops up → **Up**. Head drops → **Down**. The game only ever
sees ordinary key presses, so any game that takes arrow keys works, and new games are
one line of JSON away.

---

## 1. Run it (3 commands)

```bash
cd Move
python body_input_controller.py --install     # one time only
python body_input_controller.py --profile subway_surfer
```

A window opens. It starts **PAUSED** on purpose so it can't type into your own setup
while you get framed. Stand normally for a second, then press **P** to arm it.

Useful flags:

```bash
python body_input_controller.py --list-cameras   # which camera numbers work
python body_input_controller.py --camera 1       # use a different camera
python body_input_controller.py --armed          # skip the pause, start live
```

---

## 2. Keys inside the window

| Key | What it does |
|---|---|
| **P** | Arm / pause key output (the safety switch) |
| **C** | Calibrate — hold still 1.5 s facing forward; this is your new "neutral" |
| **R** | Full reset of tracking + neutral |
| **[** / **]** | Steering easier / harder |
| **-** / **=** | Jump easier / harder |
| **;** / **'** | Duck easier / harder |
| **,** / **.** | Fast-jump (velocity) easier / harder |
| **H** | Toggle hold-keys vs tap-only |
| **M** | Mirror on/off |
| **O** | Overlay on/off |
| **TAB** | Next game profile |
| **S** | Save everything you just tuned into the JSON |
| **Q** / **Esc** | Quit |

The three bars on screen are `steer`, `rise`, `duck`. The yellow tick is the trigger
level. Move, watch the bar, press `[` `-` `;` until your normal-sized movement crosses
the tick. Press **S**. That's the whole tuning process.

---

## 3. Test order (fastest to confirm)

1. **Notepad / any text box.** Set profile `wasd_runner` (TAB until it shows), arm with
   P, and lean. You should see `a` and `d` typed. That proves the whole chain works
   without any game involved.
2. **Geometry Dash (web or Steam).** Profile `geometry_dash` — jump is mapped to
   **Space**. Pop your head up to jump.
3. **Subway Surfers (web: poki.com / subwaysurfers-type browser ports).** Profile
   `subway_surfer`. Click the game once so it has keyboard focus, then play.
4. **Temple Run 2 / Sonic Dash / Minion Rush / Talking Tom Gold Run / Blades of Brim** —
   same arrow-key profiles, browser versions or an Android runtime on your PC.

Always click the game window once after arming so the keystrokes land there.

---

## 4. Playing on a phone or TV

The controller speaks "keyboard", so anything that accepts a keyboard accepts it:

- **Phone mirrored to the PC** (scrcpy for Android, QuickTime for iPhone display) — the
  PC keyboard drives the mirrored game.
- **Android runtime on the PC** (Google Play Games on PC, BlueStacks, Waydroid) — the
  mobile games run natively and take arrow keys directly.
- **Smart TV / console** — pair the PC as a Bluetooth keyboard, or run the browser
  versions on the TV's connected PC stick.
- **Phone-only, no PC** — the same logic ports to a phone app later: MediaPipe has
  Android/iOS builds, and the phone emits keys as a Bluetooth HID keyboard to whatever
  screen is running the game.

---

## 5. Every setting, in plain English

Edit `body_input_settings.json`. **Lower number = easier to trigger.**

| Setting | What it means |
|---|---|
| `camera_index` | Which camera. 0 is usually the built-in one. |
| `capture_width` / `capture_height` | `640 x 480` gives the **tallest** view on most webcams because 4:3 uses the whole sensor. 1280×720 crops the top and bottom — that's what cuts your head off. |
| `capture_fps` | Ask the camera for 60. Higher = snappier. |
| `mirror` | On = you move like a mirror (lean right → screen right). |
| `hold_keys` | On = steering keys stay held while you hold the lean. |
| `pulse_actions` | Actions that fire as a single tap instead of a hold. `["up","down"]` suits runners; try `["up"]` if ducks need holding. |
| `cooldown_sec` | Minimum gap between two taps of the same key. |
| `start_armed` | `true` = start sending keys the moment it opens. |
| `lean_sensitivity` | How far you must lean/turn/tilt. `0.13` default, `0.07` for tiny movements. |
| `jump_sensitivity` | How high the head must rise above neutral. |
| `jump_velocity` | How *fast* the head must rise — catches quick bobs even if you don't rise far. |
| `duck_sensitivity` | How far the head must drop. |
| `release_ratio` | Key lets go at this fraction of the trigger — stops flicker at the edge. |
| `lean_gain` / `yaw_gain` / `tilt_gain` | Weights for the three steering cues: body lean, head **turn**, head **tilt**. Set one to `0` to ignore it; raise one to `1.5` if that's your natural movement. |
| `smoothing_alpha` | `1.0` = zero smoothing / lowest latency. `0.6` is a good balance. |
| `scale_smoothing` | How quickly the "how big am I on screen" reference updates. |
| `gesture_frames_on/off` | How many frames a gesture must hold. `1` = instant, `3` = steadier. |
| `neutral_adaptation` | Slow auto-recentering so posture drift never breaks it. `0` to freeze your calibration. |
| `min_visibility` | Kept at `0.05` so missing shoulders/legs never stop tracking. |
| `game_profile` / `profiles` | Key maps per game. Copy a block, rename it, change the keys — new game added. |

---

## 6. Adding a new game

```json
"my_new_game": { "left": "a", "right": "d", "up": "space", "down": "s" }
```

Then `--profile my_new_game`, or press TAB until it appears. Supported key names:
arrow keys, `space`, `enter`, `shift`, `ctrl`, `alt`, `tab`, `esc`, and any single
letter or digit.

---

## 7. Shipping it to customers

```bash
python build_app.py
```

Produces `dist/BodyPlay.exe` (or `dist/BodyPlay` on macOS/Linux) — one file, model and
settings baked in, nothing else to download. Zip it with this README, sell the download
link, and ship updates by replacing the file.
