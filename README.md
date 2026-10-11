# Shrooms Macro

A small Windows desktop macro for SCROOMS. It reads the visible Roblox window and uses ordinary mouse and keyboard input. It does not inject code, read game memory, or modify game files.

## Download and use

Download `ShroomsMacro-1.0.11.exe` from [Releases](https://github.com/kamasix/Scrooms-Macro/releases). Place it in a writable folder and run it; Python is not required for the EXE.

1. Open SCROOMS in Roblox, stand where you want to catch, and equip your instrument.
2. Select the Roblox window in the macro. Refresh the list if necessary.
3. Place your cursor over a clear area of the game and press **F6**. Clicking Start gives you three seconds to position your cursor.
4. Press **F6** to pause or **F8** to stop immediately and release the macro's mouse press.

The window starts near the upper-left corner and can be dragged. Settings stay collapsed until needed.

Use **Auto sell** below Start/Stop to enable or disable selling. Turning it off keeps catching without inventory keys or sale clicks, preserves the saved sale interval, and shows the catch count without a sale threshold. Changing settings pauses the macro; press **F6** to resume.

Use **Join Discord** at the bottom of the window or [join the community here](https://discord.gg/rrcTR6pNnC). Opening the link pauses the macro and launches your browser.

## What it does

- Detects the rotating white pointer and narrow green **Perfect** segment. Two visible green remnants can identify an arc split by the pointer; a completely hidden target cannot be newly inferred. After stable tracking, a short timed click accounts for capture age and input latency, with both dispatch and estimated arrival inside Perfect. Without reliable motion history, it requires a fresh observed overlap.
- Captures only the verified ring area while tracking it, reducing screen-copy work. A missing ring or a changed window triggers a full-window capture again. Timed clicks expire before input dispatch and are interrupted by Pause/Stop.
- Recovers a moved ring anywhere in the visible game window through a throttled search, then tracks its verified region. Saved calibration remains the first search location.
- Matches Frequency and follows the moving catch field: **hold left mouse to move right; release to move left**. Measured relative velocity helps brake before overshooting the target and keep the dot near its middle while the target moves.
- Keeps the physical cursor still during minigames. Selling is the only stage that moves it to a detected button.
- When Auto sell is enabled, after each detected catch by default, presses the backtick/tilde key, detects **Sell All**, clicks it twice using fresh button detections, then presses backtick/tilde again to close inventory. The catch interval is adjustable in Settings.
- Waits 2.5 seconds after a detected catch for the result animation, then starts the sale. A persistent success label cannot block a sale that is due.
- Moves smoothly to sale buttons, lets the hover position settle, then holds the click for up to 80 ms. Inventory uses a physical backtick key press held for up to 100 ms. Stop interrupts movement, waits, and presses and releases owned inputs.
- Keeps observing the ring after missed revolutions or temporary recognition loss, releasing while the pointer or Perfect is missing. It pauses if Roblox loses focus, Windows input fails, or another stage times out. It preserves the catch count if it cannot find the sell button.

The sale counter records completed input sequences; it does not read the game's currency balance.

When upgrading from 1.0.5, the old default sale interval of 10 changes to 1. Custom intervals, calibration, and catch counts are preserved. Close the older version before starting the new EXE.

## Calibration

The default regions come from a widescreen layout and scale with the selected game's client area. Windows DPI and monitor offsets are handled in physical pixels, including monitors to the left of the primary display. Other aspect ratios, moved UI, or game updates can still require calibration.

Ring recovery searches only the selected game window and verifies the circle, narrow green segment, and white pointer at full resolution before tracking the new region. It does not change the saved calibration. Other minigame regions still use their saved calibration.

Open **Settings & calibration**, choose the relevant element, then show it in Roblox during the three-second countdown. Select it in the frozen image and press Enter to save or Escape to cancel. Use **Preview** to inspect recognition. **Diagnostics** saves images and recognition details locally; review and crop them before sharing.

Ring status distinguishes a missing pointer or target from a frame that became too old to click. Local diagnostics include recent worker frames with capture time, analysis time, frame age, capture mode and pixel count, recognized angles, scheduled waits, and click decisions. This short history is kept in memory and saved only when you request Diagnostics.

**Perfect correction** adjusts estimated input latency in milliseconds, starting at 8 ms. Increase it if clicks consistently land after the small Perfect segment. With at least three measured pointer movements and a stable target, the macro can schedule a nearby upcoming overlap instead of waiting for another screenshot to land exactly on green. It requires the projected pointer to be inside the narrow target throughout the dispatch-to-arrival interval, reserves a margin for measurement uncertainty, and never schedules a full revolution ahead. Clicks based on an expired frame or an expired input deadline are rejected.

If the sale has a separate confirmation button, calibrate **Sale confirmation**. The stock layout expects a second click on Sell All. Inventory uses the physical backtick key (scan code `0x29`, normally below Escape), independent of the typed character. The game's inventory binding needs to use this key.

If inventory does not open, `profiles/default/input-trace.json` records recent inventory requests and whether Windows accepted the press and release. An `accepted` entry confirms input dispatch, not the game's response. Status messages also distinguish waiting for Sell All after the key from waiting for confirmation after the first click. The trace remains local and contains no images or window identifiers.

## Run from source

Requires Windows 10/11, 64-bit Python 3.12, and a visible Roblox Player window.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe main.py
```

## Build the EXE

```powershell
.\build.ps1
```

The executable is written to `dist/ShroomsMacro.exe`. The build includes only application modules and bundled recognition templates. GitHub Actions also creates a Windows download artifact. The release workflow publishes a versioned EXE when the application version changes on `main`, when a version tag is pushed, or when started manually in Actions.

## Development checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Public checks use generated images and mocked Windows input. Original gameplay screenshots are deliberately excluded. Offline recognition and packaging checks do not establish reliable live gameplay on every machine; a narrow Perfect window can still be missed under low frame rates or capture latency.

## Local data and privacy

Settings, counts, custom calibration crops, and the bounded inventory input trace are created under `profiles/` beside the program. Optional snapshots go into `diagnostics/`. There is no network upload or telemetry in the application. These folders, logs, screenshots, and build intermediates are excluded from source control. No original gameplay screenshots or personal profiles are included in this repository.

## License

The application source is available under the [MIT license](LICENSE). Dependency and game-template attribution is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This is an independent community tool, not an official Roblox or SCROOMS product.
