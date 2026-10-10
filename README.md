# Shrooms Macro

A small Windows desktop macro for SCROOMS. It reads the visible Roblox window and uses ordinary mouse and keyboard input. It does not inject code, read game memory, or modify game files.

## Download and use

Download `ShroomsMacro-1.0.4.exe` from [Releases](https://github.com/kamasix/Scrooms-Macro/releases). Place it in a writable folder and run it; Python is not required for the EXE.

1. Open SCROOMS in Roblox, stand where you want to catch, and equip your instrument.
2. Select the Roblox window in the macro. Refresh the list if necessary.
3. Place your cursor over a clear area of the game and press **F6**. Clicking Start gives you three seconds to position your cursor.
4. Press **F6** to pause or **F8** to stop immediately and release the macro's mouse press.

The window starts near the upper-left corner and can be dragged. Settings stay collapsed until needed.

Use **Join Discord** at the bottom of the window or [join the community here](https://discord.gg/rrcTR6pNnC). Opening the link pauses the macro and launches your browser.

## What it does

- Detects the rotating white pointer and narrow green **Perfect** segment. A click requires an observed overlap in a fresh frame.
- Matches Frequency and follows the moving catch field: **hold left mouse to move right; release to move left**.
- Keeps the physical cursor still during minigames. Selling is the only stage that moves it to a detected button.
- After 10 detected catches by default, opens inventory with the backtick/tilde key and clicks **Sell All** twice, using a fresh button detection for each click.
- Pauses if Roblox loses focus, a control cannot be detected, or a stage times out. It preserves the catch count if it cannot find the sell button.

The sale counter records completed input sequences; it does not read the game's currency balance.

## Calibration

The default regions come from a widescreen layout and scale with the selected game's client area. Windows DPI and monitor offsets are handled in physical pixels, including monitors to the left of the primary display. Other aspect ratios, moved UI, or game updates can still require calibration.

Open **Settings & calibration**, choose the relevant element, then show it in Roblox during the three-second countdown. Select it in the frozen image and press Enter to save or Escape to cancel. Use **Preview** to inspect recognition. **Diagnostics** saves images and recognition details locally; review and crop them before sharing.

If the sale has a separate confirmation button, calibrate **Sale confirmation**. The stock layout expects a second click on Sell All. The inventory key uses the Windows OEM backtick/tilde key; keyboard layouts or game bindings can differ.

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

Settings, counts, and custom calibration crops are created under `profiles/` beside the program. Optional snapshots go into `diagnostics/`. There is no network upload or telemetry in the application. These folders, logs, screenshots, and build intermediates are excluded from source control. No original gameplay screenshots or personal profiles are included in this repository.

## License

The application source is available under the [MIT license](LICENSE). Dependency and game-template attribution is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This is an independent community tool, not an official Roblox or SCROOMS product.
