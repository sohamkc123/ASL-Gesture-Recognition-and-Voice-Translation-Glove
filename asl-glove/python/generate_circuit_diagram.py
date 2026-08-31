"""
generate_circuit_diagram.py
---------------------------
Generate a compact wiring-accurate block/circuit diagram for the ASL glove.

Output:
  report_plots/circuit_diagram_compact.png

Run:
  python python/generate_circuit_diagram.py
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, x, y, w, h, title, body="", fc="#f8fbff", ec="#2f3b52"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=10, fontweight="bold")
    if body:
        ax.text(x + 0.015, y + h - 0.07, body, ha="left", va="top", fontsize=8.5, linespacing=1.25)


def wire(ax, x1, y1, x2, y2, label=None, color="#333", lw=1.5, ls="-"):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls)
    if label:
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(xm, ym + 0.008, label, ha="center", va="bottom", fontsize=8, color=color)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "report_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "circuit_diagram_compact.png")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Boxes
    add_box(
        ax, 0.04, 0.73, 0.21, 0.20,
        "Power Block",
        "Battery / Source\n-> Voltage Regulator\n\nReg OUT (5V) -> ESP32 VIN\nReg OUT (5V) -> DFPlayer VCC\nReg GND -> Common GND",
        fc="#fff7eb",
    )

    add_box(
        ax, 0.32, 0.20, 0.30, 0.62,
        "ESP32 Dev Board",
        "ADC Flex: 33, 32, 35, 34, 36\nTouch: 26, 23, 19, 18, 25, 27\nI2C MPU: SDA 21, SCL 22\nDFPlayer UART: TX2 17, RX2 16\nMode Button: GPIO12 (INPUT_PULLUP)",
        fc="#eaf4ff",
    )

    add_box(
        ax, 0.70, 0.61, 0.25, 0.30,
        "Sensor Block",
        "Flex Sensors x5 (voltage dividers)\nTouch Contacts x6 (digital)\nMPU6050 (I2C)",
        fc="#eefcf1",
    )

    add_box(
        ax, 0.70, 0.25, 0.25, 0.26,
        "Audio Block",
        "DFPlayer Mini\nSpeaker 8Ω\nVCC = 5V, GND common",
        fc="#f2efff",
    )

    add_box(
        ax, 0.70, 0.08, 0.25, 0.11,
        "Mode Switch",
        "Push Button\nGPIO12 -> button -> GND",
        fc="#fff0f5",
    )

    # Ground bus
    wire(ax, 0.06, 0.06, 0.96, 0.06, color="#444", lw=2.2)
    ax.text(0.51, 0.035, "COMMON GND", ha="center", va="top", fontsize=9, fontweight="bold")

    # Power wiring
    wire(ax, 0.25, 0.83, 0.32, 0.78, "5V -> VIN", color="#b36b00")
    wire(ax, 0.25, 0.77, 0.70, 0.37, "5V -> DFPlayer VCC", color="#b36b00")

    # GND drops
    wire(ax, 0.14, 0.73, 0.14, 0.06, color="#555")  # regulator gnd to bus
    wire(ax, 0.41, 0.20, 0.41, 0.06, color="#555")  # esp gnd
    wire(ax, 0.82, 0.25, 0.82, 0.06, color="#555")  # df gnd
    wire(ax, 0.80, 0.61, 0.80, 0.06, color="#555")  # sensor gnd
    wire(ax, 0.82, 0.08, 0.82, 0.06, color="#555")  # button gnd

    # Sensor to ESP wiring
    wire(ax, 0.70, 0.80, 0.62, 0.74, "Flex -> GPIO33,32,35,34,36", color="#1a7f37")
    wire(ax, 0.70, 0.73, 0.62, 0.66, "Touch -> GPIO26,23,19,18,25,27", color="#1a7f37")
    wire(ax, 0.70, 0.66, 0.62, 0.58, "MPU SDA->21, SCL->22, VCC->3V3", color="#1a7f37")

    # ESP to audio wiring
    wire(ax, 0.62, 0.44, 0.70, 0.42, "GPIO17 (TX2) -> DF RX", color="#5b3cc4")
    wire(ax, 0.70, 0.35, 0.62, 0.37, "DF TX -> GPIO16 (RX2)", color="#5b3cc4")

    # Button signal
    wire(ax, 0.70, 0.13, 0.62, 0.27, "GPIO12", color="#b42374")

    # Notes
    ax.text(
        0.04, 0.01,
        "Note: Regulator output must be 5V when connected to ESP32 VIN.\n"
        "All grounds must be common (ESP32, MPU6050, DFPlayer, sensors, button).",
        ha="left", va="bottom", fontsize=8.2, color="#333"
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
