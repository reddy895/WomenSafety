"""
send_test_message.py
================================================================================
Utility script to immediately send a test WhatsApp message from 8618118952 to
7760489756 using the local headless Node.js WhatsApp microservice.
================================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import argparse
from pathlib import Path
import requests
from rich.console import Console

import config
import utils
from whatsapp_bot.sender import ensure_whatsapp_bot_running, send_direct_message

console = Console(highlight=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Send immediate WhatsApp text/alert via UAV WhatsApp Bot.")
    parser.add_argument(
        "--to",
        type=str,
        default="7760489756",
        help="Recipient phone number (e.g. 7760489756 or +917760489756)"
    )
    maps_link = f"https://maps.google.com/?q={config.CAMERA_LATITUDE:.4f},{config.CAMERA_LONGITUDE:.4f}"
    default_msg = (
        "🚨 [WOMEN SAFETY ALERT]: Verified High-Priority Message from Surveillance Node.\n"
        f"🕒 Timestamp: {utils.get_current_utc_timestamp()}\n"
        f"🗺️ Google Maps: {maps_link}"
    )
    parser.add_argument(
        "--message",
        type=str,
        default=default_msg,
        help="Message text to send."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to image file to attach."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    console.print("\n[bold cyan]==================================================================[/bold cyan]")
    console.print("[bold white] [WHATSAPP] WHATSAPP DISPATCH: UAV MICROSERVICE ENGINE[/bold white]")
    console.print("[bold cyan]==================================================================[/bold cyan]")
    console.print(f"Authorized Sender: [bold green]{config.SENDER_NUMBER}[/bold green]")
    console.print(f"Target Recipient : [bold yellow]{args.to}[/bold yellow]\n")

    console.print("[cyan]1. Checking WhatsApp Bot microservice status...[/cyan]")
    ensure_whatsapp_bot_running(auto_start=True)

    # Poll status for readiness
    is_ready = False
    user_info = None
    for attempt in range(8):
        try:
            res = requests.get(config.WHATSAPP_STATUS_URL, timeout=2.0)
            if res.status_code == 200:
                data = res.json()
                if data.get("ready"):
                    is_ready = True
                    user_info = data.get("user")
                    break
        except Exception:
            pass
        time.sleep(1.0)

    if not is_ready:
        console.print("[bold yellow]⚠️ WhatsApp Client is running but NOT logged in yet![/bold yellow]")
        console.print(f"Please open: [bold green]{config.WHATSAPP_QR_URL}[/bold green] in your browser and link your device (8618118952).")
        console.print("Waiting for login...")
        # Check if user links device
        for _ in range(15):
            try:
                res = requests.get(config.WHATSAPP_STATUS_URL, timeout=2.0)
                if res.status_code == 200 and res.json().get("ready"):
                    is_ready = True
                    user_info = res.json().get("user")
                    break
            except Exception:
                pass
            time.sleep(1.5)

    if is_ready:
        console.print(f"[bold green]✓ WhatsApp Client is READY and logged in as: {user_info or config.SENDER_NUMBER}[/bold green]")
    else:
        console.print("[bold red]Proceeding with send attempt (if authenticated in background)...[/bold red]")

    console.print(f"\n[cyan]2. Dispatching message immediately to {args.to}...[/cyan]")
    img_path = Path(args.image) if args.image else None
    result = send_direct_message(
        to_number=args.to,
        message=args.message,
        image_path=img_path
    )

    if result.success:
        console.print(f"[bold green]✓ SUCCESS! Message sent automatically from {config.SENDER_NUMBER} to {args.to} without interruption.[/bold green]")
        console.print(f"Engine: [white]{result.engine}[/white] | Message ID: [white]{result.message_sid}[/white]\n")
    else:
        console.print(f"[bold red]✗ Failed to send message:[/bold red] {result.error_message}")
        console.print(f"If unlinked, view QR Code at: [bold cyan]{config.WHATSAPP_QR_URL}[/bold cyan]\n")


if __name__ == "__main__":
    main()
