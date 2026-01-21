#!/usr/bin/env python3
"""
Production Voice Assistant for Jetson
Single entry point - just run: python3 run.py

Usage:
    python3 run.py                    # Use default demo menu
    python3 run.py --menu menu.json   # Use custom menu file
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Run the voice assistant"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Restaurant Voice Assistant")
    parser.add_argument("--menu", default="data/menu.json", help="Path to menu JSON file")
    parser.add_argument("--tenant", default="demo_restaurant", help="Tenant ID")
    args = parser.parse_args()
    
    print("="*60)
    print("🤖 Restaurant Voice Assistant (Jetson Edition)")
    print("="*60)
    print(f"📁 Menu: {args.menu}")
    print(f"🏢 Tenant: {args.tenant}\n")
    
    # Import and run assistant
    from assistant import TextAssistant
    
    try:
        # Initialize assistant with tenant config
        assistant = TextAssistant(
            tenant_id=args.tenant,
            tenants_path="data/tenants"
        )
        
        print("🎤 Assistant ready! Type your orders (or 'quit' to exit)\n")
        print("Examples:")
        print("  - 'show me drinks'")
        print("  - 'add 2 coffees'")
        print("  - 'what's my total'")
        print("  - 'checkout'\n")
        
        # Run interactive mode
        assistant.run()
        
    except FileNotFoundError as e:
        print(f"❌ Configuration not found: {e}")
        print("\n� Tip: Make sure tenant config exists in data/tenants/")
        return 1
    except Exception as e:
        print(f"❌ Failed to start assistant: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
