
while True:
    print("\n======================================")
    print(" HOPE HEALTH CLINIC AI SMART ASSISTANT")
    print("======================================")
    print("1. Start a new session")
    print("2. Open a past session")
    print("3. Exit")
    print("--------------------------------------")

    choice = input("Choose an option (1-3): ").strip()

    if choice == "1":
        print("\n[INFO] Starting a new AI session...")

        try:
            # Initialize Gemini client if it does not exist
            if client is None:
                print("[INFO] Connecting to Gemini API...")
                client = create_gemini_client()

            # Verify that the client was created
            if client is not None:
                print("[SUCCESS] Gemini client initialized.")
                print("[INFO] Launching your AI assistant...")

                run_new_session(client, system_prompt)

            else:
                print("[ERROR] Gemini connection failed.")
                print("Please check your API key and internet connection.")

        except Exception as e:
            print(f"[ERROR] Unable to start AI session: {e}")

    elif choice == "2":
        print("\n[INFO] Opening saved sessions...")

        try:
            open_past_session()
        except Exception as e:
            print(f"[ERROR] Unable to open past session: {e}")

    elif choice == "3":
        print("\nThank you for using Hope Health Clinic AI Assistant.")
        print("Goodbye!")
        break

    else:
        print("\n[WARNING] Invalid selection.")
        print("Please enter 1, 2, or 3.")
