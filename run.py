import time
from model.PlayerInputController import PlayerInputController


def repeat_foot_press(player_index: int = 1, server_url: str = "http://127.0.0.1:15702/"):
    controller = PlayerInputController(server_url=server_url, player_index=player_index)

    try:
        while True:
            # Step 1: Simulate button press (true)
            print("👣 Pressing foot button (true)")
            success = controller._send_input_state({
                "is_foot_button_just_pressed": True
            })
            if not success:
                print("❌ Failed to send 'press' input")

            # Step 2: Small delay to let engine detect the press
            time.sleep(0.05)

            # Step 3: Reset input (false)
            print("🔄 Clearing foot button (false)")
            controller.clear_input()

            # Step 4: Wait before repeating
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("🛑 Stopped by user")
        controller.clear_input()


if __name__ == "__main__":
    repeat_foot_press()
