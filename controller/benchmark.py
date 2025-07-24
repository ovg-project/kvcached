import argparse
import curses
import json
import subprocess
import sys
import threading
import time
from collections import deque


def load_config(config_path="two_models_config.json"):
    """Load the models configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['models']
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in configuration file '{config_path}'.")
        sys.exit(1)


def run_benchmark_client(model_name, port, output_queue, stop_event):
    """Run a benchmark client and capture its output."""
    command = ["./start_client.sh", "sglang", str(port), model_name]

    output_queue.append(
        f"Starting benchmark for {model_name} with command: {' '.join(command)}"
    )

    try:
        process = subprocess.Popen(command,
                                   cwd="../engine_integration/benchmark",
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   bufsize=1,
                                   universal_newlines=True)

        output_queue.append(
            f"Process started for {model_name}, PID: {process.pid}")

        while not stop_event.is_set() and process.poll() is None:
            line = process.stdout.readline()
            if line:
                output_queue.append(line.rstrip())
            else:
                time.sleep(0.1)

        # Get any remaining output
        if process.poll() is not None:
            remaining_output = process.stdout.read()
            for line in remaining_output.split('\n'):
                if line.strip():
                    output_queue.append(line.rstrip())

        return_code = process.wait()
        output_queue.append(
            f"=== Benchmark completed for {model_name} with return code {return_code} ==="
        )

    except Exception as e:
        output_queue.append(
            f"Error running benchmark for {model_name}: {str(e)}")
        import traceback
        output_queue.append(f"Traceback: {traceback.format_exc()}")


def main_display(stdscr, models_config, port):
    """Main curses display function."""
    try:
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)  # Non-blocking input
        stdscr.timeout(100)  # Refresh every 100ms

        # Get screen dimensions
        height, width = stdscr.getmaxyx()

        if height < 10 or width < 40:
            stdscr.addstr(0, 0, "Terminal too small! Need at least 40x10")
            stdscr.refresh()
            stdscr.getch()
            return

        # Create windows for each model
        model_names = list(models_config.keys())
        if len(model_names) != 2:
            stdscr.addstr(
                0, 0, f"Error: Expected 2 models, found {len(model_names)}")
            stdscr.refresh()
            stdscr.getch()
            return

        # Split screen vertically
        mid_col = width // 2

        # Create windows
        win1 = stdscr.subwin(height - 1, mid_col - 1, 0, 0)
        win2 = stdscr.subwin(height - 1, width - mid_col, 0, mid_col)

        # Status line at bottom
        status_win = stdscr.subwin(1, width, height - 1, 0)

        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Headers
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Model 1
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Model 2
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)  # Errors

        # Output queues for each model
        output_queues = [deque(maxlen=1000), deque(maxlen=1000)]
        stop_events = [threading.Event(), threading.Event()]

        # Start benchmark threads - both use the same port
        model1_name = model_names[0]
        model2_name = model_names[1]

        thread1 = threading.Thread(target=run_benchmark_client,
                                   args=(model1_name, port, output_queues[0],
                                         stop_events[0]))
        thread2 = threading.Thread(target=run_benchmark_client,
                                   args=(model2_name, port, output_queues[1],
                                         stop_events[1]))

        thread1.start()
        thread2.start()

        # Display loop
        scroll_pos = [0, 0]  # Scroll positions for each window

        try:
            while True:
                # Clear windows
                win1.clear()
                win2.clear()
                status_win.clear()

                # Draw headers
                header1 = f" {model1_name} (Port: {port}) "
                header2 = f" {model2_name} (Port: {port}) "

                win1.addstr(0, 0, header1[:mid_col - 2],
                            curses.color_pair(2) | curses.A_BOLD)
                win2.addstr(0, 0, header2[:width - mid_col - 1],
                            curses.color_pair(3) | curses.A_BOLD)

                # Draw separator
                for i in range(height - 1):
                    stdscr.addch(i, mid_col - 1, '|')

                # Display output for each model
                windows = [win1, win2]
                max_lines = height - 3  # Leave space for header and bottom margin

                for i, (window,
                        queue) in enumerate(zip(windows, output_queues)):
                    lines = list(queue)
                    start_idx = max(0, len(lines) - max_lines - scroll_pos[i])
                    end_idx = start_idx + max_lines

                    for j, line in enumerate(lines[start_idx:end_idx]):
                        if j + 1 < max_lines:
                            # Truncate line to fit window
                            max_width = (mid_col -
                                         2) if i == 0 else (width - mid_col -
                                                            1)
                            display_line = line[:max_width - 1]

                            color = curses.color_pair(
                                2) if i == 0 else curses.color_pair(3)
                            if "error" in line.lower(
                            ) or "failed" in line.lower():
                                color = curses.color_pair(4)

                            try:
                                window.addstr(j + 1, 0, display_line, color)
                            except curses.error:
                                pass  # Ignore if we can't write to the position

                # Status line
                status = f"Press 'q' to quit | Port: {port} | Threads: {'Running' if thread1.is_alive() or thread2.is_alive() else 'Finished'}"
                status_win.addstr(0, 0, status[:width - 1],
                                  curses.color_pair(1))

                # Refresh all windows
                win1.refresh()
                win2.refresh()
                status_win.refresh()
                stdscr.refresh()

                # Handle input
                key = stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == curses.KEY_UP:
                    scroll_pos[0] = max(0, scroll_pos[0] - 1)
                elif key == curses.KEY_DOWN:
                    scroll_pos[0] = min(
                        len(output_queues[0]) - max_lines, scroll_pos[0] + 1)

                # Check if both threads are done
                if not thread1.is_alive() and not thread2.is_alive():
                    # Show final results for a bit longer
                    time.sleep(2)
                    break

        except KeyboardInterrupt:
            pass
        finally:
            # Stop threads gracefully
            for event in stop_events:
                event.set()

            # Wait for threads to finish
            if thread1.is_alive():
                thread1.join(timeout=5)
            if thread2.is_alive():
                thread2.join(timeout=5)

    except Exception as e:
        # If curses fails, print error and exit
        print(f"Curses error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run benchmark clients for two models')
    parser.add_argument('port', type=int, help='Server port to connect to')
    parser.add_argument(
        '--config',
        default='two_models_config.json',
        help='Configuration file path (default: two_models_config.json)')
    parser.add_argument('--no-curses',
                        action='store_true',
                        help='Run without curses interface (for debugging)')

    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    models_config = load_config(args.config)

    if len(models_config) != 2:
        print(
            f"Error: Configuration must contain exactly 2 models, found {len(models_config)}"
        )
        sys.exit(1)

    model_names = list(models_config.keys())
    print(
        f"Starting benchmark clients for models: {model_names[0]} and {model_names[1]}"
    )
    print(f"Connecting to server on port: {args.port}")

    if args.no_curses:
        print("Running in debug mode without curses...")
        # Simple debug mode without curses
        output_queues = [deque(maxlen=1000), deque(maxlen=1000)]
        stop_events = [threading.Event(), threading.Event()]

        thread1 = threading.Thread(target=run_benchmark_client,
                                   args=(model_names[0], args.port,
                                         output_queues[0], stop_events[0]))
        thread2 = threading.Thread(target=run_benchmark_client,
                                   args=(model_names[1], args.port,
                                         output_queues[1], stop_events[1]))

        thread1.start()
        thread2.start()

        try:
            while thread1.is_alive() or thread2.is_alive():
                for i, queue in enumerate(output_queues):
                    if queue:
                        line = queue.popleft()
                        print(f"[Model {i+1}] {line}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            for event in stop_events:
                event.set()

        thread1.join()
        thread2.join()
    else:
        print("Press 'q' to quit the display")
        time.sleep(2)  # Give user time to read

        # Start curses interface
        curses.wrapper(main_display, models_config, args.port)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
